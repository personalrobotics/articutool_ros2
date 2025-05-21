#!/usr/bin/env python3

# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as RCLPYDuration
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion as QuaternionMsg
from articutool_interfaces.srv import TriggerCalibration
from articutool_interfaces.msg import ImuCalibrationStatus

import numpy as np
from scipy.spatial.transform import Rotation as R

import tf2_ros
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)


class OrientationCalibrationService(Node):
    """
    ROS 2 Node to provide an orientation calibration service for an IMU.
    It subscribes to raw/uncalibrated IMU data (orientation is typically
    gravity-aligned but with arbitrary yaw - relative to a "FilterWorld").
    Upon a service call, it computes a calibration offset to relate this
    FilterWorld to the robot's base frame.
    It then republishes IMU messages with the orientation transformed
    to be relative to the robot's base frame.
    """

    EPSILON = 1e-6

    def __init__(self):
        super().__init__("orientation_calibration_service")

        self._declare_parameters()
        self._load_parameters()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Calibration state
        self.R_RobotBase_to_FilterWorld_cal: Optional[R] = None
        self.is_calibrated = False
        self.last_calibration_time: Optional[Time] = None
        self.lock_calibration = False  # To prevent re-entry during service call

        # Latest data from input IMU topic
        self.latest_imu_input_msg: Optional[Imu] = None
        self.latest_R_FilterWorld_to_IMUframe: Optional[R] = None

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_input_topic_param,
            self.imu_input_callback,
            10,  # QoS depth
        )

        # Publishers
        self.calibrated_imu_pub = self.create_publisher(
            Imu,
            self.imu_output_topic_param,
            10,  # QoS depth
        )
        status_qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,  # Or BEST_EFFORT
        )
        self.calibration_status_pub = self.create_publisher(
            ImuCalibrationStatus,
            self.calibration_status_topic_param,
            status_qos_profile,  # QoS depth, latched-like via rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        )

        # Services
        self.calibration_service = self.create_service(
            TriggerCalibration,
            "~/trigger_calibration",
            self.trigger_calibration_callback,
        )

        self.add_on_set_parameters_callback(self.parameters_callback)

        self.get_logger().info("Orientation Calibration Service Node Started.")
        self.get_logger().info(f"Listening to IMU on: {self.imu_input_topic_param}")
        self.get_logger().info(
            f"Publishing calibrated IMU to: {self.imu_output_topic_param}"
        )
        self.get_logger().info(
            f"Publishing calibration status to: {self.calibration_status_topic_param}"
        )
        self.get_logger().info(f" Robot Base Frame: {self.robot_base_frame_param}")
        self.get_logger().info(
            f" Articutool IMU Mount Frame: {self.articutool_mount_frame_param}"
        )

    def _declare_parameters(self):
        string_desc = ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        self.declare_parameter(
            "imu_input_topic", "/articutool/imu_data_and_orientation", string_desc
        )
        self.declare_parameter(
            "imu_output_topic",
            "/articutool/imu_data_and_orientation_calibrated",
            string_desc,
        )
        self.declare_parameter("calibration_status_topic", "~/status", string_desc)
        self.declare_parameter("robot_base_frame", "j2n6s200_link_base", string_desc)
        self.declare_parameter("articutool_mount_frame", "atool_imu_frame", string_desc)
        self.declare_parameter(
            "tf_lookup_timeout_sec",
            1.0,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )

    def _load_parameters(self):
        self.imu_input_topic_param = self.get_parameter("imu_input_topic").value
        self.imu_output_topic_param = self.get_parameter("imu_output_topic").value
        self.calibration_status_topic_param = self.get_parameter(
            "calibration_status_topic"
        ).value
        self.robot_base_frame_param = self.get_parameter("robot_base_frame").value
        self.articutool_mount_frame_param = self.get_parameter(
            "articutool_mount_frame"
        ).value
        self.tf_lookup_timeout_sec_param = self.get_parameter(
            "tf_lookup_timeout_sec"
        ).value
        self.get_logger().debug("Parameters loaded/reloaded.")

    def parameters_callback(self, params: list[Parameter]):
        # For simplicity, require restart for most param changes or re-trigger calibration.
        # A more robust implementation might handle dynamic re-subscriptions if topics change.
        success = True
        require_recalibration = False
        for param in params:
            if param.name in ["robot_base_frame", "articutool_mount_frame"]:
                require_recalibration = True
            # Add more checks if other params need specific handling

        self._load_parameters()  # Reload all params

        if require_recalibration:
            self.get_logger().warn(
                f"Parameter '{param.name}' changed. Recalibration will be required."
            )
            self.is_calibrated = False
            self.R_RobotBase_to_FilterWorld_cal = None
            # Publish an updated status immediately
            self._publish_status()

        return SetParametersResult(successful=success)

    def imu_input_callback(self, msg: Imu):
        self.latest_imu_input_msg = msg

        # Validate and store the orientation from the input message
        try:
            q_in = msg.orientation
            if not (
                abs(q_in.w**2 + q_in.x**2 + q_in.y**2 + q_in.z**2 - 1.0)
                < self.EPSILON * 100
            ):  # Check if normalized
                if not (
                    q_in.w == 0 and q_in.x == 0 and q_in.y == 0 and q_in.z == 0
                ):  # Allow zero quat if not yet valid
                    self.get_logger().warn_throttle(
                        self.get_clock(),
                        5000,  # milliseconds
                        f"Input IMU quaternion not normalized: w={q_in.w} x={q_in.x} y={q_in.y} z={q_in.z}. Sum of squares: {q_in.w**2 + q_in.x**2 + q_in.y**2 + q_in.z**2}",
                    )
                    # Don't process further if quaternion is invalid and not zero
                    # return # Or handle as per requirements, e.g., use last known good

            self.latest_R_FilterWorld_to_IMUframe = R.from_quat(
                [q_in.x, q_in.y, q_in.z, q_in.w]
            )
        except Exception as e:
            self.get_logger().error_throttle(
                self.get_clock(), 5000, f"Error processing input IMU orientation: {e}"
            )
            return

        # Prepare output message
        calibrated_imu_msg = Imu()
        calibrated_imu_msg.header.stamp = msg.header.stamp
        calibrated_imu_msg.header.frame_id = (
            self.articutool_mount_frame_param
        )  # Output frame_id is the IMU's own frame

        # Copy raw sensor data
        calibrated_imu_msg.linear_acceleration = msg.linear_acceleration
        calibrated_imu_msg.linear_acceleration_covariance = (
            msg.linear_acceleration_covariance
        )
        calibrated_imu_msg.angular_velocity = msg.angular_velocity
        calibrated_imu_msg.angular_velocity_covariance = msg.angular_velocity_covariance
        calibrated_imu_msg.orientation_covariance = (
            msg.orientation_covariance
        )  # Usually needs adjustment if ref frame changes

        q_to_publish_scipy: R
        if (
            self.is_calibrated
            and self.R_RobotBase_to_FilterWorld_cal is not None
            and self.latest_R_FilterWorld_to_IMUframe is not None
        ):
            # Apply calibration: R_RobotBase_to_IMUframe = R_RobotBase_to_FilterWorld_cal * R_FilterWorld_to_IMUframe
            R_RobotBase_to_IMUframe_calibrated = (
                self.R_RobotBase_to_FilterWorld_cal
                * self.latest_R_FilterWorld_to_IMUframe
            )
            q_to_publish_scipy = R_RobotBase_to_IMUframe_calibrated
        else:
            # Pass through uncalibrated orientation (R_FilterWorld_to_IMUframe)
            if self.latest_R_FilterWorld_to_IMUframe is not None:
                q_to_publish_scipy = self.latest_R_FilterWorld_to_IMUframe
            else:  # Should not happen if latest_imu_input_msg is set
                q_to_publish_scipy = R.from_quat([0, 0, 0, 1])  # Identity

        q_out_xyzw = q_to_publish_scipy.as_quat()
        calibrated_imu_msg.orientation.x = q_out_xyzw[0]
        calibrated_imu_msg.orientation.y = q_out_xyzw[1]
        calibrated_imu_msg.orientation.z = q_out_xyzw[2]
        calibrated_imu_msg.orientation.w = q_out_xyzw[3]

        self.calibrated_imu_pub.publish(calibrated_imu_msg)
        self._publish_status()

    def trigger_calibration_callback(
        self, request: TriggerCalibration.Request, response: TriggerCalibration.Response
    ):
        if self.lock_calibration:
            response.success = False
            response.message = "Calibration already in progress or locked."
            self.get_logger().warn(response.message)
            return response

        self.lock_calibration = True
        self.get_logger().info(
            "Calibration triggered. Ensure robot and Articutool IMU mount frame are STATIONARY."
        )

        if (
            self.latest_R_FilterWorld_to_IMUframe is None
            or self.latest_imu_input_msg is None
        ):
            response.success = False
            response.message = (
                "Cannot calibrate: No fresh IMU data available from input topic."
            )
            self.get_logger().error(response.message)
            self.lock_calibration = False
            return response

        # Ensure the latest IMU message isn't too old
        # time_now = self.get_clock().now()
        # if (time_now - Time.from_msg(self.latest_imu_input_msg.header.stamp)) > RCLPYDuration(seconds=0.5):
        #     response.success = False
        #     response.message = "Cannot calibrate: Latest IMU data is too old."
        #     self.get_logger().error(f"{response.message} Last IMU stamp: {self.latest_imu_input_msg.header.stamp.sec}.{self.latest_imu_input_msg.header.stamp.nanosec}")
        #     self.lock_calibration = False
        #     return response

        R_FilterWorld_to_IMUframe_at_calibration = self.latest_R_FilterWorld_to_IMUframe

        try:
            # Lookup TF from RobotBase to IMUframe (articutool_mount_frame)
            # This transform should be for the same time as R_FilterWorld_to_IMUframe_at_calibration ideally,
            # but since the robot is static, Time(seconds=0) for latest TF is acceptable.
            transform_msg = self.tf_buffer.lookup_transform(
                self.robot_base_frame_param,
                self.articutool_mount_frame_param,
                rclpy.time.Time(seconds=0),  # Get the latest available transform
                timeout=RCLPYDuration(seconds=self.tf_lookup_timeout_sec_param),
            )

            q_tf = transform_msg.transform.rotation
            R_RobotBase_to_IMUframe_TF = R.from_quat([q_tf.x, q_tf.y, q_tf.z, q_tf.w])

            # Calculate calibration offset:
            # R_RobotBase_to_FilterWorld_cal = R_RobotBase_to_IMUframe_TF * (R_FilterWorld_to_IMUframe_at_calibration)^-1
            self.R_RobotBase_to_FilterWorld_cal = (
                R_RobotBase_to_IMUframe_TF
                * R_FilterWorld_to_IMUframe_at_calibration.inv()
            )

            self.is_calibrated = True
            self.last_calibration_time = (
                self.get_clock().now()
            )  # Record calibration time
            response.success = True
            response.message = "Calibration successful."

            # Populate the computed offset in the response for debugging
            offset_quat_xyzw = self.R_RobotBase_to_FilterWorld_cal.as_quat()
            response.computed_offset_jacobase_to_filterworld.x = offset_quat_xyzw[0]
            response.computed_offset_jacobase_to_filterworld.y = offset_quat_xyzw[1]
            response.computed_offset_jacobase_to_filterworld.z = offset_quat_xyzw[2]
            response.computed_offset_jacobase_to_filterworld.w = offset_quat_xyzw[3]

            self.get_logger().info(
                f"{response.message} Computed R_RobotBase_to_FilterWorld (xyzw): {offset_quat_xyzw}"
            )

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            response.success = False
            response.message = (
                f"Calibration failed: TF lookup error for '{self.robot_base_frame_param}' "
                f"to '{self.articutool_mount_frame_param}': {e}"
            )
            self.get_logger().error(response.message)
            self.is_calibrated = False  # Ensure calibration status is false on failure
            self.R_RobotBase_to_FilterWorld_cal = None
        except Exception as e:
            response.success = False
            response.message = f"Calibration failed with an unexpected error: {e}"
            self.get_logger().error(response.message)
            self.is_calibrated = False
            self.R_RobotBase_to_FilterWorld_cal = None
        finally:
            self.lock_calibration = False
            self._publish_status()  # Publish status after attempt

        return response

    def _publish_status(self):
        status_msg = ImuCalibrationStatus()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        status_msg.is_yaw_calibrated = (
            self.is_calibrated
        )  # is_yaw_calibrated reflects the overall calibration status
        if self.is_calibrated and self.last_calibration_time:
            status_msg.last_yaw_calibration_time = self.last_calibration_time.to_msg()
        else:
            # Use a zero time if not calibrated or time not set
            status_msg.last_yaw_calibration_time = Time(
                seconds=0, nanoseconds=0
            ).to_msg()
        # yaw_offset_rad field is removed/ignored as per clarification
        self.calibration_status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = OrientationCalibrationService()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("Shutting down due to KeyboardInterrupt.")
    except Exception as e:
        if node:
            node.get_logger().fatal(f"Unhandled exception: {e}")
        else:
            print(f"Unhandled exception before node init: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
