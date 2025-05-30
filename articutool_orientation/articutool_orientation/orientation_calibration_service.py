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
from geometry_msgs.msg import Quaternion as QuaternionMsg, Vector3 as Vector3Msg
from articutool_interfaces.srv import TriggerCalibration
from articutool_interfaces.msg import ImuCalibrationStatus

import numpy as np
from scipy.spatial.transform import Rotation as R
import threading  # For managing collection state safely if needed, though service calls are usually in own thread

import tf2_ros
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)
import time as py_time  # For sleep in service call


class OrientationCalibrationService(Node):
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

        # State for data collection during calibration
        self.is_collecting_for_calibration = False
        self.calibration_collection_start_time: Optional[Time] = None
        self.collected_imu_orientations: List[R] = []
        self.collected_imu_angular_velocities: List[np.ndarray] = []
        self.collection_lock = threading.Lock()  # To protect shared collection lists

        # Latest data from input IMU topic (for continuous publishing)
        self.latest_imu_input_msg: Optional[Imu] = None
        self.latest_R_FilterWorld_to_IMUframe: Optional[R] = None

        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_input_topic_param,
            self.imu_input_callback,
            10,
        )
        self.calibrated_imu_pub = self.create_publisher(
            Imu, self.imu_output_topic_param, 10
        )
        status_qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.calibration_status_pub = self.create_publisher(
            ImuCalibrationStatus,
            self.calibration_status_topic_param,
            status_qos_profile,
        )
        self.calibration_service = self.create_service(
            TriggerCalibration,
            "~/trigger_calibration",
            self.trigger_calibration_callback,
        )
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.get_logger().info("Orientation Calibration Service Node Started.")
        # ... (other log messages from original)

    def _declare_parameters(self):
        string_desc = ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        double_desc = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)
        int_desc = ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER)

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
        self.declare_parameter("tf_lookup_timeout_sec", 1.0, double_desc)

        # New parameters for robust calibration
        self.declare_parameter("calibration_sampling_duration_sec", 0.5, double_desc)
        self.declare_parameter(
            "min_samples_for_calibration", 10, int_desc
        )  # e.g., for 50Hz IMU, 0.5s = 25 samples
        self.declare_parameter(
            "max_angular_velocity_stillness_threshold_rps", 0.05, double_desc
        )  # rad/s

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

        self.calibration_sampling_duration_sec_param = self.get_parameter(
            "calibration_sampling_duration_sec"
        ).value
        self.min_samples_for_calibration_param = self.get_parameter(
            "min_samples_for_calibration"
        ).value
        self.max_angular_velocity_stillness_threshold_rps_param = self.get_parameter(
            "max_angular_velocity_stillness_threshold_rps"
        ).value
        self.get_logger().debug("Parameters loaded/reloaded.")

    def parameters_callback(self, params: list[Parameter]):
        success = True
        require_recalibration = False
        for param in params:
            if param.name in [
                "robot_base_frame",
                "articutool_mount_frame",
                "calibration_sampling_duration_sec",
                "min_samples_for_calibration",
                "max_angular_velocity_stillness_threshold_rps",
            ]:
                require_recalibration = (
                    True  # Or just log a warning that parameters changed
                )

        self._load_parameters()
        if require_recalibration:
            self.get_logger().warn(
                f"A calibration-critical parameter changed. Recalibration may be needed or behavior will change on next calibration."
            )
            # Optionally reset current calibration if parameters fundamental to it change
            # self.is_calibrated = False
            # self.R_RobotBase_to_FilterWorld_cal = None
            # self._publish_status()
        return SetParametersResult(successful=success)

    def imu_input_callback(self, msg: Imu):
        self.latest_imu_input_msg = msg
        current_R_FilterWorld_to_IMUframe: Optional[R] = None
        current_angular_velocity_rps: Optional[np.ndarray] = None

        try:
            q_in = msg.orientation
            # Basic validation (allow zero quat for uninitialized IMUs)
            if not (
                abs(q_in.w**2 + q_in.x**2 + q_in.y**2 + q_in.z**2 - 1.0)
                < self.EPSILON * 100
                or (q_in.w == 0 and q_in.x == 0 and q_in.y == 0 and q_in.z == 0)
            ):
                self.get_logger().warn(
                    "Input IMU quaternion not normalized and not zero.",
                    throttle_duration_sec=5.0,
                )
                # Decide if to proceed with potentially bad data or return

            current_R_FilterWorld_to_IMUframe = R.from_quat(
                [q_in.x, q_in.y, q_in.z, q_in.w]
            )
            current_angular_velocity_rps = np.array(
                [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            )
            self.latest_R_FilterWorld_to_IMUframe = current_R_FilterWorld_to_IMUframe

        except Exception as e:
            self.get_logger().error(
                f"Error processing input IMU orientation: {e}",
                throttle_duration_sec=5.0,
            )
            return

        # Store data if calibration is active
        with self.collection_lock:
            if (
                self.is_collecting_for_calibration
                and current_R_FilterWorld_to_IMUframe is not None
                and current_angular_velocity_rps is not None
            ):
                # Only collect if the service call has started collection
                if self.calibration_collection_start_time is not None and (
                    self.get_clock().now() - self.calibration_collection_start_time
                ) <= RCLPYDuration(
                    seconds=self.calibration_sampling_duration_sec_param + 0.1
                ):  # Add small buffer
                    self.collected_imu_orientations.append(
                        current_R_FilterWorld_to_IMUframe
                    )
                    self.collected_imu_angular_velocities.append(
                        current_angular_velocity_rps
                    )

        # Republish logic (always active, uses calibration if available)
        calibrated_imu_msg = Imu()
        calibrated_imu_msg.header.stamp = msg.header.stamp
        calibrated_imu_msg.header.frame_id = self.articutool_mount_frame_param
        calibrated_imu_msg.linear_acceleration = msg.linear_acceleration
        calibrated_imu_msg.linear_acceleration_covariance = (
            msg.linear_acceleration_covariance
        )
        calibrated_imu_msg.angular_velocity = msg.angular_velocity
        calibrated_imu_msg.angular_velocity_covariance = msg.angular_velocity_covariance
        calibrated_imu_msg.orientation_covariance = msg.orientation_covariance

        q_to_publish_scipy: R
        if (
            self.is_calibrated
            and self.R_RobotBase_to_FilterWorld_cal is not None
            and current_R_FilterWorld_to_IMUframe is not None
        ):
            R_RobotBase_to_IMUframe_calibrated = (
                self.R_RobotBase_to_FilterWorld_cal * current_R_FilterWorld_to_IMUframe
            )
            q_to_publish_scipy = R_RobotBase_to_IMUframe_calibrated
        elif current_R_FilterWorld_to_IMUframe is not None:
            q_to_publish_scipy = current_R_FilterWorld_to_IMUframe
        else:
            q_to_publish_scipy = R.from_quat([0, 0, 0, 1])  # Identity if no valid input

        q_out_xyzw = q_to_publish_scipy.as_quat()
        calibrated_imu_msg.orientation.x = q_out_xyzw[0]
        calibrated_imu_msg.orientation.y = q_out_xyzw[1]
        calibrated_imu_msg.orientation.z = q_out_xyzw[2]
        calibrated_imu_msg.orientation.w = q_out_xyzw[3]
        self.calibrated_imu_pub.publish(calibrated_imu_msg)

        # Publish status only if it changes or periodically (handled by service call end)
        # self._publish_status() # Publishing here might be too frequent

    def trigger_calibration_callback(
        self, request: TriggerCalibration.Request, response: TriggerCalibration.Response
    ):
        with self.collection_lock:  # Ensure only one calibration attempt at a time
            if (
                self.is_collecting_for_calibration
            ):  # Check if already in progress from another call
                response.success = False
                response.message = "Calibration collection already in progress."
                self.get_logger().warn(response.message)
                return response

            self.is_collecting_for_calibration = True
            self.calibration_collection_start_time = self.get_clock().now()
            self.collected_imu_orientations = []
            self.collected_imu_angular_velocities = []

        self.get_logger().info(
            f"Calibration triggered. Collecting data for {self.calibration_sampling_duration_sec_param} seconds. "
            "Ensure robot and Articutool IMU mount frame are STATIONARY."
        )

        # Wait for data collection period
        # The service call itself will block the BT, this loop makes the service call take time.
        collection_end_time = self.calibration_collection_start_time + RCLPYDuration(
            seconds=self.calibration_sampling_duration_sec_param
        )
        while self.get_clock().now() < collection_end_time:
            if not rclpy.ok():  # Check if node is shutting down
                response.success = False
                response.message = "Node shutting down during calibration."
                self.get_logger().warn(response.message)
                with self.collection_lock:
                    self.is_collecting_for_calibration = False
                return response
            py_time.sleep(0.05)  # Sleep briefly, allowing IMU callback to run

        with self.collection_lock:
            self.is_collecting_for_calibration = False  # Stop collection
            collected_orientations_copy = list(self.collected_imu_orientations)
            collected_ang_vels_copy = list(self.collected_imu_angular_velocities)

        self.get_logger().info(
            f"Collected {len(collected_orientations_copy)} IMU samples for calibration."
        )

        if len(collected_orientations_copy) < self.min_samples_for_calibration_param:
            response.success = False
            response.message = (
                f"Cannot calibrate: Insufficient IMU samples collected "
                f"({len(collected_orientations_copy)} < {self.min_samples_for_calibration_param}). "
                f"Check IMU topic and rate."
            )
            self.get_logger().error(response.message)
            self._publish_status()
            return response  # Publish uncalibrated status

        # Check for stillness using angular velocities
        avg_ang_vel_norm = 0.0
        if collected_ang_vels_copy:
            avg_ang_vel_norm = np.mean(
                [np.linalg.norm(vel) for vel in collected_ang_vels_copy]
            )

        if avg_ang_vel_norm > self.max_angular_velocity_stillness_threshold_rps_param:
            response.success = False
            response.message = (
                f"Calibration failed: IMU not considered still. "
                f"Avg ang vel norm: {avg_ang_vel_norm:.3f} rad/s "
                f"(Threshold: {self.max_angular_velocity_stillness_threshold_rps_param:.3f} rad/s)."
            )
            self.get_logger().warn(response.message)
            self.is_calibrated = False
            self.R_RobotBase_to_FilterWorld_cal = None
            self._publish_status()
            return response

        # Average the collected orientations
        try:
            # R.mean() requires scipy 1.6.0+
            R_FilterWorld_to_IMUframe_at_calibration = R.mean(
                R.concatenate(collected_orientations_copy)
            )
            self.get_logger().info(
                f"Averaged orientation from {len(collected_orientations_copy)} samples."
            )
        except Exception as e:
            response.success = False
            response.message = f"Failed to average IMU orientations: {e}"
            self.get_logger().error(response.message)
            self.is_calibrated = False
            self.R_RobotBase_to_FilterWorld_cal = None
            self._publish_status()
            return response

        try:
            transform_msg = self.tf_buffer.lookup_transform(
                self.robot_base_frame_param,
                self.articutool_mount_frame_param,
                rclpy.time.Time(seconds=0),
                timeout=RCLPYDuration(seconds=self.tf_lookup_timeout_sec_param),
            )
            q_tf = transform_msg.transform.rotation
            R_RobotBase_to_IMUframe_TF = R.from_quat([q_tf.x, q_tf.y, q_tf.z, q_tf.w])
            self.R_RobotBase_to_FilterWorld_cal = (
                R_RobotBase_to_IMUframe_TF
                * R_FilterWorld_to_IMUframe_at_calibration.inv()
            )
            self.is_calibrated = True
            self.last_calibration_time = self.get_clock().now()
            response.success = True
            response.message = "Calibration successful using averaged IMU data."
            offset_quat_xyzw = self.R_RobotBase_to_FilterWorld_cal.as_quat()
            response.computed_offset_jacobase_to_filterworld = QuaternionMsg(
                x=offset_quat_xyzw[0],
                y=offset_quat_xyzw[1],
                z=offset_quat_xyzw[2],
                w=offset_quat_xyzw[3],
            )
            self.get_logger().info(
                f"{response.message} Computed R_RobotBase_to_FilterWorld (xyzw): {offset_quat_xyzw}"
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            response.success = False
            response.message = f"Calibration failed: TF error: {e}"
            self.get_logger().error(response.message)
            self.is_calibrated = False
            self.R_RobotBase_to_FilterWorld_cal = None
        except Exception as e:
            response.success = False
            response.message = f"Calibration failed with unexpected error: {e}"
            self.get_logger().error(response.message)
            self.is_calibrated = False
            self.R_RobotBase_to_FilterWorld_cal = None
        finally:
            self._publish_status()
        return response

    def _publish_status(self):
        status_msg = ImuCalibrationStatus()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        status_msg.is_yaw_calibrated = self.is_calibrated
        if self.is_calibrated and self.last_calibration_time:
            status_msg.last_yaw_calibration_time = self.last_calibration_time.to_msg()
        else:
            status_msg.last_yaw_calibration_time = Time(
                seconds=0, nanoseconds=0
            ).to_msg()
        self.calibration_status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = OrientationCalibrationService()
        # Using MultiThreadedExecutor because service callbacks might block for a short period
        # while waiting for IMU samples, and we want the IMU subscriber to keep running.
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("Shutting down due to KeyboardInterrupt.")
    except Exception as e:
        logger = (
            node.get_logger()
            if node
            else rclpy.logging.get_logger("orientation_calibration_service_main")
        )
        logger.fatal(f"Unhandled exception: {e}\n{traceback.format_exc()}")
    finally:
        if node and node.executor is None:  # Should be handled by executor shutdown
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
