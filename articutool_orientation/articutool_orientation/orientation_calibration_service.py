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
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion as QuaternionMsg, Vector3 as Vector3Msg
from articutool_interfaces.srv import TriggerCalibration
from articutool_interfaces.msg import ImuCalibrationStatus

import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
import traceback

import tf2_ros
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)
import time as py_time


class OrientationCalibrationService(Node):
    EPSILON = 1e-6

    def __init__(self):
        super().__init__("orientation_calibration_service")

        self._declare_parameters()
        self._load_parameters()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.R_RobotBase_to_FilterWorld_cal: Optional[R] = None
        self.is_calibrated = False
        self.last_calibration_time: Optional[Time] = None

        self.is_collecting_for_calibration = False
        self.calibration_collection_start_time: Optional[Time] = None
        self.collected_imu_orientations: List[R] = []
        self.collected_imu_angular_velocities: List[np.ndarray] = []
        self.collection_lock = threading.Lock()

        self.latest_imu_input_msg: Optional[Imu] = None
        self.latest_R_FilterWorld_to_IMUframe: Optional[R] = None

        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_input_topic_param,
            self.imu_input_callback,
            rclpy.qos.qos_profile_sensor_data,
        )
        self.get_logger().info(f"Subscribed to IMU topic: {self.imu_input_topic_param}")

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

        self.service_callback_group = MutuallyExclusiveCallbackGroup()
        self.calibration_service = self.create_service(
            TriggerCalibration,
            "~/trigger_calibration",
            self.trigger_calibration_callback,
            callback_group=self.service_callback_group,
        )
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.get_logger().info("Orientation Calibration Service Node Started.")
        self.get_logger().info(
            f"Publishing calibrated IMU to: {self.imu_output_topic_param}"
        )
        self.get_logger().info(
            f"Publishing calibration status to: {self.calibration_status_topic_param}"
        )
        self.get_logger().info(f"Robot Base Frame: {self.robot_base_frame_param}")
        self.get_logger().info(
            f"Articutool IMU Mount Frame: {self.articutool_mount_frame_param}"
        )

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
        self.declare_parameter("calibration_sampling_duration_sec", 0.5, double_desc)
        self.declare_parameter("min_samples_for_calibration", 10, int_desc)
        self.declare_parameter(
            "max_angular_velocity_stillness_threshold_rps", 0.05, double_desc
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
        for param in params:
            if param.name in [
                "robot_base_frame",
                "articutool_mount_frame",
                "calibration_sampling_duration_sec",
                "min_samples_for_calibration",
                "max_angular_velocity_stillness_threshold_rps",
            ]:
                self.get_logger().info(
                    f"Calibration-critical parameter '{param.name}' changed. Recalibration may be needed."
                )
        self._load_parameters()
        return SetParametersResult(successful=success)

    def imu_input_callback(self, msg: Imu):
        self.get_logger().debug("IMU CB: Entered.")
        self.latest_imu_input_msg = msg
        current_R_FilterWorld_to_IMUframe: Optional[R] = None
        current_angular_velocity_rps: Optional[np.ndarray] = None

        try:
            q_in = msg.orientation
            if not (
                abs(q_in.w**2 + q_in.x**2 + q_in.y**2 + q_in.z**2 - 1.0)
                < self.EPSILON * 100
                or (q_in.w == 0 and q_in.x == 0 and q_in.y == 0 and q_in.z == 0)
            ):
                self.get_logger().warn(
                    "Input IMU quaternion not normalized and not zero.",
                    throttle_duration_sec=5.0,
                )

            current_R_FilterWorld_to_IMUframe = R.from_quat(
                [q_in.x, q_in.y, q_in.z, q_in.w]
            )
            current_angular_velocity_rps = np.array(
                [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            )
            self.latest_R_FilterWorld_to_IMUframe = current_R_FilterWorld_to_IMUframe
        except Exception as e:
            self.get_logger().error(
                f"Error processing input IMU orientation: {e} {traceback.format_exc()}",
                throttle_duration_sec=5.0,
            )
            return

        with self.collection_lock:
            if self.is_collecting_for_calibration:
                if self.calibration_collection_start_time is not None:
                    current_time_ros = self.get_clock().now()
                    elapsed_since_calib_start_ns = (
                        current_time_ros.nanoseconds
                        - self.calibration_collection_start_time.nanoseconds
                    )
                    elapsed_since_calib_start_s = elapsed_since_calib_start_ns / 1e9
                    collection_window_s = self.calibration_sampling_duration_sec_param
                    # Check if within the active collection window
                    if (
                        elapsed_since_calib_start_s >= 0
                        and elapsed_since_calib_start_s < collection_window_s
                    ):
                        if (
                            current_R_FilterWorld_to_IMUframe is not None
                            and current_angular_velocity_rps is not None
                        ):
                            self.collected_imu_orientations.append(
                                current_R_FilterWorld_to_IMUframe
                            )
                            self.collected_imu_angular_velocities.append(
                                current_angular_velocity_rps
                            )
                            self.get_logger().debug(
                                f"IMU CB: Sample collected. Total: {len(self.collected_imu_orientations)}"
                            )
                        # else: # This case should be rare if processing above is fine
                        #     self.get_logger().warn("IMU CB: Valid IMU data not available for collection this tick.")
                    # else: # Log if outside active window but still in "is_collecting" state
                    # self.get_logger().debug(f"IMU CB: is_collecting=True, but outside active window. Elapsed: {elapsed_since_calib_start_s:.3f}s / {collection_window_s:.3f}s")
                # else: # This would be a logic error if is_collecting_for_calibration is True
                #     self.get_logger().error("IMU CB: is_collecting_for_calibration is True, but calibration_collection_start_time is None!")

        # Republish logic (always active)
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
            q_to_publish_scipy = R.from_quat([0, 0, 0, 1])
        q_out_xyzw = q_to_publish_scipy.as_quat()
        calibrated_imu_msg.orientation.x = q_out_xyzw[0]
        calibrated_imu_msg.orientation.y = q_out_xyzw[1]
        calibrated_imu_msg.orientation.z = q_out_xyzw[2]
        calibrated_imu_msg.orientation.w = q_out_xyzw[3]
        self.calibrated_imu_pub.publish(calibrated_imu_msg)

    def trigger_calibration_callback(
        self, request: TriggerCalibration.Request, response: TriggerCalibration.Response
    ):
        with self.collection_lock:
            if self.is_collecting_for_calibration:
                response.success = False
                response.message = "Calibration collection already in progress."
                self.get_logger().warn(response.message)
                return response

            self.is_collecting_for_calibration = True
            self.calibration_collection_start_time = self.get_clock().now()
            self.collected_imu_orientations = []
            self.collected_imu_angular_velocities = []
            self.get_logger().info(  # Changed to INFO for better visibility
                f"trigger_calibration_callback: Initiating data collection. Start time: {self.calibration_collection_start_time.nanoseconds}"
            )

        self.get_logger().info(
            f"Calibration triggered. Collecting data for {self.calibration_sampling_duration_sec_param} seconds. Ensure robot is STATIONARY."
        )

        collection_start_pytime = (
            py_time.monotonic()
        )  # Using monotonic clock for sleep duration
        sampling_duration = self.calibration_sampling_duration_sec_param

        while (py_time.monotonic() - collection_start_pytime) < sampling_duration:
            if not rclpy.ok():
                response.success = False
                response.message = "Node shutting down during calibration."
                self.get_logger().warn(response.message)
                with self.collection_lock:
                    self.is_collecting_for_calibration = False
                    self.calibration_collection_start_time = None
                return response
            py_time.sleep(0.02)

        self.get_logger().info(
            "trigger_calibration_callback: Finished data collection sleep loop."
        )

        with self.collection_lock:
            self.is_collecting_for_calibration = False
            self.get_logger().info(
                "trigger_calibration_callback: Set is_collecting_for_calibration=False"
            )
            collected_orientations_copy = list(self.collected_imu_orientations)
            collected_ang_vels_copy = list(self.collected_imu_angular_velocities)
            # Reset start time here, after collection is definitively over and data copied
            self.calibration_collection_start_time = None
            self.get_logger().info(
                "trigger_calibration_callback: Reset calibration_collection_start_time."
            )

        self.get_logger().info(
            f"Collected {len(collected_orientations_copy)} IMU samples for calibration."
        )

        # ... (rest of the processing: sufficiency check, stillness check, averaging, TF lookup) ...
        if len(collected_orientations_copy) < self.min_samples_for_calibration_param:
            response.success = False
            response.message = (
                f"Cannot calibrate: Insufficient IMU samples collected "
                f"({len(collected_orientations_copy)} < {self.min_samples_for_calibration_param}). Check IMU topic and rate."
            )
            self.get_logger().error(response.message)
        else:
            avg_ang_vel_norm = 0.0
            if collected_ang_vels_copy:
                avg_ang_vel_norm = np.mean(
                    [np.linalg.norm(vel) for vel in collected_ang_vels_copy]
                )

            if (
                avg_ang_vel_norm
                > self.max_angular_velocity_stillness_threshold_rps_param
            ):
                response.success = False
                response.message = f"Calibration failed: IMU not considered still. Avg ang vel norm: {avg_ang_vel_norm:.3f} rad/s (Thresh: {self.max_angular_velocity_stillness_threshold_rps_param:.3f} rad/s)."
                self.get_logger().warn(response.message)
                self.is_calibrated = False
                self.R_RobotBase_to_FilterWorld_cal = None
            else:
                try:
                    R_FilterWorld_to_IMUframe_at_calibration = R.mean(
                        R.concatenate(collected_orientations_copy)
                    )
                    self.get_logger().info(
                        f"Averaged orientation from {len(collected_orientations_copy)} samples."
                    )

                    transform_msg = self.tf_buffer.lookup_transform(
                        self.robot_base_frame_param,
                        self.articutool_mount_frame_param,
                        rclpy.time.Time(seconds=0),
                        timeout=RCLPYDuration(seconds=self.tf_lookup_timeout_sec_param),
                    )
                    q_tf = transform_msg.transform.rotation
                    R_RobotBase_to_IMUframe_TF = R.from_quat(
                        [q_tf.x, q_tf.y, q_tf.z, q_tf.w]
                    )
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
                except (
                    LookupException,
                    ConnectivityException,
                    ExtrapolationException,
                ) as e:
                    response.success = False
                    response.message = f"Calibration failed: TF error: {e}"
                    self.get_logger().error(response.message)
                    self.is_calibrated = False
                    self.R_RobotBase_to_FilterWorld_cal = None
                except Exception as e:
                    response.success = False
                    response.message = f"Calibration failed with unexpected error: {e} {traceback.format_exc()}"
                    self.get_logger().error(response.message)
                    self.is_calibrated = False
                    self.R_RobotBase_to_FilterWorld_cal = None

        self._publish_status()  # Publish status regardless of success/failure of this attempt
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
        if node and node.executor is None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
