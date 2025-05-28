#!/usr/bin/env python3

# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import rclpy
from rclpy.impl.rcutils_logger import Throttle
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration as RCLPYDuration
from rclpy.executors import ExternalShutdownException
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult

from geometry_msgs.msg import Quaternion, QuaternionStamped, Vector3
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

from articutool_interfaces.srv import SetOrientationControl
from articutool_interfaces.msg import (
    ImuCalibrationStatus,
)

import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import pinocchio as pin


import os
import tempfile
import subprocess
from typing import Optional, Tuple, List
from ament_index_python.packages import get_package_share_directory

MODE_DISABLED = SetOrientationControl.Request.MODE_DISABLED
MODE_LEVELING = SetOrientationControl.Request.MODE_LEVELING
MODE_FULL_ORIENTATION = SetOrientationControl.Request.MODE_FULL_ORIENTATION


class OrientationControl(Node):
    WORLD_Z_UP_VECTOR = np.array([0.0, 0.0, 1.0])
    EPSILON = 1e-6
    JACOBIAN_PINV_RCOND = 1e-3

    # Fixed transform R_IMU_Handle, derived from URDF rpy="0 -pi/2 -pi" for Handle->IMU joint
    # R_Handle_IMU = Rz(-pi) * Ry(-pi/2) * Rx(0)
    # R_IMU_Handle = (R_Handle_IMU)^-1 = Rx(0)^-1 * Ry(pi/2) * Rz(pi)
    _R_handle_to_imu_urdf = R.from_euler(
        "xyz", [0, -math.pi / 2, -math.pi], degrees=False
    )
    R_IMU_TO_HANDLE_FIXED_SCIPY = _R_handle_to_imu_urdf.inv()

    def __init__(self):
        super().__init__("orientation_control")

        self._declare_parameters()
        self._load_parameters()

        self.pin_model: Optional[pin.Model] = None
        self.pin_data: Optional[pin.Data] = None
        self.imu_frame_id_pin: int = -1
        self.tooltip_frame_id_pin: int = -1
        self.articutool_joint_ids_pin: List[int] = []
        self.articutool_q_indices_pin: List[int] = []
        self.articutool_v_indices_pin: List[int] = []

        try:
            self._setup_pinocchio()
            self.get_logger().info("Pinocchio setup successful.")
        except Exception as e:
            self.get_logger().error(
                f"Pinocchio setup failed: {e}. MODE_FULL_ORIENTATION may be unavailable.",
            )
            self.pin_model = None

        self.current_mode = MODE_DISABLED
        self.target_orientation_jacobase: Optional[R] = (
            None  # For MODE_FULL_ORIENTATION
        )

        # Offsets for MODE_LEVELING (stored in RADIANS)
        self.current_pitch_offset_leveling: float = 0.0
        self.current_roll_offset_leveling: float = 0.0

        # Data from UNCALIBRATED IMU topic (for MODE_LEVELING and raw sensor values)
        self.last_uncalibrated_imu_msg_time: Optional[Time] = None
        self.current_filterworld_to_imu_raw: Optional[R] = (
            None  # R_FilterWorld_to_IMUframe
        )
        self.current_linear_accel_imu: Optional[np.ndarray] = None
        self.current_angular_velocity_imu: Optional[np.ndarray] = None

        # Data from CALIBRATED IMU topic (for MODE_FULL_ORIENTATION)
        self.last_calibrated_imu_msg_time: Optional[Time] = None
        self.current_RobotBase_to_IMUframe_calibrated: Optional[R] = (
            None  # R_RobotBase_to_IMUframe
        )

        # External calibration status
        self.is_externally_calibrated: bool = False
        self.last_external_calibration_time: Optional[Time] = None

        self.current_joint_positions: Optional[np.ndarray] = None
        self.last_error_leveling = np.zeros(3)
        self.integral_error_leveling = np.zeros(3)
        self.last_error_full_orientation = np.zeros(3)
        self.integral_error_full_orientation = np.zeros(3)
        self.last_time: Optional[Time] = None

        self.add_on_set_parameters_callback(self.parameters_callback)

        self.mode_srv = self.create_service(
            SetOrientationControl,
            "/articutool/set_orientation_control",
            self.set_orientation_control_callback,
        )

        # Subscriber for UNCALIBRATED IMU data (e.g., from imu_filter_madgwick)
        self.uncalibrated_imu_sub = self.create_subscription(
            Imu, self.uncalibrated_imu_topic, self.uncalibrated_feedback_callback, 1
        )
        # Subscriber for CALIBRATED IMU data (from OrientationCalibrationService)
        self.calibrated_imu_sub = self.create_subscription(
            Imu, self.calibrated_imu_topic, self.calibrated_feedback_callback, 1
        )
        # Subscriber for external calibration status
        self.calibration_status_sub = self.create_subscription(
            ImuCalibrationStatus,
            self.calibration_status_topic,
            self.calibration_status_callback,
            rclpy.qos.QoSProfile(
                depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
            ),  # Get last status
        )

        self.joint_state_sub = self.create_subscription(
            JointState, self.joint_state_topic, self.joint_state_callback, 10
        )
        self.cmd_pub = self.create_publisher(Float64MultiArray, self.command_topic, 10)

        if self.rate > 0:
            self.timer = self.create_timer(1.0 / self.rate, self.control_loop)
        else:
            self.get_logger().error(
                "Loop rate is zero or negative. Control loop will not run."
            )
            self.timer = None

        self.get_logger().info("Articutool Orientation Controller Node Started.")
        self.get_logger().info(
            f"MODE_LEVELING will use hardcoded R_IMU_TO_HANDLE_FIXED_SCIPY: {self.R_IMU_TO_HANDLE_FIXED_SCIPY.as_quat()} (xyzw)"
        )

    def _declare_parameters(self):
        p_gain_desc = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="PID Proportional gain"
        )
        i_gain_desc = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="PID Integral gain"
        )
        d_gain_desc = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, description="PID Derivative gain"
        )
        integral_clamp_desc = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description="Max absolute value for integral term",
        )
        loop_rate_desc = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description="Control loop frequency (Hz)",
        )
        string_desc = ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        str_array_desc = ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)
        dbl_array_desc = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)

        self.declare_parameter("pid_gains.p", 0.5, p_gain_desc)
        self.declare_parameter("pid_gains.i", 0.0, i_gain_desc)
        self.declare_parameter("pid_gains.d", 0.01, d_gain_desc)
        self.declare_parameter("integral_clamp", 0.5, integral_clamp_desc)
        self.declare_parameter("loop_rate", 50.0, loop_rate_desc)

        self.declare_parameter(
            "uncalibrated_imu_topic",
            "/articutool/imu_data_and_orientation",
            string_desc,
        )
        self.declare_parameter(
            "calibrated_imu_topic",
            "/articutool/imu_data_and_orientation_calibrated",
            string_desc,
        )
        self.declare_parameter(
            "calibration_status_topic",
            "/orientation_calibration_service/status",
            string_desc,
        )  # Adjust default if needed

        self.declare_parameter(
            "command_topic", "/articutool/velocity_controller/commands", string_desc
        )
        self.declare_parameter(
            "joint_state_topic", "/articutool/joint_states", string_desc
        )
        self.declare_parameter("urdf_path", "", string_desc)  # Path to XACRO or URDF
        self.declare_parameter("robot_base_frame", "j2n6s200_link_base", string_desc)
        self.declare_parameter(
            "articutool_base_link", "atool_handle", string_desc
        )  # Frame F0 for analytic Jacobian, also Pinocchio
        self.declare_parameter(
            "imu_link_frame", "atool_imu_frame", string_desc
        )  # For Pinocchio
        self.declare_parameter(
            "tooltip_frame", "tool_tip", string_desc
        )  # For Pinocchio
        self.declare_parameter(
            "joint_names", ["atool_joint1", "atool_joint2"], str_array_desc
        )
        self.declare_parameter(
            "joint_limits.lower", [-math.pi / 2.0, -math.pi], dbl_array_desc
        )
        self.declare_parameter(
            "joint_limits.upper", [math.pi / 2.0, math.pi], dbl_array_desc
        )
        self.declare_parameter(
            "joint_limits.threshold",
            0.1,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )
        self.declare_parameter(
            "joint_limits.dampening_factor",
            1.0,  # Power for dampening curve (1.0 = linear)
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )
        self.declare_parameter(
            "leveling_singularity_cos_roll_threshold",
            0.6,  # cos(roll_joint_angle) below which pitch cmd is dampened
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Cosine of roll angle threshold for detecting leveling mode pitch singularity.",
            ),
        )
        self.declare_parameter(
            "leveling_error_deadband_rad",
            0.015,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )
        self.declare_parameter(
            "leveling_singularity_damp_power",
            5.0,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )
        self.declare_parameter(
            "max_time_since_last_calibration_sec",
            30.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Max age of external calibration for FULL_ORIENTATION mode.",
            ),
        )

    def _load_parameters(self):
        self.Kp = self.get_parameter("pid_gains.p").value
        self.Ki = self.get_parameter("pid_gains.i").value
        self.Kd = self.get_parameter("pid_gains.d").value
        self.integral_max = self.get_parameter("integral_clamp").value
        self.rate = self.get_parameter("loop_rate").value

        self.uncalibrated_imu_topic = self.get_parameter("uncalibrated_imu_topic").value
        self.calibrated_imu_topic = self.get_parameter("calibrated_imu_topic").value
        self.calibration_status_topic = self.get_parameter(
            "calibration_status_topic"
        ).value

        self.command_topic = self.get_parameter("command_topic").value
        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.xacro_filename = self.get_parameter("urdf_path").value
        self.robot_base_frame = self.get_parameter("robot_base_frame").value
        self.articutool_base_link_name = self.get_parameter(
            "articutool_base_link"
        ).value
        self.imu_link_name = self.get_parameter("imu_link_frame").value
        self.tooltip_link_name = self.get_parameter("tooltip_frame").value
        self.articutool_joint_names = self.get_parameter("joint_names").value
        if len(self.articutool_joint_names) != 2:
            self.get_logger().fatal(
                f"Expected 2 joint_names, got {len(self.articutool_joint_names)}."
            )
            raise ValueError("Articutool joint_names must have 2 entries.")
        self.joint_limits_lower = np.array(
            self.get_parameter("joint_limits.lower").value
        )
        self.joint_limits_upper = np.array(
            self.get_parameter("joint_limits.upper").value
        )
        if len(self.joint_limits_lower) != 2 or len(self.joint_limits_upper) != 2:
            self.get_logger().fatal(
                f"Expected 2 joint_limits, got L:{len(self.joint_limits_lower)}, U:{len(self.joint_limits_upper)}."
            )
            raise ValueError("Articutool joint_limits must have 2 entries.")
        self.joint_limit_threshold = self.get_parameter("joint_limits.threshold").value
        self.joint_limit_dampening_factor = self.get_parameter(
            "joint_limits.dampening_factor"
        ).value
        self.leveling_singularity_cos_roll_threshold = self.get_parameter(
            "leveling_singularity_cos_roll_threshold"
        ).value
        self.leveling_error_deadband_rad = self.get_parameter(
            "leveling_error_deadband_rad"
        ).value
        self.leveling_singularity_damp_power = self.get_parameter(
            "leveling_singularity_damp_power"
        ).value
        self.max_time_since_last_calibration_sec = self.get_parameter(
            "max_time_since_last_calibration_sec"
        ).value
        self.get_logger().debug("Parameters loaded/reloaded.")

    def parameters_callback(self, params: List[Parameter]):
        accepted_params_names = []
        pinocchio_reload_needed = False

        for param in params:
            if param.name in [
                "pid_gains.p",
                "pid_gains.i",
                "pid_gains.d",
                "integral_clamp",
                "joint_limits.threshold",
                "joint_limits.dampening_factor",
                "loop_rate",
                "leveling_singularity_cos_roll_threshold",
                "leveling_error_deadband_rad",
                "leveling_singularity_damp_power",
                "max_time_since_last_calibration_sec",
            ]:
                accepted_params_names.append(param.name)
            # Parameters affecting Pinocchio model or critical frame names
            elif param.name in [
                "urdf_path",
                "articutool_base_link",
                "imu_link_frame",
                "tooltip_frame",
                "joint_names",
            ]:
                self.get_logger().warn(
                    f"Change to '{param.name}' typically requires node restart for Pinocchio model to be reloaded. Current model (if any) will NOT be reloaded dynamically."
                )
                # If you want to attempt dynamic reload, set pinocchio_reload_needed = True
                # and call _setup_pinocchio() again, but this is complex to do safely.
            elif param.name in [
                "uncalibrated_imu_topic",
                "calibrated_imu_topic",
                "calibration_status_topic",
                "robot_base_frame",
            ]:
                accepted_params_names.append(
                    param.name
                )  # Allow topic/frame name changes, will require re-sub/re-init if implemented
                self.get_logger().warn(
                    f"Parameter '{param.name}' changed. If it's a topic, subscriptions might need to be recreated (not implemented dynamically)."
                )

        if accepted_params_names:
            self._load_parameters()
            if "loop_rate" in accepted_params_names and self.timer is not None:
                self.timer.cancel()
                if self.rate > 0:
                    self.timer = self.create_timer(1.0 / self.rate, self.control_loop)
                else:
                    self.get_logger().error(
                        "Loop rate zero/negative. Control loop stopped."
                    )
            # If topic names changed, one would ideally destroy and recreate subscribers.
            # For simplicity, this example doesn't do that dynamically.
            return SetParametersResult(successful=True)
        return SetParametersResult(
            successful=False
        )  # If no accepted params were changed

    def _setup_pinocchio(self):
        if not self.xacro_filename:
            self.get_logger().warn("URDF path not provided. Pinocchio setup skipped.")
            self.pin_model = None
            return
        if not os.path.exists(self.xacro_filename):
            self.get_logger().error(
                f"Xacro/URDF file not found: {self.xacro_filename}."
            )
            self.pin_model = None
            return

        temp_urdf_path = None
        try:
            self.get_logger().info(
                f"Processing Xacro/URDF for Pinocchio: {self.xacro_filename}"
            )
            if self.xacro_filename.endswith(".xacro"):
                process = subprocess.run(
                    ["ros2", "run", "xacro", "xacro", self.xacro_filename],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                urdf_xml_string = process.stdout
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".urdf", delete=False
                ) as temp_urdf_file:
                    temp_urdf_path = temp_urdf_file.name
                    temp_urdf_file.write(urdf_xml_string)
                model_file_to_load = temp_urdf_path
            elif self.xacro_filename.endswith(".urdf"):
                model_file_to_load = self.xacro_filename
            else:
                self.get_logger().error(
                    f"Unsupported robot model file extension: {self.xacro_filename}. Must be .xacro or .urdf."
                )
                self.pin_model = None
                return

            self.pin_model = pin.buildModelFromUrdf(model_file_to_load)
            self.pin_data = self.pin_model.createData()
            self.get_logger().info(
                f"Pinocchio model loaded: {self.pin_model.name}, Nq={self.pin_model.nq}, Nv={self.pin_model.nv}"
            )

            # Get frame IDs crucial for control
            if self.pin_model.existFrame(self.imu_link_name):
                self.imu_frame_id_pin = self.pin_model.getFrameId(self.imu_link_name)
            else:
                self.get_logger().error(
                    f"Pinocchio: IMU frame '{self.imu_link_name}' not found!"
                )
                self.pin_model = None
                return

            if self.pin_model.existFrame(self.tooltip_link_name):
                self.tooltip_frame_id_pin = self.pin_model.getFrameId(
                    self.tooltip_link_name
                )
            else:
                self.get_logger().error(
                    f"Pinocchio: Tooltip frame '{self.tooltip_link_name}' not found!"
                )
                self.pin_model = None
                return

            self.articutool_joint_ids_pin = []
            self.articutool_q_indices_pin = []
            self.articutool_v_indices_pin = []
            for joint_name in self.articutool_joint_names:
                if not self.pin_model.existJointName(joint_name):
                    self.get_logger().error(
                        f"Articutool joint '{joint_name}' not found in Pinocchio model."
                    )
                    self.pin_model = None
                    return
                joint_id = self.pin_model.getJointId(joint_name)
                self.articutool_joint_ids_pin.append(joint_id)
                self.articutool_q_indices_pin.append(
                    self.pin_model.joints[joint_id].idx_q
                )
                self.articutool_v_indices_pin.append(
                    self.pin_model.joints[joint_id].idx_v
                )
            self.get_logger().info(
                f"Articutool Pinocchio joint IDs: {self.articutool_joint_ids_pin}, "
                f"q_indices: {self.articutool_q_indices_pin}, v_indices: {self.articutool_v_indices_pin}"
            )

        except Exception as e:
            self.get_logger().error(f"Pinocchio setup failed: {e}.")
            self.pin_model = None
        finally:
            if temp_urdf_path and os.path.exists(temp_urdf_path):
                try:
                    os.unlink(temp_urdf_path)
                except OSError as e_unlink:
                    self.get_logger().error(
                        f"Failed to delete temp URDF file {temp_urdf_path}: {e_unlink}"
                    )

    def set_orientation_control_callback(
        self,
        request: SetOrientationControl.Request,
        response: SetOrientationControl.Response,
    ):
        self.get_logger().info(
            f"Received SetOrientationControl Request: mode={request.control_mode}, "
            f"pitch_offset (deg)={request.pitch_offset:.2f}, roll_offset (deg)={request.roll_offset:.2f}"
        )
        response.success = False
        if request.control_mode == MODE_DISABLED:
            if self.current_mode != MODE_DISABLED:
                self._publish_zero_command()
            self.current_mode = MODE_DISABLED
            response.success = True
            response.message = "Orientation control disabled."
        elif request.control_mode == MODE_LEVELING:
            pitch_offset_rad = math.radians(request.pitch_offset)
            roll_offset_rad = math.radians(request.roll_offset)
            offsets_changed = (
                abs(self.current_pitch_offset_leveling - pitch_offset_rad)
                > self.EPSILON
                or abs(self.current_roll_offset_leveling - roll_offset_rad)
                > self.EPSILON
            )
            if self.current_mode != MODE_LEVELING or offsets_changed:
                self.get_logger().info(
                    f"{'Switching to' if self.current_mode != MODE_LEVELING else 'Updating'} LEVELING mode."
                )
                self._reset_pid_for_mode(MODE_LEVELING)
            self.current_pitch_offset_leveling = pitch_offset_rad
            self.current_roll_offset_leveling = roll_offset_rad
            self.current_mode = MODE_LEVELING
            response.success = True
            response.message = "Leveling mode (analytic Jacobian with offsets) enabled."

        elif request.control_mode == MODE_FULL_ORIENTATION:
            if self.pin_model is None:
                response.message = "Cannot enable FULL_ORIENTATION: Pinocchio model not loaded/failed to load."
                self.get_logger().error(response.message)
            elif not self.is_externally_calibrated:
                response.message = (
                    "Cannot enable FULL_ORIENTATION: System not externally calibrated."
                )
                self.get_logger().error(response.message)
            elif (
                self.last_external_calibration_time is not None
                and (
                    self.get_clock().now() - self.last_external_calibration_time
                ).nanoseconds
                / 1e9
                > self.max_time_since_last_calibration_sec
            ):
                response.message = (
                    f"Cannot enable FULL_ORIENTATION: External calibration is too old "
                    f"(> {self.max_time_since_last_calibration_sec}s). Please re-calibrate."
                )
                self.get_logger().error(response.message)
                self.is_externally_calibrated = False  # Mark as uncalibrated if too old
            else:
                try:
                    target_q_msg = request.target_orientation_robot_base
                    new_target = R.from_quat(
                        [target_q_msg.x, target_q_msg.y, target_q_msg.z, target_q_msg.w]
                    )
                    target_changed = (
                        self.target_orientation_jacobase is None
                        or not np.allclose(
                            new_target.as_quat(),
                            self.target_orientation_jacobase.as_quat(),
                            atol=1e-6,
                        )
                    )
                    if self.current_mode != MODE_FULL_ORIENTATION or target_changed:
                        self.get_logger().info(
                            f"{'Switching to' if self.current_mode != MODE_FULL_ORIENTATION else 'Updating'} FULL_ORIENTATION mode target."
                        )
                        self._reset_pid_for_mode(MODE_FULL_ORIENTATION)
                    self.target_orientation_jacobase = new_target
                    self.current_mode = MODE_FULL_ORIENTATION
                    response.success = True
                    response.message = "Full orientation mode enabled/updated."
                except Exception as e:
                    response.message = f"Error setting target for Full Orientation: {e}"
                    self.get_logger().error(response.message)

            if (
                not response.success and self.current_mode != MODE_DISABLED
            ):  # If any check failed for FULL_ORIENT
                self._publish_zero_command()
                self.current_mode = MODE_DISABLED  # Fallback to disabled
        else:
            response.message = f"Invalid control_mode: {request.control_mode}"
            self.get_logger().error(response.message)

        return response

    def uncalibrated_feedback_callback(self, msg: Imu):
        """Handles IMU data from the uncalibrated (FilterWorld-relative) topic."""
        self.last_uncalibrated_imu_msg_time = Time.from_msg(msg.header.stamp)
        is_valid_quat = not (
            math.isnan(msg.orientation.w)
            or (
                abs(msg.orientation.w) < self.EPSILON
                and abs(msg.orientation.x) < self.EPSILON
                and abs(msg.orientation.y) < self.EPSILON
                and abs(msg.orientation.z) < self.EPSILON
            )
        )
        if is_valid_quat:
            self.current_filterworld_to_imu_raw = R.from_quat(
                [
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w,
                ]
            )
        elif (
            self.current_filterworld_to_imu_raw is None
        ):  # Only log if it was never valid
            self.get_logger().warn(
                "Received zero or NaN quaternion from UNCALIBRATED IMU. Waiting for valid data.",
                throttle_duration_sec=1.0,
            )

        self.current_linear_accel_imu = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ]
        )
        self.current_angular_velocity_imu = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        )

    def calibrated_feedback_callback(self, msg: Imu):
        """Handles IMU data from the CALIBRATED (RobotBase-relative) topic."""
        self.last_calibrated_imu_msg_time = Time.from_msg(msg.header.stamp)
        is_valid_quat = not (
            math.isnan(msg.orientation.w)
            or (
                abs(msg.orientation.w) < self.EPSILON
                and abs(msg.orientation.x) < self.EPSILON
                and abs(msg.orientation.y) < self.EPSILON
                and abs(msg.orientation.z) < self.EPSILON
            )
        )
        if is_valid_quat:
            self.current_RobotBase_to_IMUframe_calibrated = R.from_quat(
                [
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w,
                ]
            )
        elif (
            self.current_RobotBase_to_IMUframe_calibrated is None
        ):  # Only log if it was never valid
            self.get_logger().warn(
                "Received zero or NaN quaternion from CALIBRATED IMU. Waiting for valid data.",
                throttle_duration_sec=1.0,
            )
        # Note: Linear accel and angular velocity from this topic are not explicitly stored separately here,
        # assuming the uncalibrated topic's values are sufficient for raw sensor data needs.

    def calibration_status_callback(self, msg: ImuCalibrationStatus):
        """Handles updates on the external calibration status."""
        status_changed = self.is_externally_calibrated != msg.is_yaw_calibrated
        self.is_externally_calibrated = msg.is_yaw_calibrated
        self.last_external_calibration_time = Time.from_msg(
            msg.last_yaw_calibration_time
        )

        if status_changed:
            self.get_logger().info(
                f"External calibration status changed. Calibrated: {self.is_externally_calibrated}"
            )

        if (
            not self.is_externally_calibrated
            and self.current_mode == MODE_FULL_ORIENTATION
        ):
            self.get_logger().error(
                "External calibration lost! Disabling FULL_ORIENTATION mode."
            )
            self.current_mode = MODE_DISABLED
            self._publish_zero_command()
            self._reset_pid_for_mode(MODE_FULL_ORIENTATION)  # Reset PID for this mode

    def joint_state_callback(self, msg: JointState):
        if self.current_joint_positions is None:
            self.current_joint_positions = np.zeros(len(self.articutool_joint_names))

        joint_map = {name: i for i, name in enumerate(self.articutool_joint_names)}
        for i, name in enumerate(msg.name):
            if name in joint_map:
                idx_in_our_array = joint_map[name]
                if i < len(msg.position):
                    self.current_joint_positions[idx_in_our_array] = msg.position[i]

    def control_loop(self):
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 1e-9:
            return

        # Prerequisites for MODE_LEVELING: uncalibrated IMU orientation and joint states
        mode_leveling_ready = (
            self.current_filterworld_to_imu_raw is not None
            and self.current_joint_positions is not None
        )

        # Prerequisites for MODE_FULL_ORIENTATION: Pinocchio, external calibration, target, calibrated IMU orientation, and joint states
        is_calibration_fresh = True
        if (
            self.is_externally_calibrated
            and self.last_external_calibration_time is not None
        ):
            if (
                now - self.last_external_calibration_time
            ).nanoseconds / 1e9 > self.max_time_since_last_calibration_sec:
                is_calibration_fresh = False
                if (
                    self.current_mode == MODE_FULL_ORIENTATION
                ):  # Log if actively in this mode and cal becomes stale
                    self.get_logger().warn(
                        "External calibration is too old for FULL_ORIENTATION mode.",
                        throttle_duration_sec=1.0,
                    )
        elif (
            self.is_externally_calibrated
            and self.last_external_calibration_time is None
        ):  # Calibrated but no timestamp? Should not happen.
            is_calibration_fresh = False
            self.get_logger().warn(
                "External calibration status indicates calibrated, but timestamp is missing.",
                throttle_duration_sec=5.0,
            )

        mode_full_orientation_ready = (
            self.pin_model is not None
            and self.is_externally_calibrated
            and is_calibration_fresh  # Check freshness
            and self.target_orientation_jacobase is not None
            and self.current_RobotBase_to_IMUframe_calibrated is not None
            and self.current_joint_positions is not None
        )

        if self.current_mode == MODE_DISABLED:
            self._publish_zero_command()
            return

        raw_commanded_dq: Optional[np.ndarray] = None
        final_commanded_dq: Optional[np.ndarray] = None

        try:
            if self.current_mode == MODE_LEVELING:
                if mode_leveling_ready:
                    raw_commanded_dq = (
                        self._calculate_leveling_control_analytic_jacobian(dt)
                    )
                else:
                    self.get_logger().warn(
                        "MODE_LEVELING: Prerequisites (Uncalibrated IMU/Joints) not met. Commanding zero.",
                        throttle_duration_sec=1.0,
                    )
            elif self.current_mode == MODE_FULL_ORIENTATION:
                if mode_full_orientation_ready:
                    raw_commanded_dq = self._calculate_full_orientation_control(dt)
                else:
                    self.get_logger().warn(
                        "MODE_FULL_ORIENTATION: Prerequisites (Pinocchio/ExternalCalib/Target/CalibratedIMU/Joints) not met. Commanding zero.",
                        throttle_duration_sec=1.0,
                    )

            if (
                raw_commanded_dq is not None
                and self.current_joint_positions is not None
            ):
                final_commanded_dq = self._enforce_joint_limits_predictive(
                    self.current_joint_positions, raw_commanded_dq, dt
                )
                self._publish_command(final_commanded_dq)
            elif raw_commanded_dq is not None:
                self.get_logger().warn(
                    "Cannot apply predictive joint limit enforcement: current_joint_positions are not available. Commanding zero.",
                    throttle_duration_sec=1.0,
                )
                self._publish_zero_command()
            else:
                self._publish_zero_command()

        except Exception as e:
            self.get_logger().error(
                f"Unhandled exception in control_loop (Mode: {self.current_mode}): {e}"
            )
            import traceback

            self.get_logger().error(traceback.format_exc())
            self._publish_zero_command()

    def _enforce_joint_limits_predictive(
        self, current_q: np.ndarray, desired_dq: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Enforces joint limits by predicting the next state and clamping velocity.
        """
        final_dq = np.copy(desired_dq)

        if dt <= self.EPSILON:  # Cannot predict if dt is effectively zero
            # Fallback: command zero if at or beyond limit and trying to move further
            # This is a basic safety, but ideally dt should always be positive and sensible.
            for i in range(len(self.articutool_joint_names)):
                q_i = current_q[i]
                dq_i = desired_dq[i]  # Use the original desired_dq for this check
                lower_limit = self.joint_limits_lower[i]
                upper_limit = self.joint_limits_upper[i]

                # If already at or beyond limit and trying to move further out, stop that joint's motion.
                if (dq_i < -self.EPSILON and q_i <= lower_limit + self.EPSILON) or (
                    dq_i > self.EPSILON and q_i >= upper_limit - self.EPSILON
                ):
                    final_dq[i] = 0.0
            return final_dq

        for i in range(len(self.articutool_joint_names)):
            q_i = current_q[i]
            # Use the desired_dq for prediction, which is final_dq[i] at this point in the loop iteration
            # if it hasn't been modified by the dt <= EPSILON block.
            # For clarity, let's use the original desired_dq for prediction.
            dq_i_desired = desired_dq[i]

            lower_limit = self.joint_limits_lower[i]
            upper_limit = self.joint_limits_upper[i]

            # Predict next position if this dq_i_desired is applied
            q_next_predicted = q_i + dq_i_desired * dt

            # Hard clamp the *predicted position* to the joint limits
            q_next_clamped = np.clip(q_next_predicted, lower_limit, upper_limit)

            # Recalculate the velocity to achieve this clamped position in the given dt
            # This new velocity command ensures that, for this dt,
            # the joint is commanded to move to a position just at or within its limits.
            final_dq[i] = (q_next_clamped - q_i) / dt

        return final_dq

    def _calculate_leveling_control_analytic_jacobian(
        self, dt: float
    ) -> Optional[np.ndarray]:
        if (
            self.current_filterworld_to_imu_raw is None
            or self.current_joint_positions is None
        ):
            return None
        try:
            theta_p_curr = self.current_joint_positions[0]
            theta_r_curr = self.current_joint_positions[1]

            cp, sp = math.cos(theta_p_curr), math.sin(theta_p_curr)
            cr, sr = math.cos(theta_r_curr), math.sin(theta_r_curr)

            phi_o = self.current_pitch_offset_leveling
            psi_o = self.current_roll_offset_leveling

            c_phi_o, s_phi_o = math.cos(phi_o), math.sin(phi_o)
            c_psi_o, s_psi_o = math.cos(psi_o), math.sin(psi_o)

            y_eff_x = -s_psi_o * c_phi_o
            y_eff_y = c_psi_o * c_phi_o
            y_eff_z = s_phi_o
            y_eff = np.array([y_eff_x, y_eff_y, y_eff_z])

            R_F0_tooltip_mat = np.array(
                [[cp * sr, cp * cr, -sp], [-cr, sr, 0], [sp * sr, sp * cr, cp]]
            )
            y_eff_in_F0 = R_F0_tooltip_mat @ y_eff
            R_FilterW_F0: R = (
                self.current_filterworld_to_imu_raw * self.R_IMU_TO_HANDLE_FIXED_SCIPY
            )
            y_eff_in_FilterW_curr = R_FilterW_F0.apply(y_eff_in_F0)
            target_y_tip_in_FilterW = self.WORLD_Z_UP_VECTOR

            axis_err_FilterW = np.cross(y_eff_in_FilterW_curr, target_y_tip_in_FilterW)
            dot_prod = np.dot(y_eff_in_FilterW_curr, target_y_tip_in_FilterW)
            angle_err = math.acos(np.clip(dot_prod, -1.0, 1.0))

            error_vec_FilterW = np.zeros(3)
            norm_axis_err_val = np.linalg.norm(axis_err_FilterW)

            if abs(angle_err) < self.leveling_error_deadband_rad:
                error_vec_FilterW = np.zeros(3)
            elif norm_axis_err_val > self.EPSILON:
                error_vec_FilterW = (axis_err_FilterW / norm_axis_err_val) * angle_err
            elif dot_prod < (-1.0 + self.EPSILON):  # 180 deg error
                arbitrary_axis = np.array([1.0, 0.0, 0.0])
                if (
                    np.linalg.norm(np.cross(y_eff_in_FilterW_curr, arbitrary_axis))
                    < self.EPSILON
                ):
                    arbitrary_axis = np.array([0.0, 1.0, 0.0])
                rotation_axis = np.cross(y_eff_in_FilterW_curr, arbitrary_axis)
                if np.linalg.norm(rotation_axis) > self.EPSILON:
                    error_vec_FilterW = (
                        rotation_axis / np.linalg.norm(rotation_axis)
                    ) * math.pi
                else:
                    error_vec_FilterW = np.array([math.pi, 0.0, 0.0])

            if np.allclose(
                error_vec_FilterW, 0.0, atol=self.EPSILON
            ):  # Check if error is effectively zero
                omega_corr_FilterW = np.zeros(3)
                # Reset integral if error is zero to prevent windup when not needed
                self.integral_error_leveling.fill(0.0)
            else:
                self.integral_error_leveling += error_vec_FilterW * dt
                self.integral_error_leveling = np.clip(
                    self.integral_error_leveling, -self.integral_max, self.integral_max
                )
                derivative_error = (
                    (error_vec_FilterW - self.last_error_leveling) / dt
                    if dt > self.EPSILON
                    else np.zeros(3)
                )
                omega_corr_FilterW = (
                    self.Kp * error_vec_FilterW
                    + self.Ki * self.integral_error_leveling
                    + self.Kd * derivative_error
                )
            self.last_error_leveling = np.copy(error_vec_FilterW)

            omega_corr_in_F0 = R_FilterW_F0.inv().apply(omega_corr_FilterW)

            dR_dtheta_p_mat = np.array(
                [[-sp * sr, -sp * cr, -cp], [0, 0, 0], [cp * sr, cp * cr, -sp]]
            )
            dR_dtheta_r_mat = np.array(
                [[cp * cr, -cp * sr, 0], [sr, cr, 0], [sp * cr, -sp * sr, 0]]
            )

            J_col1 = dR_dtheta_p_mat @ y_eff
            J_col2 = dR_dtheta_r_mat @ y_eff
            J_eff_in_F0 = np.column_stack((J_col1, J_col2))

            if J_eff_in_F0.shape != (3, 2):
                self.get_logger().error(
                    f"Leveling Jacobian shape error: {J_eff_in_F0.shape}. Expected (3,2). Commanding zero."
                )
                return np.zeros(len(self.articutool_joint_names))

            try:
                J_eff_in_F0_pinv = np.linalg.pinv(
                    J_eff_in_F0, rcond=self.JACOBIAN_PINV_RCOND
                )
            except np.linalg.LinAlgError:
                self.get_logger().warn(
                    "Leveling control (Analytic with offsets): Jacobian pseudo-inverse failed. Commanding zero.",
                    throttle_duration_sec=1.0,
                )
                return np.zeros(len(self.articutool_joint_names))

            desired_y_eff_linear_velocity_in_F0 = np.cross(
                omega_corr_in_F0, y_eff_in_F0
            )
            dq_calculated = J_eff_in_F0_pinv @ desired_y_eff_linear_velocity_in_F0

            pitch_singularity_scale = 1.0
            abs_cr = abs(cr)
            effective_singularity_threshold = max(
                self.leveling_singularity_cos_roll_threshold, self.EPSILON
            )
            if abs_cr < effective_singularity_threshold:
                normalized_cr_for_scaling = abs_cr / effective_singularity_threshold
                pitch_singularity_scale = (
                    np.clip(normalized_cr_for_scaling, 0.0, 1.0)
                    ** self.leveling_singularity_damp_power
                )
                if pitch_singularity_scale < 0.99:
                    self.get_logger().warn(
                        f"Pitch singularity damp: scale={pitch_singularity_scale:.3f}, cr={cr:.3f}",
                        throttle_duration_sec=1.0,
                    )
                dq_calculated[0] *= pitch_singularity_scale

            return dq_calculated

        except Exception as e:
            self.get_logger().error(f"Error in Analytic Jacobian Leveling Control: {e}")
            import traceback

            self.get_logger().error(traceback.format_exc())
            return np.zeros(len(self.articutool_joint_names))  # Return zero on error

    def _calculate_full_orientation_control(self, dt: float) -> Optional[np.ndarray]:
        if self.pin_model is None:
            self.get_logger().error(
                "Full Orient: Pinocchio model NA", throttle_duration_sec=1.0
            )
            return None
        if not self.is_externally_calibrated:
            self.get_logger().error(
                "Full Orient: Not externally calibrated!", throttle_duration_sec=1.0
            )
            return None
        if self.target_orientation_jacobase is None:
            self.get_logger().warn("Full Orient: Target NA", throttle_duration_sec=1.0)
            return None
        if self.current_RobotBase_to_IMUframe_calibrated is None:
            self.get_logger().warn(
                "Full Orient: Calibrated IMU data NA", throttle_duration_sec=1.0
            )
            return None
        if self.current_joint_positions is None:
            self.get_logger().warn(
                "Full Orient: Joint states NA", throttle_duration_sec=1.0
            )
            return None

        try:
            q_jb_imu = self.current_RobotBase_to_IMUframe_calibrated
            q_pin_config = self._get_pinocchio_config()
            R_imu_tooltip_current_pin = self._get_pinocchio_imu_tooltip_orientation(
                q_pin_config
            )
            if R_imu_tooltip_current_pin is None:
                self.get_logger().error(
                    "Full Orient: Failed to get R_IMU_Tooltip from Pinocchio."
                )
                return None

            q_jb_tooltip_current = q_jb_imu * R_imu_tooltip_current_pin

            q_error_jacobase = (
                self.target_orientation_jacobase * q_jb_tooltip_current.inv()
            )
            error_vec_jacobase = q_error_jacobase.as_rotvec()

            if np.allclose(
                error_vec_jacobase, 0.0, atol=self.EPSILON
            ):  # Check if error is effectively zero
                omega_desired_jacobase = np.zeros(3)
                # Reset integral if error is zero
                self.integral_error_full_orientation.fill(0.0)
            else:
                self.integral_error_full_orientation += error_vec_jacobase * dt
                self.integral_error_full_orientation = np.clip(
                    self.integral_error_full_orientation,
                    -self.integral_max,
                    self.integral_max,
                )
                derivative_error = (
                    (error_vec_jacobase - self.last_error_full_orientation) / dt
                    if dt > self.EPSILON
                    else np.zeros(3)
                )
                omega_desired_jacobase = (
                    self.Kp * error_vec_jacobase
                    + self.Ki * self.integral_error_full_orientation
                    + self.Kd * derivative_error
                )
            self.last_error_full_orientation = np.copy(error_vec_jacobase)

            omega_desired_tooltip_local = q_jb_tooltip_current.inv().apply(
                omega_desired_jacobase
            )

            dq_desired = self._calculate_joint_velocities_pinocchio_jacobian(
                q_pin_config, omega_desired_tooltip_local
            )
            if dq_desired is None:
                self.get_logger().error(
                    "Full Orient: Failed to calculate joint velocities from Pinocchio Jacobian."
                )
                return None

            return dq_desired
        except Exception as e:
            self.get_logger().error(
                f"Error in Full Orientation Control (Pinocchio Jacobian): {e}"
            )
            import traceback

            self.get_logger().error(traceback.format_exc())
            return None  # Return None on error

    def _get_pinocchio_config(self) -> np.ndarray:
        if self.pin_model is None:
            # This should ideally not be reached if checks are done prior to calling.
            self.get_logger().error("_get_pinocchio_config called with no model.")
            raise ValueError("Pinocchio model not loaded")
        if self.current_joint_positions is None:
            self.get_logger().error(
                "_get_pinocchio_config called with no joint positions."
            )
            raise ValueError("Joint positions are None")
        if len(self.current_joint_positions) != len(self.articutool_joint_names):
            self.get_logger().error(
                f"Joint position/name mismatch: {len(self.current_joint_positions)} vs {len(self.articutool_joint_names)}"
            )
            raise ValueError(f"Mismatch joint positions vs names")

        q = pin.neutral(self.pin_model)  # Start with a neutral configuration

        # Ensure q is large enough for all joint configurations.
        # This is a defensive check; pin.neutral should handle it for typical models.
        if self.pin_model.nq > len(q):
            # This case might indicate an issue with pin.neutral or model complexity not handled.
            self.get_logger().warn(
                f"Pinocchio model nq ({self.pin_model.nq}) > len(neutral_q) ({len(q)}). Expanding q."
            )
            q_expanded = np.zeros(self.pin_model.nq)
            q_expanded[: len(q)] = q
            q = q_expanded

        for i, joint_name in enumerate(self.articutool_joint_names):
            if not self.pin_model.existJointName(joint_name):
                # This should be caught earlier, but defensive check.
                self.get_logger().error(
                    f"Joint '{joint_name}' not in Pinocchio model during config creation."
                )
                raise ValueError(f"Joint '{joint_name}' not in Pinocchio model")

            joint_id = self.pin_model.getJointId(joint_name)
            joint_model = self.pin_model.joints[joint_id]
            idx_q = joint_model.idx_q
            nq_joint = joint_model.nq

            current_joint_val = self.current_joint_positions[i]

            if nq_joint == 1:
                if idx_q < len(q):
                    q[idx_q] = current_joint_val
                else:
                    self.get_logger().error(
                        f"idx_q {idx_q} out of bounds for q (len {len(q)}) for joint {joint_name}"
                    )
                    raise IndexError(f"idx_q out of bounds for joint {joint_name}")
            elif nq_joint == 2:
                if idx_q + 1 < len(q):
                    q[idx_q] = math.cos(current_joint_val)
                    q[idx_q + 1] = math.sin(current_joint_val)
                else:
                    self.get_logger().error(
                        f"idx_q+1 {idx_q + 1} out of bounds for q (len {len(q)}) for joint {joint_name}"
                    )
                    raise IndexError(f"idx_q+1 out of bounds for joint {joint_name}")
            else:
                # Handle other joint types if necessary, or log a warning/error
                self.get_logger().warn(
                    f"Unhandled nq ({nq_joint}) for joint {joint_name}. Assigning raw value if possible."
                )
                if idx_q < len(q):  # Fallback for simple cases
                    q[idx_q] = current_joint_val

        return q

    def _get_pinocchio_imu_tooltip_orientation(
        self, q_pin_config: np.ndarray
    ) -> Optional[R]:
        # Calculates R_IMUframe_to_ToolTip using Pinocchio
        if (
            self.pin_model is None
            or self.pin_data is None
            or self.imu_frame_id_pin < 0
            or self.tooltip_frame_id_pin < 0
        ):
            self.get_logger().error(
                "Pinocchio not ready for IMU->Tooltip FK.", throttle_duration_sec=1.0
            )
            return None
        try:
            pin.forwardKinematics(self.pin_model, self.pin_data, q_pin_config)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            T_world_imu_pin = self.pin_data.oMf[self.imu_frame_id_pin]
            T_world_tooltip_pin = self.pin_data.oMf[self.tooltip_frame_id_pin]

            # Check for valid rotations before inversion
            if not (
                T_world_imu_pin.rotation.trace() > -0.99999
                and T_world_tooltip_pin.rotation.trace() > -0.99999
            ):
                self.get_logger().warn(
                    "Invalid rotation matrix in FK for IMU or Tooltip.",
                    throttle_duration_sec=1.0,
                )

            T_imu_tooltip_se3 = T_world_imu_pin.inverse() * T_world_tooltip_pin
            return R.from_matrix(T_imu_tooltip_se3.rotation)
        except Exception as e:
            self.get_logger().error(f"Error in Pinocchio FK for IMU->Tooltip: {e}")
            import traceback

            self.get_logger().error(traceback.format_exc())
            return None

    def _calculate_joint_velocities_pinocchio_jacobian(
        self, q_pin_config: np.ndarray, omega_desired_tooltip_local: np.ndarray
    ) -> Optional[np.ndarray]:
        if (
            self.pin_model is None
            or self.pin_data is None
            or self.tooltip_frame_id_pin < 0
        ):
            self.get_logger().error(
                "Pinocchio not ready for Jacobian.", throttle_duration_sec=1.0
            )
            return None
        try:
            # Ensure FK and frame placements are current for this q_pin_config
            pin.forwardKinematics(self.pin_model, self.pin_data, q_pin_config)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            J_tooltip_local_full = pin.computeFrameJacobian(
                self.pin_model,
                self.pin_data,
                q_pin_config,
                self.tooltip_frame_id_pin,
                pin.ReferenceFrame.LOCAL,
            )
            if (
                not self.articutool_v_indices_pin
                or len(self.articutool_v_indices_pin) != 2
            ):
                self.get_logger().error(
                    f"Pinocchio v_indices for Articutool not set up correctly: {self.articutool_v_indices_pin}"
                )
                return np.zeros(2)

            # Select columns corresponding to Articutool's actuated joints
            # J_tooltip_local_full is 6xNv (linear vel, angular vel)
            # We need the angular velocity part (rows 3,4,5) for the Articutool joints
            J_tooltip_angular_local_articutool_joints = J_tooltip_local_full[
                3:6, self.articutool_v_indices_pin
            ]
            if J_tooltip_angular_local_articutool_joints.shape != (3, 2):
                self.get_logger().error(
                    f"Pinocchio Jacobian (angular part for Articutool joints) shape error: {J_tooltip_angular_local_articutool_joints.shape}. Expected (3,2)."
                )
                return np.zeros(2)

            try:
                J_pinv = np.linalg.pinv(
                    J_tooltip_angular_local_articutool_joints,
                    rcond=self.JACOBIAN_PINV_RCOND,
                )
            except np.linalg.LinAlgError:
                self.get_logger().warn(
                    "Full Orient: Jacobian pseudo-inverse failed. Commanding zero.",
                    throttle_duration_sec=1.0,
                )
                return np.zeros(2)

            dq_desired = J_pinv @ omega_desired_tooltip_local
            if dq_desired.shape != (2,):
                self.get_logger().error(
                    f"Calculated dq_desired shape error: {dq_desired.shape}. Expected (2,)."
                )
                return np.zeros(2)
            return dq_desired
        except Exception as e:
            self.get_logger().error(
                f"Error calculating joint velocities with Pinocchio Jacobian: {e}"
            )
            import traceback

            self.get_logger().error(traceback.format_exc())
            return None

    def _reset_pid_for_mode(self, mode_to_reset_for: int):
        if mode_to_reset_for == MODE_LEVELING:
            self.integral_error_leveling.fill(0.0)
            self.last_error_leveling.fill(0.0)
        elif mode_to_reset_for == MODE_FULL_ORIENTATION:
            self.integral_error_full_orientation.fill(0.0)
            self.last_error_full_orientation.fill(0.0)
        self.last_time = None
        self.get_logger().info(
            f"PID controller state reset for mode {mode_to_reset_for}."
        )

    def _dampen_velocities_near_limits(
        self, current_q: Optional[np.ndarray], desired_dq: np.ndarray
    ) -> np.ndarray:
        dampened_dq = np.copy(desired_dq)
        if current_q is None:
            self.get_logger().warn(
                "_dampen_velocities_near_limits called with current_q=None. Returning original dq.",
                throttle_duration_sec=5.0,
            )
            return desired_dq
        if len(current_q) != len(self.articutool_joint_names) or len(desired_dq) != len(
            self.articutool_joint_names
        ):
            self.get_logger().error(
                f"Mismatched lengths in _dampen_velocities_near_limits. q:{len(current_q)}, dq:{len(desired_dq)}"
            )
            return (
                np.zeros(len(self.articutool_joint_names))
                if len(desired_dq) != len(self.articutool_joint_names)
                else desired_dq
            )

        for i in range(len(self.articutool_joint_names)):
            q_i, dq_i = current_q[i], desired_dq[i]
            lower_limit, upper_limit = (
                self.joint_limits_lower[i],
                self.joint_limits_upper[i],
            )
            # Use the loaded parameter for threshold, ensure it's positive
            threshold = max(self.joint_limit_threshold, self.EPSILON)
            damp_power = self.joint_limit_dampening_factor

            scale = 1.0

            if dq_i < -self.EPSILON:
                distance_to_lower = q_i - lower_limit
                if distance_to_lower < threshold:
                    # Scale is (current_distance / threshold_distance) ^ power
                    # Clip to ensure scale is between 0 and 1
                    scale = (
                        np.clip(distance_to_lower / threshold, 0.0, 1.0) ** damp_power
                    )
            elif dq_i > self.EPSILON:  # Moving towards upper limit
                distance_to_upper = upper_limit - q_i
                if distance_to_upper < threshold:
                    scale = (
                        np.clip(distance_to_upper / threshold, 0.0, 1.0) ** damp_power
                    )
            dampened_dq[i] *= scale
        return dampened_dq

    def _publish_command(self, joint_velocities: Optional[np.ndarray]):
        if joint_velocities is None:
            self.get_logger().warn(
                "Received None for joint_velocities in _publish_command. Sending zero.",
                throttle_duration_sec=1.0,
            )
            joint_velocities = np.zeros(len(self.articutool_joint_names))

        if len(joint_velocities) != len(self.articutool_joint_names):
            self.get_logger().error(
                f"Command length ({len(joint_velocities)}) mismatch with expected ({len(self.articutool_joint_names)}). Sending zero."
            )
            joint_velocities = np.zeros(len(self.articutool_joint_names))

        cmd_msg = Float64MultiArray()
        cmd_msg.data = joint_velocities.tolist()
        self.cmd_pub.publish(cmd_msg)

    def _publish_zero_command(self):
        self._publish_command(np.zeros(len(self.articutool_joint_names)))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = OrientationControl()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        if node:
            node.get_logger().info(
                "Shutting down cleanly due to interrupt or external request."
            )
    except ValueError as e:
        logger = (
            node.get_logger()
            if node
            else rclpy.logging.get_logger("orientation_control_prerun_error")
        )
        logger.fatal(f"Node initialization or runtime ValueError: {e}")
        import traceback

        logger.error(traceback.format_exc())
    except Exception as e:
        logger = (
            node.get_logger()
            if node
            else rclpy.logging.get_logger("orientation_control_unhandled_error")
        )
        logger.fatal(f"Unhandled exception in OrientationControl node: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        if node:
            if node.current_mode != MODE_DISABLED:
                node.get_logger().info("Publishing zero command before shutdown.")
                node._publish_zero_command()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
