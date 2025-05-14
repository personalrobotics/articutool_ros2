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

from articutool_interfaces.srv import SetOrientationControl, TriggerCalibration

import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

import tf2_ros
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)

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

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Pinocchio members will be initialized by _setup_pinocchio()
        # They are primarily for MODE_FULL_ORIENTATION and calibration.
        self.pin_model: Optional[pin.Model] = None
        self.pin_data: Optional[pin.Data] = None
        self.imu_frame_id_pin: int = -1
        self.tooltip_frame_id_pin: int = -1
        self.handle_frame_id_pin: int = -1
        self.articutool_joint_ids_pin: List[int] = []
        self.articutool_q_indices_pin: List[int] = []
        self.articutool_v_indices_pin: List[int] = []
        self.R_handle_imu_pin: Optional[R] = None  # Pinocchio-derived R_Handle_IMU
        self.R_imu_handle_pin: Optional[R] = None  # Pinocchio-derived R_IMU_Handle

        # Attempt to set up Pinocchio. If it fails, MODE_FULL_ORIENTATION and calibration
        # might be unavailable, but MODE_LEVELING should still work.
        try:
            self._setup_pinocchio()
            self.get_logger().info("Pinocchio setup successful.")
        except Exception as e:
            self.get_logger().error(
                f"Pinocchio setup failed: {e}. MODE_FULL_ORIENTATION and calibration may be unavailable.",
                exc_info=True,
            )
            # Nullify Pinocchio-dependent members to ensure they are not used if setup fails
            self.pin_model = None
            self.pin_data = None
            self.R_imu_handle_pin = None

        self.current_mode = MODE_DISABLED
        self.target_orientation_jacobase: Optional[R] = None

        self.last_imu_msg_time: Optional[Time] = None
        self.current_filterworld_to_imu_raw: Optional[R] = None
        self.current_linear_accel_imu: Optional[np.ndarray] = None
        self.current_angular_velocity_imu: Optional[np.ndarray] = None

        self.q_JacoBase_to_FilterWorld_cal: Optional[R] = None
        self.is_calibrated = False

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
        self.cal_srv = self.create_service(
            TriggerCalibration,
            "/articutool/trigger_orientation_calibration",
            self.trigger_calibration_callback,
        )
        self.imu_sub = self.create_subscription(
            Imu, self.imu_topic, self.feedback_callback, 1
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
        # ... (Same as before, ensure defaults are sensible, e.g. Kp=0.5, Ki=0, Kd=0.01) ...
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
            "imu_topic", "/articutool/imu_data_and_orientation", string_desc
        )
        self.declare_parameter(
            "command_topic", "/articutool/velocity_controller/commands", string_desc
        )
        self.declare_parameter(
            "joint_state_topic", "/articutool/joint_states", string_desc
        )
        self.declare_parameter("urdf_path", "", string_desc)
        self.declare_parameter("robot_base_frame", "j2n6s200_link_base", string_desc)
        self.declare_parameter("articutool_base_link", "atool_handle", string_desc)
        self.declare_parameter("imu_link_frame", "atool_imu_frame", string_desc)
        self.declare_parameter("tooltip_frame", "tool_tip", string_desc)
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
            1.0,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )

    def _load_parameters(self):
        # ... (Same as before) ...
        self.Kp = self.get_parameter("pid_gains.p").value
        self.Ki = self.get_parameter("pid_gains.i").value
        self.Kd = self.get_parameter("pid_gains.d").value
        self.integral_max = self.get_parameter("integral_clamp").value
        self.rate = self.get_parameter("loop_rate").value
        self.imu_topic = self.get_parameter("imu_topic").value
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
        self.get_logger().debug("Parameters loaded/reloaded.")

    def parameters_callback(self, params: List[Parameter]):
        # ... (Same as before) ...
        require_recalibration = False
        accepted_params_names = []
        for param in params:
            if param.name in [
                "pid_gains.p",
                "pid_gains.i",
                "pid_gains.d",
                "integral_clamp",
                "joint_limits.threshold",
                "joint_limits.dampening_factor",
                "loop_rate",
            ]:
                accepted_params_names.append(param.name)
            elif param.name == "robot_base_frame":
                accepted_params_names.append(param.name)
                require_recalibration = True
            elif param.name in [
                "urdf_path",
                "articutool_base_link",
                "imu_link_frame",
                "tooltip_frame",
                "joint_names",
            ]:
                self.get_logger().warn(
                    f"Change to '{param.name}' typically requires node restart. Current Pinocchio model (if any) will NOT be reloaded dynamically."
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
            if require_recalibration:
                self.is_calibrated = False
                self.q_JacoBase_to_FilterWorld_cal = None
                self.get_logger().warn(
                    "Recalibration is required due to parameter change."
                )
                if self.current_mode == MODE_FULL_ORIENTATION:
                    self.get_logger().error(
                        "Disabling FULL_ORIENTATION mode due to required recalibration."
                    )
                    self.current_mode = MODE_DISABLED
                    self._publish_zero_command()
            return SetParametersResult(successful=True)
        return SetParametersResult(successful=False)

    def _setup_pinocchio(self):
        # This method now primarily serves MODE_FULL_ORIENTATION and calibration.
        # MODE_LEVELING will use its own hardcoded/analytic values where possible.
        if not self.xacro_filename:
            self.get_logger().warn(
                "URDF path not provided. Pinocchio setup skipped. MODE_FULL_ORIENTATION and calibration will be unavailable."
            )
            self.pin_model = None  # Ensure it's None if not loaded
            return
        if not os.path.exists(self.xacro_filename):
            self.get_logger().error(
                f"Xacro file not found at {self.xacro_filename}. Pinocchio setup failed. MODE_FULL_ORIENTATION and calibration will be unavailable."
            )
            self.pin_model = None
            return

        temp_urdf_path = None
        try:
            self.get_logger().info(
                f"Processing Xacro file for Pinocchio: {self.xacro_filename}"
            )
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

            # It's crucial that this URDF correctly models the Articutool *within the full robot context*
            # or is a standalone Articutool URDF where atool_handle is the root.
            self.pin_model = pin.buildModelFromUrdf(temp_urdf_path)
            self.pin_data = self.pin_model.createData()
            self.get_logger().info(
                f"Pinocchio model loaded: {self.pin_model.name}, Nq={self.pin_model.nq}, Nv={self.pin_model.nv}"
            )

            self.handle_frame_id_pin = self.pin_model.getFrameId(
                self.articutool_base_link_name
            )
            self.imu_frame_id_pin = self.pin_model.getFrameId(self.imu_link_name)
            self.tooltip_frame_id_pin = self.pin_model.getFrameId(
                self.tooltip_link_name
            )

            q_neutral = pin.neutral(self.pin_model)
            pin.forwardKinematics(self.pin_model, self.pin_data, q_neutral)
            pin.updateFramePlacements(self.pin_model, self.pin_data)

            T_pinworld_handle = self.pin_data.oMf[self.handle_frame_id_pin]
            T_pinworld_imu = self.pin_data.oMf[self.imu_frame_id_pin]
            T_handle_imu_se3 = (
                T_pinworld_handle.inverse() * T_pinworld_imu
            )  # T_Handle_IMU
            self.R_handle_imu_pin = R.from_matrix(T_handle_imu_se3.rotation)
            self.R_imu_handle_pin = self.R_handle_imu_pin.inv()
            self.get_logger().info(
                f"Pinocchio-derived R_handle_imu_pin (Handle to IMU, xyzw): {self.R_handle_imu_pin.as_quat()}"
            )
            self.get_logger().info(
                f"Pinocchio-derived R_imu_handle_pin (IMU to Handle, xyzw): {self.R_imu_handle_pin.as_quat()}"
            )

            # Sanity check for IMU frame parentage in Pinocchio
            imu_frame_obj = self.pin_model.frames[self.imu_frame_id_pin]
            imu_parent_joint_id = imu_frame_obj.parentJoint
            imu_parent_joint_name = self.pin_model.names[imu_parent_joint_id]
            self.get_logger().info(
                f"Pinocchio: IMU frame '{self.imu_link_name}' (ID {self.imu_frame_id_pin}) is attached to JOINT '{imu_parent_joint_name}' (ID {imu_parent_joint_id})."
            )
            if imu_parent_joint_id == 0:  # Universe joint
                self.get_logger().warn(
                    "Pinocchio sees IMU frame's parent joint as Universe! "
                    "This might lead to incorrect R_handle_imu_pin if URDF is not a single tree starting from Jaco base "
                    "or if Articutool model is loaded standalone without 'atool_handle' as its fixed root."
                )

            self.articutool_joint_ids_pin = []
            self.articutool_q_indices_pin = []
            self.articutool_v_indices_pin = []
            for joint_name in self.articutool_joint_names:
                if not self.pin_model.existJointName(joint_name):
                    # This is a critical failure for MODE_FULL_ORIENTATION
                    self.get_logger().error(
                        f"Articutool joint '{joint_name}' not found in Pinocchio model. MODE_FULL_ORIENTATION will fail."
                    )
                    self.pin_model = None  # Invalidate model for safety
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
            self.get_logger().error(
                f"Pinocchio setup failed during model processing: {e}. MODE_FULL_ORIENTATION and calibration may be unavailable.",
                exc_info=True,
            )
            self.pin_model = None  # Ensure it's None on failure
            self.R_imu_handle_pin = None
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
        # ... (Logic unchanged from previous version) ...
        self.get_logger().info(
            f"Received SetOrientationControl Request: mode={request.control_mode}"
        )
        response.success = False
        if request.control_mode == MODE_DISABLED:
            if self.current_mode != MODE_DISABLED:
                self._publish_zero_command()
            self.current_mode = MODE_DISABLED
            response.success = True
            response.message = "Orientation control disabled."
        elif request.control_mode == MODE_LEVELING:
            if self.current_mode != MODE_LEVELING:
                self.get_logger().info(
                    "Switching to LEVELING mode (Y_tip along World_Z_up)."
                )
                self._reset_pid_for_mode(MODE_LEVELING)
            else:
                self.get_logger().info("Already in LEVELING mode.")
            self.current_mode = MODE_LEVELING
            response.success = True
            response.message = "Leveling mode (analytic Jacobian) enabled."
        elif request.control_mode == MODE_FULL_ORIENTATION:
            if self.pin_model is None:  # Check if Pinocchio setup was successful
                response.message = "Cannot enable FULL_ORIENTATION: Pinocchio model not loaded/failed to load."
                self.get_logger().error(response.message)
                if self.current_mode != MODE_DISABLED:
                    self._publish_zero_command()
                self.current_mode = MODE_DISABLED
                return response
            if not self.is_calibrated:
                response.message = (
                    "Cannot enable FULL_ORIENTATION: System not calibrated."
                )
                self.get_logger().error(response.message)
                if self.current_mode != MODE_DISABLED:
                    self._publish_zero_command()
                self.current_mode = MODE_DISABLED
                return response
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
                if self.current_mode != MODE_DISABLED:
                    self._publish_zero_command()
                self.current_mode = MODE_DISABLED
                return response
        else:
            response.message = f"Invalid control_mode: {request.control_mode}"
            self.get_logger().error(response.message)
            return response
        return response

    def trigger_calibration_callback(
        self, request: TriggerCalibration.Request, response: TriggerCalibration.Response
    ):
        # ... (Logic unchanged, but it relies on self.R_imu_handle_pin from Pinocchio) ...
        self.get_logger().info(
            "Calibration triggered. Ensure robot and Articutool base are STATIONARY."
        )
        self.is_calibrated = False
        self.q_JacoBase_to_FilterWorld_cal = None
        response.success = False
        if self.current_filterworld_to_imu_raw is None:
            response.message = "Cannot calibrate: No IMU data."
            self.get_logger().error(response.message)
            return response
        if self.R_imu_handle_pin is None:  # This is set by _setup_pinocchio
            response.message = "Cannot calibrate: R_imu_handle_pin (from Pinocchio) not available. Pinocchio setup might have failed."
            self.get_logger().error(response.message)
            return response
        try:
            transform_jb_handle = self.tf_buffer.lookup_transform(
                self.robot_base_frame,
                self.articutool_base_link_name,
                rclpy.time.Time(seconds=0),
                timeout=RCLPYDuration(seconds=1.0),
            )
            q_jb_handle_tf = R.from_quat(
                [
                    transform_jb_handle.transform.rotation.x,
                    transform_jb_handle.transform.rotation.y,
                    transform_jb_handle.transform.rotation.z,
                    transform_jb_handle.transform.rotation.w,
                ]
            )
            q_fw_imu = self.current_filterworld_to_imu_raw
            q_imu_handle = self.R_imu_handle_pin  # Use Pinocchio-derived version
            q_fw_handle = q_fw_imu * q_imu_handle
            self.q_JacoBase_to_FilterWorld_cal = q_jb_handle_tf * q_fw_handle.inv()
            self.is_calibrated = True
            response.success = True
            response.message = "Calibration successful."
            cal_quat_msg = self.q_JacoBase_to_FilterWorld_cal.as_quat()
            (
                response.computed_offset_jacobase_to_filterworld.x,
                response.computed_offset_jacobase_to_filterworld.y,
                response.computed_offset_jacobase_to_filterworld.z,
                response.computed_offset_jacobase_to_filterworld.w,
            ) = cal_quat_msg
            self.get_logger().info(
                f"{response.message} R_JacoBase_FilterWorld (xyzw): {cal_quat_msg}"
            )
        except Exception as e:
            response.message = f"Calibration failed: {e}"
            self.get_logger().error(response.message)
            self.is_calibrated = False
            self.q_JacoBase_to_FilterWorld_cal = None
        return response

    def feedback_callback(self, msg: Imu):
        # ... (Same as before) ...
        self.last_imu_msg_time = Time.from_msg(msg.header.stamp)
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
        elif self.current_filterworld_to_imu_raw is None:
            self.get_logger().warn(
                "Received zero or NaN quaternion from IMU filter.",
                throttle_duration_sec=5.0,
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

    def joint_state_callback(self, msg: JointState):
        # ... (Same as before) ...
        if self.current_joint_positions is None:  # Initialize if first time
            self.current_joint_positions = np.zeros(len(self.articutool_joint_names))

        joint_map = {name: i for i, name in enumerate(self.articutool_joint_names)}
        for i, name in enumerate(msg.name):
            if name in joint_map:  # Check if this joint is one we are controlling
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

        # Prerequisite check
        mode_leveling_ready = (
            self.current_filterworld_to_imu_raw is not None
            and self.current_joint_positions is not None
        )
        # For MODE_FULL_ORIENTATION, Pinocchio model must be loaded (self.pin_model implies self.R_imu_handle_pin is also set if successful)
        mode_full_orientation_ready = (
            mode_leveling_ready
            and self.pin_model is not None
            and self.is_calibrated
            and self.target_orientation_jacobase is not None
        )

        if self.current_mode == MODE_DISABLED:
            return  # Do nothing further if disabled explicitly

        commanded_dq: Optional[np.ndarray] = None
        try:
            if self.current_mode == MODE_LEVELING:
                if mode_leveling_ready:
                    commanded_dq = self._calculate_leveling_control_analytic_jacobian(
                        dt
                    )
                else:
                    self.get_logger().warn(
                        "MODE_LEVELING: Prerequisites not met (IMU/Joints). Commanding zero.",
                        throttle_duration_sec=1.0,
                    )
            elif self.current_mode == MODE_FULL_ORIENTATION:
                if mode_full_orientation_ready:
                    commanded_dq = self._calculate_full_orientation_control(dt)
                else:
                    self.get_logger().warn(
                        "MODE_FULL_ORIENTATION: Prerequisites not met (IMU/Joints/Pinocchio/Calibration/Target). Commanding zero.",
                        throttle_duration_sec=1.0,
                    )

            if commanded_dq is not None:
                self._publish_command(commanded_dq)
            else:
                self._publish_zero_command()  # If calculation failed or mode conditions not met
        except Exception as e:
            self.get_logger().error(
                f"Unhandled exception in control_loop (Mode: {self.current_mode}): {e}",
                exc_info=True,
            )
            self._publish_zero_command()

    def _calculate_leveling_control_analytic_jacobian(
        self, dt: float
    ) -> Optional[np.ndarray]:
        """
        Mode LEVELING: Keeps tool_tip Y-axis aligned with FilterWorld Z-up (gravity).
        Uses an analytic Jacobian and a pre-calculated R_IMU_TO_HANDLE_FIXED_SCIPY.
        This method is now Pinocchio-independent.
        """
        if (
            self.current_filterworld_to_imu_raw is None
            or self.current_joint_positions is None
        ):
            self.get_logger().warn(
                "Leveling (Analytic): Missing IMU or Joint data.",
                throttle_duration_sec=1.0,
            )
            return None

        try:
            theta_p_curr = self.current_joint_positions[0]
            theta_r_curr = self.current_joint_positions[1]

            # Use the pre-calculated fixed transform
            R_FilterW_Handle: R = (
                self.current_filterworld_to_imu_raw * self.R_IMU_TO_HANDLE_FIXED_SCIPY
            )

            cp, sp = math.cos(theta_p_curr), math.sin(theta_p_curr)
            cr, sr = math.cos(theta_r_curr), math.sin(theta_r_curr)
            y_tip_in_F0_curr = np.array([cp * cr, sr, sp * cr])  # Y-tip in Handle frame
            y_tip_in_FilterW_curr = R_FilterW_Handle.apply(y_tip_in_F0_curr)
            target_y_tip_in_FilterW = self.WORLD_Z_UP_VECTOR

            axis_err_FilterW = np.cross(y_tip_in_FilterW_curr, target_y_tip_in_FilterW)
            dot_prod = np.dot(y_tip_in_FilterW_curr, target_y_tip_in_FilterW)
            angle_err = math.acos(np.clip(dot_prod, -1.0, 1.0))

            error_vec_FilterW = np.zeros(3)
            norm_axis_err_val = np.linalg.norm(axis_err_FilterW)
            if norm_axis_err_val > self.EPSILON:
                error_vec_FilterW = (axis_err_FilterW / norm_axis_err_val) * angle_err
            elif dot_prod < (-1.0 + self.EPSILON):  # Anti-parallel
                arbitrary_axis = np.array([1.0, 0.0, 0.0])
                if (
                    np.linalg.norm(np.cross(y_tip_in_FilterW_curr, arbitrary_axis))
                    < self.EPSILON
                ):
                    arbitrary_axis = np.array([0.0, 1.0, 0.0])
                rotation_axis = np.cross(y_tip_in_FilterW_curr, arbitrary_axis)
                if np.linalg.norm(rotation_axis) > self.EPSILON:
                    error_vec_FilterW = (
                        rotation_axis / np.linalg.norm(rotation_axis)
                    ) * math.pi
                else:
                    error_vec_FilterW = np.array([math.pi, 0.0, 0.0])

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

            omega_corr_in_F0 = R_FilterW_Handle.inv().apply(omega_corr_FilterW)

            # Analytic Pointing Jacobian J_point_in_F0 (maps [dp, dr] to d(y_tip_in_F0)/dt)
            J_point_in_F0 = np.array(
                [[-sp * cr, -cp * sr], [0.0, cr], [cp * cr, -sp * sr]]
            )

            try:
                J_point_in_F0_pinv = np.linalg.pinv(
                    J_point_in_F0, rcond=self.JACOBIAN_PINV_RCOND
                )
            except np.linalg.LinAlgError:
                self.get_logger().warn(
                    "Leveling control (Analytic): Pointing Jacobian pseudo-inverse failed.",
                    throttle_duration_sec=1.0,
                )
                return np.zeros(2)

            desired_y_tip_linear_velocity_in_F0 = np.cross(
                omega_corr_in_F0, y_tip_in_F0_curr
            )
            self.get_logger().info(
                f"  desired_y_tip_linear_velocity_in_F0: {np.round(desired_y_tip_linear_velocity_in_F0, 4)}"
            )  # Add this log
            dq_desired = J_point_in_F0_pinv @ desired_y_tip_linear_velocity_in_F0

            self.get_logger().info(f"LEVELING CTRL (Analytic) TICK (dt={dt:.4f}):")
            self.get_logger().info(
                f"  Angles (p,r): ({theta_p_curr:.3f}, {theta_r_curr:.3f})"
            )
            self.get_logger().info(
                f"  R_FilterW_Handle (quat xyzw): {R_FilterW_Handle.as_quat()}"
            )
            self.get_logger().info(
                f"  y_tip_in_F0_curr: {np.round(y_tip_in_F0_curr, 3)}"
            )
            self.get_logger().info(
                f"  y_tip_in_FilterW_curr: {np.round(y_tip_in_FilterW_curr, 3)}"
            )
            self.get_logger().info(
                f"  angle_err_rad: {angle_err:.4f} ({(angle_err * 180 / math.pi):.2f} deg)"
            )
            self.get_logger().info(
                f"  error_vec_FilterW: {np.round(error_vec_FilterW, 4)}"
            )
            self.get_logger().info(
                f"  omega_corr_FilterW: {np.round(omega_corr_FilterW, 4)}"
            )
            self.get_logger().info(
                f"  omega_corr_in_F0: {np.round(omega_corr_in_F0, 4)}"
            )
            self.get_logger().info(f"  dq_desired (raw): {np.round(dq_desired, 4)}")

            return self._dampen_velocities_near_limits(
                self.current_joint_positions, dq_desired
            )

        except Exception as e:
            self.get_logger().error(f"Error in Analytic Jacobian Leveling Control: {e}")
            return None

    def _calculate_full_orientation_control(self, dt: float) -> Optional[np.ndarray]:
        # ... (Logic remains the same, relies on self.pin_model and self.R_imu_handle_pin) ...
        # This method is called if self.pin_model is not None and calibrated.
        # Its prerequisites are checked in the main control_loop.
        # If self.pin_model is None, this mode won't be effectively callable.
        if (
            self.pin_model is None or self.R_imu_handle_pin is None
        ):  # Guard against Pinocchio not being ready
            self.get_logger().error(
                "Full Orientation: Pinocchio model or R_imu_handle_pin not available.",
                throttle_duration_sec=1.0,
            )
            return None
        try:
            if not self.is_calibrated or self.q_JacoBase_to_FilterWorld_cal is None:
                self.get_logger().error(
                    "Full Orientation: Not calibrated!", throttle_duration_sec=1.0
                )
                return None
            if self.target_orientation_jacobase is None:
                self.get_logger().warn(
                    "Full Orientation: Target not set.", throttle_duration_sec=1.0
                )
                return None

            q_jb_fw_cal = self.q_JacoBase_to_FilterWorld_cal
            q_fw_imu_raw = self.current_filterworld_to_imu_raw
            q_jb_imu = q_jb_fw_cal * q_fw_imu_raw

            q_pin_config = self._get_pinocchio_config()
            q_imu_tooltip_current_pin = self._get_pinocchio_imu_tooltip_orientation(
                q_pin_config
            )
            if q_imu_tooltip_current_pin is None:
                return None

            q_jb_tooltip_current = q_jb_imu * q_imu_tooltip_current_pin
            q_error_jacobase = (
                self.target_orientation_jacobase * q_jb_tooltip_current.inv()
            )
            error_vec_jacobase = q_error_jacobase.as_rotvec()

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
                return None

            if (
                self.get_logger().get_effective_level()
                <= rclpy.logging.LoggingSeverity.DEBUG
            ):
                self.get_logger().debug(
                    f"FullOrient Ctrl: err_norm={np.linalg.norm(error_vec_jacobase):.3f}, "
                    f"omega_des_L={np.round(omega_desired_tooltip_local, 3)}, "
                    f"dq_des={np.round(dq_desired, 3)}"
                )
            return self._dampen_velocities_near_limits(
                self.current_joint_positions, dq_desired
            )
        except Exception as e:
            self.get_logger().error(
                f"Error in Full Orientation Control (Pinocchio Jacobian): {e}",
                exc_info=True,
            )
            return None

    def _get_pinocchio_config(self) -> np.ndarray:
        # ... (Same as before) ...
        if self.pin_model is None:
            raise ValueError("Pinocchio model not loaded.")
        if self.current_joint_positions is None:
            raise ValueError("Joint positions None.")
        if len(self.current_joint_positions) != len(self.articutool_joint_names):
            raise ValueError(
                f"Mismatch joint positions ({len(self.current_joint_positions)}) vs names ({len(self.articutool_joint_names)})"
            )
        q = pin.neutral(self.pin_model)
        for i, joint_pin_id in enumerate(self.articutool_joint_ids_pin):
            q_idx = self.pin_model.joints[joint_pin_id].idx_q
            if self.pin_model.joints[joint_pin_id].nq == 1:
                q[q_idx] = self.current_joint_positions[i]
        return q

    def _get_pinocchio_imu_tooltip_orientation(
        self, q_pin_config: np.ndarray
    ) -> Optional[R]:
        # ... (Same as before) ...
        if (
            self.pin_model is None
            or self.pin_data is None
            or self.imu_frame_id_pin < 0
            or self.tooltip_frame_id_pin < 0
        ):
            self.get_logger().error(
                "Pinocchio model/data/frames not initialized for FK.",
                throttle_duration_sec=1.0,
            )
            return None
        try:
            pin.forwardKinematics(self.pin_model, self.pin_data, q_pin_config)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            T_world_imu_pin = self.pin_data.oMf[self.imu_frame_id_pin]
            T_world_tooltip_pin = self.pin_data.oMf[self.tooltip_frame_id_pin]
            T_imu_tooltip_se3 = T_world_imu_pin.inverse() * T_world_tooltip_pin
            return R.from_matrix(T_imu_tooltip_se3.rotation)
        except Exception as e:
            self.get_logger().error(f"Error in Pinocchio FK for IMU->Tooltip: {e}")
            return None

    def _calculate_joint_velocities_pinocchio_jacobian(
        self, q_pin_config: np.ndarray, omega_desired_tooltip_local: np.ndarray
    ) -> Optional[np.ndarray]:
        # ... (Same as before) ...
        if (
            self.pin_model is None
            or self.pin_data is None
            or self.tooltip_frame_id_pin < 0
        ):
            self.get_logger().error(
                "Pinocchio model/data/frames not initialized for Jacobian.",
                throttle_duration_sec=1.0,
            )
            return None
        try:
            pin.computeFrameJacobian(
                self.pin_model,
                self.pin_data,
                q_pin_config,
                self.tooltip_frame_id_pin,
                pin.ReferenceFrame.LOCAL,
            )
            J_tooltip_local_full = pin.getFrameJacobian(
                self.pin_model,
                self.pin_data,
                self.tooltip_frame_id_pin,
                pin.ReferenceFrame.LOCAL,
            )
            if (
                not self.articutool_v_indices_pin
                or len(self.articutool_v_indices_pin) != 2
            ):
                self.get_logger().error(
                    "Pinocchio v_indices for Articutool joints not set up correctly."
                )
                return np.zeros(2)
            J_tooltip_angular_local_articutool_joints = J_tooltip_local_full[
                3:6, self.articutool_v_indices_pin
            ]
            if J_tooltip_angular_local_articutool_joints.shape != (3, 2):
                self.get_logger().error(
                    f"Pinocchio Jacobian shape error: {J_tooltip_angular_local_articutool_joints.shape}."
                )
                return np.zeros(2)
            if np.allclose(
                J_tooltip_angular_local_articutool_joints, 0, atol=self.EPSILON
            ):  # Added atol
                self.get_logger().warn(
                    "Pinocchio Jacobian for Articutool joints is all zeros.",
                    throttle_duration_sec=2.0,
                )
            J_pinv = np.linalg.pinv(
                J_tooltip_angular_local_articutool_joints,
                rcond=self.JACOBIAN_PINV_RCOND,
            )
            dq_desired = J_pinv @ omega_desired_tooltip_local
            return dq_desired
        except Exception as e:
            self.get_logger().error(
                f"Error calculating joint velocities with Pinocchio Jacobian: {e}",
                exc_info=True,
            )
            return None

    def _reset_pid_for_mode(self, mode_to_reset_for: int):
        # ... (Same as before) ...
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
        if (
            current_q is None
            or len(current_q) != len(self.articutool_joint_names)
            or len(desired_dq) != len(self.articutool_joint_names)
        ):
            self.get_logger().error(
                f"Dampening: current_q ({current_q}) is None or length mismatch with desired_dq ({len(desired_dq)}) or joint_names ({len(self.articutool_joint_names)})."
            )
            # Return a zero array of the correct size if desired_dq is problematic, or desired_dq if current_q is the issue
            if len(desired_dq) != len(self.articutool_joint_names):
                return np.zeros(len(self.articutool_joint_names))
            return desired_dq

        for i in range(len(self.articutool_joint_names)):
            q_i, dq_i = current_q[i], desired_dq[i]
            lower_limit, upper_limit = (
                self.joint_limits_lower[i],
                self.joint_limits_upper[i],
            )
            # Ensure threshold is positive to avoid division by zero or unintended behavior
            threshold = max(self.joint_limit_threshold, self.EPSILON)
            damp_factor = self.joint_limit_dampening_factor

            scale = 1.0  # Default: no dampening

            # Moving towards lower limit
            if dq_i < -self.EPSILON:
                distance_to_lower = q_i - lower_limit
                if distance_to_lower < threshold:
                    # Scale is 0 at the limit, 1 at the threshold distance
                    current_scale = distance_to_lower / threshold
                    scale = np.clip(current_scale, 0.0, 1.0) ** damp_factor
                    # self.get_logger().debug(f"Joint {i} dampening towards lower: q={q_i:.3f}, dq={dq_i:.3f}, dist={distance_to_lower:.3f}, scale={scale:.3f}")

            # Moving towards upper limit
            elif dq_i > self.EPSILON:
                distance_to_upper = upper_limit - q_i
                if distance_to_upper < threshold:
                    # Scale is 0 at the limit, 1 at the threshold distance
                    current_scale = distance_to_upper / threshold
                    scale = np.clip(current_scale, 0.0, 1.0) ** damp_factor
                    # self.get_logger().debug(f"Joint {i} dampening towards upper: q={q_i:.3f}, dq={dq_i:.3f}, dist={distance_to_upper:.3f}, scale={scale:.3f}")

            dampened_dq[i] *= scale

        if not np.allclose(
            desired_dq, dampened_dq, atol=1e-4
        ):  # Add tolerance for logging
            self.get_logger().debug(
                f"Dampening: q={np.round(current_q, 3)}, original_dq={np.round(desired_dq, 3)}, dampened_dq={np.round(dampened_dq, 3)}"
            )
        return dampened_dq

    def _publish_command(self, joint_velocities: Optional[np.ndarray]):  # Can be None
        # ... (Same as before) ...
        if joint_velocities is None:
            joint_velocities = np.zeros(len(self.articutool_joint_names))
        if len(joint_velocities) != len(self.articutool_joint_names):
            self.get_logger().error(f"Command length mismatch. Sending zero.")
            joint_velocities = np.zeros(len(self.articutool_joint_names))
        cmd_msg = Float64MultiArray()
        cmd_msg.data = joint_velocities.tolist()
        self.cmd_pub.publish(cmd_msg)

    def _publish_zero_command(self):
        self._publish_command(np.zeros(len(self.articutool_joint_names)))


def main(args=None):
    # ... (Same as before) ...
    rclpy.init(args=args)
    node = None
    try:
        node = OrientationControl()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        if node:
            node.get_logger().info("Shutting down cleanly.")
    except RuntimeError as e:
        logger = (
            node.get_logger()
            if node
            else rclpy.logging.get_logger("orientation_control_early_error")
        )
        logger.fatal(f"Node initialization failed: {e}")
    except Exception as e:
        logger = (
            node.get_logger()
            if node
            else rclpy.logging.get_logger("orientation_control_early_error")
        )
        logger.fatal(f"Unhandled exception: {e}")
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
