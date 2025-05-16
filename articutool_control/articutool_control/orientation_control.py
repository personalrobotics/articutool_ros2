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
        self.R_handle_imu_pin: Optional[R] = None
        self.R_imu_handle_pin: Optional[R] = None

        try:
            self._setup_pinocchio()
            self.get_logger().info("Pinocchio setup successful.")
        except Exception as e:
            self.get_logger().error(
                f"Pinocchio setup failed: {e}. MODE_FULL_ORIENTATION and calibration may be unavailable.",
            )
            self.pin_model = None
            self.pin_data = None
            self.R_imu_handle_pin = None

        self.current_mode = MODE_DISABLED
        self.target_orientation_jacobase: Optional[R] = None

        # Offsets for MODE_LEVELING (stored in RADIANS)
        self.current_pitch_offset_leveling: float = 0.0
        self.current_roll_offset_leveling: float = 0.0

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
        self.declare_parameter("urdf_path", "", string_desc)  # Path to XACRO or URDF
        self.declare_parameter("robot_base_frame", "j2n6s200_link_base", string_desc)
        self.declare_parameter(
            "articutool_base_link", "atool_handle", string_desc
        )  # Frame F0 for analytic Jacobian
        self.declare_parameter("imu_link_frame", "atool_imu_frame", string_desc)
        self.declare_parameter(
            "tooltip_frame", "tool_tip", string_desc
        )  # Frame whose Y-axis (potentially offset) is controlled
        self.declare_parameter(
            "joint_names", ["atool_joint1", "atool_joint2"], str_array_desc
        )  # Pitch and Roll joints of Articutool
        self.declare_parameter(
            "joint_limits.lower", [-math.pi / 2.0, -math.pi], dbl_array_desc
        )
        self.declare_parameter(
            "joint_limits.upper", [math.pi / 2.0, math.pi], dbl_array_desc
        )
        self.declare_parameter(
            "joint_limits.threshold",
            0.1,  # Radians from limit to start dampening
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
            0.015,  # Radians, error below this is considered zero for leveling
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE),
        )
        self.declare_parameter(
            "leveling_singularity_damp_power",
            5.0,  # Power for singularity dampening factor
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Power for the singularity dampening curve. >1.0 dampens pitch cmd more aggressively as cos(roll) approaches 0.",
            ),
        )

    def _load_parameters(self):
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
        self.leveling_singularity_cos_roll_threshold = self.get_parameter(
            "leveling_singularity_cos_roll_threshold"
        ).value
        self.leveling_error_deadband_rad = self.get_parameter(
            "leveling_error_deadband_rad"
        ).value
        self.leveling_singularity_damp_power = self.get_parameter(
            "leveling_singularity_damp_power"
        ).value
        self.get_logger().debug("Parameters loaded/reloaded.")

    def parameters_callback(self, params: List[Parameter]):
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
                "leveling_singularity_cos_roll_threshold",
                "leveling_error_deadband_rad",
                "leveling_singularity_damp_power",
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
                    "Recalibration is required due to parameter change (e.g., robot_base_frame)."
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
        if not self.xacro_filename:
            self.get_logger().warn(
                "URDF path not provided. Pinocchio setup skipped. MODE_FULL_ORIENTATION and calibration will be unavailable."
            )
            self.pin_model = None
            return
        if not os.path.exists(self.xacro_filename):
            self.get_logger().error(
                f"Xacro/URDF file not found at {self.xacro_filename}. Pinocchio setup failed. MODE_FULL_ORIENTATION and calibration will be unavailable."
            )
            self.pin_model = None
            return

        temp_urdf_path = None
        try:
            self.get_logger().info(
                f"Processing Xacro/URDF file for Pinocchio: {self.xacro_filename}"
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
            T_handle_imu_se3 = T_pinworld_handle.inverse() * T_pinworld_imu
            self.R_handle_imu_pin = R.from_matrix(T_handle_imu_se3.rotation)
            self.R_imu_handle_pin = self.R_handle_imu_pin.inv()
            self.get_logger().info(
                f"Pinocchio-derived R_handle_imu_pin (Handle to IMU, xyzw): {self.R_handle_imu_pin.as_quat()}"
            )
            self.get_logger().info(
                f"Pinocchio-derived R_imu_handle_pin (IMU to Handle, xyzw): {self.R_imu_handle_pin.as_quat()}"
            )

            diff_rot = self.R_IMU_TO_HANDLE_FIXED_SCIPY * self.R_imu_handle_pin.inv()
            angle_diff = diff_rot.magnitude()
            self.get_logger().info(
                f"Angle difference between hardcoded R_IMU_Handle and Pinocchio R_IMU_Handle: {math.degrees(angle_diff):.3f} deg"
            )
            if angle_diff > 0.1:
                self.get_logger().warn(
                    "Significant difference between hardcoded R_IMU_TO_HANDLE_FIXED_SCIPY and Pinocchio-derived R_imu_handle_pin. "
                    "Ensure URDF for 'atool_handle_to_imu_frame' joint matches the hardcoded rotation (0, -pi/2, -pi) and that Pinocchio model is loaded correctly."
                )

            imu_frame_obj = self.pin_model.frames[self.imu_frame_id_pin]
            imu_parent_joint_id = imu_frame_obj.parentJoint
            imu_parent_joint_name = self.pin_model.names[imu_parent_joint_id]
            self.get_logger().info(
                f"Pinocchio: IMU frame '{self.imu_link_name}' (ID {self.imu_frame_id_pin}) is attached to JOINT '{imu_parent_joint_name}' (ID {imu_parent_joint_id})."
            )
            if imu_parent_joint_id == 0:
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
                    self.get_logger().error(
                        f"Articutool joint '{joint_name}' not found in Pinocchio model. MODE_FULL_ORIENTATION will fail."
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
            self.get_logger().error(
                f"Pinocchio setup failed during model processing: {e}. MODE_FULL_ORIENTATION and calibration may be unavailable.",
            )
            self.pin_model = None
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
            # Convert incoming degree offsets to radians for internal use
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
                    f"{'Switching to' if self.current_mode != MODE_LEVELING else 'Updating'} LEVELING mode. "
                    f"Pitch Offset: {request.pitch_offset:.2f} deg ({pitch_offset_rad:.3f} rad), "
                    f"Roll Offset: {request.roll_offset:.2f} deg ({roll_offset_rad:.3f} rad)"
                )
                self._reset_pid_for_mode(MODE_LEVELING)

            self.current_pitch_offset_leveling = pitch_offset_rad  # Store in radians
            self.current_roll_offset_leveling = roll_offset_rad  # Store in radians
            self.current_mode = MODE_LEVELING
            response.success = True
            response.message = "Leveling mode (analytic Jacobian with offsets) enabled."

        elif request.control_mode == MODE_FULL_ORIENTATION:
            if self.pin_model is None:
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

        R_imu_handle_to_use: Optional[R] = None
        source_msg = ""
        if self.R_imu_handle_pin is not None:
            R_imu_handle_to_use = self.R_imu_handle_pin
            source_msg = "Pinocchio-derived"
        else:
            R_imu_handle_to_use = self.R_IMU_TO_HANDLE_FIXED_SCIPY
            source_msg = "hardcoded URDF-based"
            self.get_logger().warn(
                "Calibration using hardcoded R_IMU_TO_HANDLE_FIXED_SCIPY as Pinocchio version is unavailable."
            )

        if R_imu_handle_to_use is None:
            response.message = (
                "Cannot calibrate: R_imu_handle not available from any source."
            )
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
            q_imu_handle = R_imu_handle_to_use

            q_fw_handle = q_fw_imu * q_imu_handle
            self.q_JacoBase_to_FilterWorld_cal = q_jb_handle_tf * q_fw_handle.inv()

            self.is_calibrated = True
            response.success = True
            response.message = (
                f"Calibration successful (using {source_msg} R_imu_handle)."
            )
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
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            response.message = f"Calibration failed: TF lookup error for '{self.robot_base_frame}' to '{self.articutool_base_link_name}': {e}"
            self.get_logger().error(response.message)
        except Exception as e:
            response.message = f"Calibration failed with an unexpected error: {e}"
            self.get_logger().error(response.message)
            self.is_calibrated = False
            self.q_JacoBase_to_FilterWorld_cal = None
        return response

    def feedback_callback(self, msg: Imu):
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
                "Received zero or NaN quaternion from IMU filter. Waiting for valid data.",
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

        if (
            self.current_filterworld_to_imu_raw is None
            or self.current_joint_positions is None
        ):
            if self.current_mode != MODE_DISABLED:
                self.get_logger().warn(
                    f"Mode {self.current_mode}: Prerequisites (IMU/Joints) not met. Commanding zero.",
                    throttle_duration_sec=1.0,
                )
                self._publish_zero_command()
            return

        mode_leveling_ready = True

        mode_full_orientation_ready = (
            self.pin_model is not None
            and self.is_calibrated
            and self.target_orientation_jacobase is not None
        )

        if self.current_mode == MODE_DISABLED:
            return

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
                        "MODE_FULL_ORIENTATION: Prerequisites not met (Pinocchio/Calibration/Target). Commanding zero.",
                        throttle_duration_sec=1.0,
                    )

            if commanded_dq is not None:
                self._publish_command(commanded_dq)
            else:
                self._publish_zero_command()
        except Exception as e:
            self.get_logger().error(
                f"Unhandled exception in control_loop (Mode: {self.current_mode}): {e}",
            )
            self._publish_zero_command()

    def _calculate_leveling_control_analytic_jacobian(
        self, dt: float
    ) -> Optional[np.ndarray]:
        try:
            theta_p_curr = self.current_joint_positions[0]
            theta_r_curr = self.current_joint_positions[1]

            cp, sp = math.cos(theta_p_curr), math.sin(theta_p_curr)
            cr, sr = math.cos(theta_r_curr), math.sin(theta_r_curr)

            # Offsets are already stored in RADIANS
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
            elif dot_prod < (-1.0 + self.EPSILON):
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

            if np.allclose(error_vec_FilterW, 0.0):
                omega_corr_FilterW = np.zeros(3)
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
                if normalized_cr_for_scaling >= 0:
                    pitch_singularity_scale = (
                        normalized_cr_for_scaling**self.leveling_singularity_damp_power
                    )
                else:
                    pitch_singularity_scale = 0.0
                pitch_singularity_scale = np.clip(pitch_singularity_scale, 0.0, 1.0)
                if pitch_singularity_scale < 0.99:
                    self.get_logger().warn(
                        f"Pitch singularity dampening: |cos(roll_joint)|={abs_cr:.4f} (thresh={effective_singularity_threshold:.4f}), "
                        f"scale={pitch_singularity_scale:.4f}. Pitch dq: {dq_calculated[0]:.4f} -> {dq_calculated[0] * pitch_singularity_scale:.4f}",
                        throttle_duration_sec=1.0,
                    )
                dq_calculated[0] *= pitch_singularity_scale

            final_dq_desired = self._dampen_velocities_near_limits(
                self.current_joint_positions, dq_calculated
            )
            return final_dq_desired

        except Exception as e:
            self.get_logger().error(
                f"Error in Analytic Jacobian Leveling Control (with offsets): {e}"
            )
            import traceback

            self.get_logger().error(traceback.format_exc())
            return np.zeros(len(self.articutool_joint_names))

    def _calculate_full_orientation_control(self, dt: float) -> Optional[np.ndarray]:
        if self.pin_model is None or self.R_imu_handle_pin is None:
            self.get_logger().error(
                "Full Orientation: Pinocchio model or R_imu_handle_pin not available.",
                throttle_duration_sec=1.0,
            )
            return None
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
        if (
            self.current_filterworld_to_imu_raw is None
            or self.current_joint_positions is None
        ):
            self.get_logger().warn(
                "Full Orientation: IMU or Joint states missing.",
                throttle_duration_sec=1.0,
            )
            return None

        try:
            q_jb_fw_cal = self.q_JacoBase_to_FilterWorld_cal
            q_fw_imu_raw = self.current_filterworld_to_imu_raw
            q_jb_imu = q_jb_fw_cal * q_fw_imu_raw

            q_pin_config = self._get_pinocchio_config()
            q_imu_tooltip_current_pin = self._get_pinocchio_imu_tooltip_orientation(
                q_pin_config
            )
            if q_imu_tooltip_current_pin is None:
                self.get_logger().error(
                    "Full Orientation: Failed to get R_IMU_Tooltip from Pinocchio."
                )
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
                self.get_logger().error(
                    "Full Orientation: Failed to calculate joint velocities from Pinocchio Jacobian."
                )
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
            )
            import traceback

            self.get_logger().error(traceback.format_exc())
            return None

    def _get_pinocchio_config(self) -> np.ndarray:
        if self.pin_model is None:
            raise ValueError("Pinocchio model not loaded for _get_pinocchio_config.")
        if self.current_joint_positions is None:
            raise ValueError("Joint positions are None in _get_pinocchio_config.")
        if len(self.current_joint_positions) != len(self.articutool_joint_names):
            raise ValueError(
                f"Mismatch joint positions ({len(self.current_joint_positions)}) vs names ({len(self.articutool_joint_names)})"
            )

        q = pin.neutral(self.pin_model)

        for i, joint_name in enumerate(self.articutool_joint_names):
            if not self.pin_model.existJointName(joint_name):
                raise ValueError(
                    f"Joint '{joint_name}' not found in Pinocchio model during config creation."
                )

            joint_id = self.pin_model.getJointId(joint_name)
            if joint_id < 1 or joint_id >= self.pin_model.njoints:
                raise ValueError(f"Invalid joint ID {joint_id} for {joint_name}")

            if (
                self.pin_model.joints[joint_id].nq == 1
                and self.pin_model.joints[joint_id].nv == 1
            ):
                q_idx = self.pin_model.joints[joint_id].idx_q
                q[q_idx] = self.current_joint_positions[i]
            else:
                self.get_logger().warn(
                    f"Joint '{joint_name}' has nq={self.pin_model.joints[joint_id].nq}, nv={self.pin_model.joints[joint_id].nv}. Assuming simple assignment.",
                    throttle_duration_sec=5.0,
                )
                q_idx = self.pin_model.joints[joint_id].idx_q
                if q_idx < len(q):
                    q[q_idx] = self.current_joint_positions[i]
                else:
                    raise ValueError(
                        f"idx_q {q_idx} out of bounds for q (len {len(q)}) for joint {joint_name}"
                    )
        return q

    def _get_pinocchio_imu_tooltip_orientation(
        self, q_pin_config: np.ndarray
    ) -> Optional[R]:
        if (
            self.pin_model is None
            or self.pin_data is None
            or self.imu_frame_id_pin < 0
            or self.tooltip_frame_id_pin < 0
        ):
            self.get_logger().error(
                "Pinocchio model/data/frames not initialized for FK in _get_pinocchio_imu_tooltip_orientation.",
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
                "Pinocchio model/data/frames not initialized for Jacobian in _calculate_joint_velocities_pinocchio_jacobian.",
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
                    f"Pinocchio Jacobian shape error: {J_tooltip_angular_local_articutool_joints.shape}. Expected (3,2)."
                )
                return np.zeros(2)

            if np.allclose(
                J_tooltip_angular_local_articutool_joints, 0, atol=self.EPSILON
            ):
                self.get_logger().warn(
                    "Pinocchio Jacobian for Articutool joints (angular part) is all zeros. Possible singularity or model issue.",
                    throttle_duration_sec=2.0,
                )

            J_pinv = np.linalg.pinv(
                J_tooltip_angular_local_articutool_joints,
                rcond=self.JACOBIAN_PINV_RCOND,
            )
            dq_desired = J_pinv @ omega_desired_tooltip_local

            if dq_desired.shape != (2,):
                self.get_logger().error(
                    f"Calculated dq_desired shape error: {dq_desired.shape}. Expected (2,)."
                )
                return np.zeros(2)

            return dq_desired
        except Exception as e:
            self.get_logger().error(
                f"Error calculating joint velocities with Pinocchio Jacobian: {e}",
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
                "Dampening: current_q is None. Not dampening.",
                throttle_duration_sec=2.0,
            )
            return desired_dq

        if len(current_q) != len(self.articutool_joint_names) or len(desired_dq) != len(
            self.articutool_joint_names
        ):
            self.get_logger().error(
                f"Dampening: Length mismatch. q({len(current_q)}), dq({len(desired_dq)}), names({len(self.articutool_joint_names)}). Not dampening."
            )
            if len(desired_dq) != len(self.articutool_joint_names):
                return np.zeros(len(self.articutool_joint_names))
            return desired_dq

        for i in range(len(self.articutool_joint_names)):
            q_i, dq_i = current_q[i], desired_dq[i]
            lower_limit, upper_limit = (
                self.joint_limits_lower[i],
                self.joint_limits_upper[i],
            )
            threshold = max(self.joint_limit_threshold, self.EPSILON)
            damp_power = self.joint_limit_dampening_factor

            scale = 1.0

            if dq_i < -self.EPSILON:
                distance_to_lower = q_i - lower_limit
                if distance_to_lower < threshold:
                    current_scale_factor = distance_to_lower / threshold
                    scale = np.clip(current_scale_factor, 0.0, 1.0) ** damp_power
            elif dq_i > self.EPSILON:
                distance_to_upper = upper_limit - q_i
                if distance_to_upper < threshold:
                    current_scale_factor = distance_to_upper / threshold
                    scale = np.clip(current_scale_factor, 0.0, 1.0) ** damp_power

            dampened_dq[i] *= scale

        if (
            not np.allclose(desired_dq, dampened_dq, atol=1e-4)
            and self.get_logger().get_effective_level()
            <= rclpy.logging.LoggingSeverity.DEBUG
        ):
            self.get_logger().debug(
                f"Dampening: q={np.round(current_q, 3)}, original_dq={np.round(desired_dq, 3)}, dampened_dq={np.round(dampened_dq, 3)}"
            )
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
