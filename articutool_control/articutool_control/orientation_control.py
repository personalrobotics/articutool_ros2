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
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from action_msgs.msg import GoalStatus
import time

from geometry_msgs.msg import Quaternion, QuaternionStamped, Vector3
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

from articutool_interfaces.srv import SetOrientationControl
from articutool_interfaces.msg import ImuCalibrationStatus
from articutool_interfaces.action import ExecuteArticutoolPrimitive
from .primitives import (
    PrimitiveAction,
    SettleTossPrimitive,
    TwirlPrimitive,
    VibratePrimitive,
    DepositBitePrimitive,
    NoodleShedPrimitive,
)

import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

import os
import tempfile
import subprocess
import traceback
from typing import Optional, Tuple, List, Dict, Any
from ament_index_python.packages import get_package_share_directory

MODE_DISABLED = SetOrientationControl.Request.MODE_DISABLED
MODE_LEVELING = SetOrientationControl.Request.MODE_LEVELING
MODE_FULL_ORIENTATION = SetOrientationControl.Request.MODE_FULL_ORIENTATION


class ArticutoolController(Node):
    WORLD_Z_UP_VECTOR = np.array([0.0, 0.0, 1.0])
    EPSILON = 1e-6
    JACOBIAN_PINV_RCOND = 1e-3

    _R_handle_to_imu_urdf = R.from_euler(
        "xyz", [0, -math.pi / 2, -math.pi], degrees=False
    )
    R_IMU_TO_HANDLE_FIXED_SCIPY = _R_handle_to_imu_urdf.inv()

    def __init__(self):
        super().__init__("articutool_controller")

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
        except Exception as e:
            self.get_logger().error(
                f"Pinocchio setup failed during init: {e} {traceback.format_exc()}"
            )
            self.pin_model = None

        self.current_orientation_control_mode: int = MODE_DISABLED
        self.target_orientation_jacobase: Optional[R] = None
        self.current_pitch_offset_leveling: float = 0.0
        self.current_roll_offset_leveling: float = 0.0

        self.last_error_leveling = np.zeros(3)
        self.integral_error_leveling = np.zeros(3)
        self.last_error_full_orientation = np.zeros(3)
        self.integral_error_full_orientation = np.zeros(3)

        self.active_primitive: Optional[PrimitiveAction] = None
        self.active_primitive_goal_handle: Optional[ServerGoalHandle] = None
        self.primitive_is_initialized: bool = False

        self.last_uncalibrated_imu_msg_time: Optional[Time] = None
        self.current_filterworld_to_imu_raw: Optional[R] = None
        self.current_linear_accel_imu: Optional[np.ndarray] = None
        self.current_angular_velocity_imu: Optional[np.ndarray] = None
        self.last_calibrated_imu_msg_time: Optional[Time] = None
        self.current_RobotBase_to_IMUframe_calibrated: Optional[R] = None
        self.is_externally_calibrated: bool = False
        self.last_external_calibration_time: Optional[Time] = None
        self.current_joint_positions: Optional[np.ndarray] = None
        self.last_time: Optional[Time] = None

        # --- Primitive Registry ---
        self.primitive_registry: Dict[str, type[PrimitiveAction]] = {
            "TWIRL_CW": TwirlPrimitive,
            "TWIRL_CCW": TwirlPrimitive,
            "VIBRATE_ROLL": VibratePrimitive,
            "DEPOSIT_BITE": DepositBitePrimitive,
            "SETTLE_TOSS": SettleTossPrimitive,
            "NOODLE_SHED": NoodleShedPrimitive,
            # Add new primitives here, e.g.:
            # "SIFT_AND_CENTER": SiftAndCenterPrimitive,
        }

        self.add_on_set_parameters_callback(self.parameters_callback)

        self.orientation_mode_srv = self.create_service(
            SetOrientationControl,
            "~/set_orientation_control_mode",
            self.set_orientation_control_mode_callback,
        )
        self.primitive_action_server = ActionServer(
            self,
            ExecuteArticutoolPrimitive,
            "~/execute_primitive",
            execute_callback=self.execute_primitive_callback,
            goal_callback=self.primitive_goal_callback,
            cancel_callback=self.primitive_cancel_callback,
        )

        self.uncalibrated_imu_sub = self.create_subscription(
            Imu, self.uncalibrated_imu_topic, self.uncalibrated_feedback_callback, 1
        )
        self.calibrated_imu_sub = self.create_subscription(
            Imu, self.calibrated_imu_topic, self.calibrated_feedback_callback, 1
        )
        self.calibration_status_sub = self.create_subscription(
            ImuCalibrationStatus,
            self.calibration_status_topic,
            self.calibration_status_callback,
            rclpy.qos.QoSProfile(
                depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
            ),
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
        self.get_logger().info(f"{self.get_name()} node started.")

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
            "joint_limits.lower", [-math.pi / 2.0, -13 * math.pi], dbl_array_desc
        )
        self.declare_parameter(
            "joint_limits.upper", [math.pi / 2.0, 13 * math.pi], dbl_array_desc
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
        self.declare_parameter(
            "leveling_singularity_cos_roll_threshold",
            0.6,
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
            raise ValueError("Articutool joint_names must have 2 entries.")
        self.joint_limits_lower = np.array(
            self.get_parameter("joint_limits.lower").value
        )
        self.joint_limits_upper = np.array(
            self.get_parameter("joint_limits.upper").value
        )
        if len(self.joint_limits_lower) != 2 or len(self.joint_limits_upper) != 2:
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
        self.get_logger().debug("ArticutoolController parameters (re)loaded.")

    def parameters_callback(self, params: List[Parameter]):
        self.get_logger().info("Parameters callback triggered.")
        changed_rate = False
        for param in params:
            if param.name == "loop_rate":
                if (
                    param.type_ == ParameterType.PARAMETER_DOUBLE
                    and param.value != self.rate
                ):
                    changed_rate = True
        self._load_parameters()
        if changed_rate and self.timer is not None:
            self.timer.cancel()
            if self.rate > 0:
                self.timer = self.create_timer(1.0 / self.rate, self.control_loop)
                self.get_logger().info(f"Control loop rate updated to {self.rate} Hz.")
            else:
                self.get_logger().error(
                    "Loop rate set to zero or negative. Control loop stopped."
                )
                self.timer = None
        return SetParametersResult(successful=True)

    def _setup_pinocchio(self):
        if not self.xacro_filename:
            self.get_logger().warn("URDF path not provided. Pinocchio setup skipped.")
            self.pin_model = None
            return
        resolved_xacro_path = self.xacro_filename
        if "package://" in self.xacro_filename:
            try:
                package_name = self.xacro_filename.split("package://")[1].split("/")[0]
                relative_path = "/".join(
                    self.xacro_filename.split("package://")[1].split("/")[1:]
                )
                package_share_directory = get_package_share_directory(package_name)
                resolved_xacro_path = os.path.join(
                    package_share_directory, relative_path
                )
                self.get_logger().info(
                    f"Resolved URDF path from '{self.xacro_filename}' to '{resolved_xacro_path}'"
                )
            except Exception as e:
                self.get_logger().error(
                    f"Could not resolve package path {self.xacro_filename}: {e}"
                )
                self.pin_model = None
                return
        if not os.path.exists(resolved_xacro_path):
            self.get_logger().error(
                f"Xacro/URDF file not found at resolved path: {resolved_xacro_path}"
            )
            self.pin_model = None
            return
        temp_urdf_path = None
        try:
            self.get_logger().info(
                f"Processing Xacro/URDF for Pinocchio: {resolved_xacro_path}"
            )
            if resolved_xacro_path.endswith(".xacro"):
                process = subprocess.run(
                    ["ros2", "run", "xacro", "xacro", resolved_xacro_path],
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
            elif resolved_xacro_path.endswith(".urdf"):
                model_file_to_load = resolved_xacro_path
            else:
                self.get_logger().error(
                    f"Unsupported robot model file extension: {resolved_xacro_path}."
                )
                self.pin_model = None
                return
            self.pin_model = pin.buildModelFromUrdf(model_file_to_load)
            self.pin_data = self.pin_model.createData()
            self.get_logger().info(
                f"Pinocchio model loaded: {self.pin_model.name}, Nq={self.pin_model.nq}, Nv={self.pin_model.nv}"
            )
            if self.pin_model.existFrame(self.imu_link_name):
                self.imu_frame_id_pin = self.pin_model.getFrameId(self.imu_link_name)
            else:
                self.get_logger().error(
                    f"Pinocchio: IMU frame '{self.imu_link_name}' not found in loaded URDF!"
                )
                self.pin_model = None
                return
            if self.pin_model.existFrame(self.tooltip_link_name):
                self.tooltip_frame_id_pin = self.pin_model.getFrameId(
                    self.tooltip_link_name
                )
            else:
                self.get_logger().error(
                    f"Pinocchio: Tooltip frame '{self.tooltip_link_name}' not found in loaded URDF!"
                )
                self.pin_model = None
                return
            self.articutool_joint_ids_pin = []
            self.articutool_q_indices_pin = []
            self.articutool_v_indices_pin = []
            for joint_name in self.articutool_joint_names:
                if not self.pin_model.existJointName(joint_name):
                    self.get_logger().error(
                        f"Articutool joint '{joint_name}' (from parameters) not found in Pinocchio model from URDF."
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
                f"Articutool Pinocchio joint IDs: {self.articutool_joint_ids_pin}, q_indices: {self.articutool_q_indices_pin}, v_indices: {self.articutool_v_indices_pin}"
            )
        except subprocess.CalledProcessError as e_sub:
            self.get_logger().error(
                f"Xacro processing failed: {e_sub}\nStdout: {e_sub.stdout}\nStderr: {e_sub.stderr}"
            )
            self.pin_model = None
        except Exception as e:
            self.get_logger().error(
                f"Pinocchio setup failed: {e}\n{traceback.format_exc()}"
            )
            self.pin_model = None
        finally:
            if temp_urdf_path and os.path.exists(temp_urdf_path):
                try:
                    os.unlink(temp_urdf_path)
                except OSError as e_unlink:
                    self.get_logger().error(
                        f"Failed to delete temp URDF file {temp_urdf_path}: {e_unlink}"
                    )

    def set_orientation_control_mode_callback(
        self,
        request: SetOrientationControl.Request,
        response: SetOrientationControl.Response,
    ):
        self.get_logger().info(
            f"SetOrientationControlMode Request: mode={request.control_mode}, "
            f"pitch_offset={request.pitch_offset:.2f} deg, roll_offset={request.roll_offset:.2f} deg, "
            f"target_quat (xyzw if provided)=({request.target_orientation_robot_base.x:.2f}, "
            f"{request.target_orientation_robot_base.y:.2f}, {request.target_orientation_robot_base.z:.2f}, "
            f"{request.target_orientation_robot_base.w:.2f})"
        )

        if (
            self.active_primitive_goal_handle is not None
            and self.active_primitive_goal_handle.is_active
        ):
            self.get_logger().warn(
                f"Primitive action '{self.current_primitive_name}' active. Aborting it."
            )
            self.active_primitive_goal_handle.abort()
            self.active_primitive_goal_handle = None
            self.current_primitive_name = None
            self.current_primitive_params = []
            self.primitive_internal_state = {}
        self.current_orientation_control_mode = request.control_mode
        self.target_orientation_jacobase = None
        self.current_pitch_offset_leveling = 0.0
        self.current_roll_offset_leveling = 0.0
        response.success = True
        if self.current_orientation_control_mode == MODE_FULL_ORIENTATION:
            is_calib_fresh = (
                self.last_external_calibration_time is not None
                and (
                    self.get_clock().now() - self.last_external_calibration_time
                ).nanoseconds
                / 1e9
                <= self.max_time_since_last_calibration_sec
            )
            if (
                self.pin_model is None
                or not self.is_externally_calibrated
                or not is_calib_fresh
            ):
                msg = "Cannot enable FULL_ORIENTATION: Pinocchio/Calibration prerequisites not met."
                self.get_logger().error(msg)
                response.success = False
                response.message = msg
                self.current_orientation_control_mode = MODE_DISABLED
            else:
                try:
                    self.target_orientation_jacobase = R.from_quat(
                        [
                            request.target_orientation_robot_base.x,
                            request.target_orientation_robot_base.y,
                            request.target_orientation_robot_base.z,
                            request.target_orientation_robot_base.w,
                        ]
                    )
                    self._reset_pid_for_mode(MODE_FULL_ORIENTATION)
                    response.message = "MODE_FULL_ORIENTATION enabled."
                except Exception as e:
                    response.success = False
                    response.message = f"Error setting target for Full Orientation: {e}"
                    self.get_logger().error(response.message)
                    self.current_orientation_control_mode = MODE_DISABLED
        elif self.current_orientation_control_mode == MODE_LEVELING:
            self.current_pitch_offset_leveling = math.radians(request.pitch_offset)
            self.current_roll_offset_leveling = math.radians(request.roll_offset)
            self._reset_pid_for_mode(MODE_LEVELING)
            response.message = "MODE_LEVELING enabled."
        elif self.current_orientation_control_mode == MODE_DISABLED:
            response.message = "MODE_DISABLED enabled."
        else:
            response.success = False
            response.message = f"Invalid control_mode: {request.control_mode}"
            self.get_logger().error(response.message)
            self.current_orientation_control_mode = MODE_DISABLED
        if (
            not response.success
            or self.current_orientation_control_mode == MODE_DISABLED
        ):
            self._publish_zero_command()
        self.get_logger().info(response.message)
        return response

    def primitive_goal_callback(self, goal_request: ExecuteArticutoolPrimitive.Goal):
        self.get_logger().info(
            f"Received primitive goal request: '{goal_request.primitive_name}'"
        )
        if self.current_orientation_control_mode != MODE_DISABLED:
            msg = f"Rejecting primitive '{goal_request.primitive_name}'. Articutool not in MODE_DISABLED."
            self.get_logger().error(msg)
            return GoalResponse.REJECT
        if (
            self.active_primitive_goal_handle is not None
            and self.active_primitive_goal_handle.is_active
        ):
            msg = f"Rejecting primitive '{goal_request.primitive_name}'. Another primitive '{self.current_primitive_name}' active."
            self.get_logger().warn(msg)
            return GoalResponse.REJECT
        self.get_logger().info(
            f"Accepting primitive goal: '{goal_request.primitive_name}'"
        )
        return GoalResponse.ACCEPT

    def execute_primitive_callback(self, goal_handle: ServerGoalHandle):
        primitive_name = goal_handle.request.primitive_name
        self.get_logger().info(f"Executing primitive request: '{primitive_name}'")

        if primitive_name not in self.primitive_registry:
            self.get_logger().error(f"Unknown primitive '{primitive_name}'. Aborting.")
            goal_handle.abort()
            result = ExecuteArticutoolPrimitive.Result()
            result.success = False
            result.message = f"Unknown primitive name: {primitive_name}"
            return result

        if self.current_joint_positions is None:
            self.get_logger().error(
                "Cannot start primitive: current joint positions are unknown. Aborting."
            )
            goal_handle.abort()
            result = ExecuteArticutoolPrimitive.Result()
            result.success = False
            result.message = "Initial joint positions unknown."
            return result

        # Instantiate the correct primitive class
        primitive_class = self.primitive_registry[primitive_name]
        params = list(goal_handle.request.parameters)

        # Special handling for TWIRL_CCW
        if primitive_name == "TWIRL_CCW":
            if params:
                params[0] = -abs(params[0])  # Make target rotations negative

        self.active_primitive = primitive_class(
            node_logger=self.get_logger(), params=params
        )
        self.active_primitive_goal_handle = goal_handle
        self.primitive_is_initialized = False

        # Call the primitive's own start method
        self.active_primitive.start(self.current_joint_positions)
        self.primitive_is_initialized = True

        # The rest of the action lifecycle is managed by the control_loop checking the instance
        # and this callback waiting for completion.

        try:
            while (
                rclpy.ok()
                and self.active_primitive_goal_handle == goal_handle
                and goal_handle.is_active
            ):
                time.sleep(0.05)  # Poll for completion

            self.get_logger().debug(
                f"Primitive '{primitive_name}' execution loop finished. Final status: {goal_handle.status}"
            )

        except Exception as e:
            self.get_logger().error(
                f"Exception in execute_primitive_callback for '{primitive_name}': {e}\n{traceback.format_exc()}"
            )
            if goal_handle.is_active:
                goal_handle.abort()

        finally:
            # Construct result based on final handle state
            result = ExecuteArticutoolPrimitive.Result()
            if goal_handle.status == GoalStatus.STATUS_SUCCEEDED:
                result.success = True
                result.message = f"Primitive '{primitive_name}' succeeded."
            else:  # Aborted or Canceled
                result.success = False
                result.message = f"Primitive '{primitive_name}' did not succeed (status: {goal_handle.status})."

            if self.current_joint_positions is not None:
                result.final_joint_values = list(self.current_joint_positions)

            if self.active_primitive_goal_handle == goal_handle:
                self.active_primitive = None  # Clear instance
                self.active_primitive_goal_handle = None
                self.primitive_is_initialized = False  # <-- RESET FLAG HERE

            return result

    def primitive_cancel_callback(self, cancel_request_goal_handle: ServerGoalHandle):
        self.get_logger().info(
            f"Received cancel request for primitive action ID: {cancel_request_goal_handle.goal_id.uuid}"
        )
        return CancelResponse.ACCEPT

    def uncalibrated_feedback_callback(self, msg: Imu):
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
        elif self.current_filterworld_to_imu_raw is None:
            self.get_logger().warn(
                "UNCALIBRATED IMU: Zero/NaN quaternion.", throttle_duration_sec=1.0
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
        elif self.current_RobotBase_to_IMUframe_calibrated is None:
            self.get_logger().warn(
                "CALIBRATED IMU: Zero/NaN quaternion.", throttle_duration_sec=1.0
            )

    def calibration_status_callback(self, msg: ImuCalibrationStatus):
        old_calib_status = self.is_externally_calibrated
        self.is_externally_calibrated = msg.is_yaw_calibrated
        self.last_external_calibration_time = Time.from_msg(
            msg.last_yaw_calibration_time
        )
        if old_calib_status != self.is_externally_calibrated:
            self.get_logger().info(
                f"External calibration status changed. Calibrated: {self.is_externally_calibrated}"
            )
        if (
            not self.is_externally_calibrated
            and self.current_orientation_control_mode == MODE_FULL_ORIENTATION
        ):
            self.get_logger().error(
                "External calibration lost during MODE_FULL_ORIENTATION! Switching to MODE_DISABLED."
            )
            self.current_orientation_control_mode = MODE_DISABLED
            self._reset_pid_for_mode(MODE_FULL_ORIENTATION)
            self._publish_zero_command()

    def joint_state_callback(self, msg: JointState):
        if self.current_joint_positions is None or len(
            self.current_joint_positions
        ) != len(self.articutool_joint_names):
            self.current_joint_positions = np.zeros(len(self.articutool_joint_names))
        for i, name_in_msg in enumerate(msg.name):
            try:
                idx_in_our_array = self.articutool_joint_names.index(name_in_msg)
                if i < len(msg.position):
                    self.current_joint_positions[idx_in_our_array] = msg.position[i]
            except ValueError:
                pass

    def _update_active_primitive(self, dt: float) -> Tuple[np.ndarray, bool]:
        # This check should be redundant if logic is correct, but is a good safeguard
        if self.active_primitive is None or self.active_primitive_goal_handle is None:
            return np.zeros(2), True

        if not self.active_primitive_goal_handle.is_active:
            self.active_primitive = None
            self.active_primitive_goal_handle = None
            return np.zeros(2), True
        if not self.primitive_is_initialized:
            # Not ready yet, command zero velocity and wait for initialization
            return np.zeros(2), False
        if self.active_primitive.is_finished:
            if self.active_primitive.was_successful:
                self.active_primitive_goal_handle.succeed()
            else:
                self.active_primitive_goal_handle.abort()
            # The execute_primitive_callback will see the handle is no longer active and return
            return np.zeros(2), True

        if self.current_joint_positions is None:
            self.get_logger().error(
                "Aborting primitive: joint positions became unknown during execution."
            )
            self.active_primitive_goal_handle.abort()
            return np.zeros(2), True

        # Run the primitive's update logic
        dq_command, feedback_string, percent_complete = self.active_primitive.update(
            dt, self.current_joint_positions
        )

        # Publish feedback
        feedback_msg = ExecuteArticutoolPrimitive.Feedback()
        feedback_msg.feedback_string = feedback_string
        feedback_msg.percent_complete = percent_complete
        if self.current_joint_positions is not None:
            feedback_msg.current_joint_values = list(self.current_joint_positions)
        self.active_primitive_goal_handle.publish_feedback(feedback_msg)

        return dq_command, False  # Not finished yet

    def control_loop(self):
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= self.EPSILON:
            return
        final_raw_commanded_dq = np.zeros(2)
        if (
            self.active_primitive_goal_handle is not None
            and self.active_primitive_goal_handle.is_active
        ):
            if self.current_orientation_control_mode != MODE_DISABLED:
                self.get_logger().error(
                    f"CRITICAL LOGIC ERROR: Primitive '{self.current_primitive_name}' active, but global mode is '{self.current_orientation_control_mode}' (NOT DISABLED). Primitive takes precedence.",
                    throttle_duration_sec=5.0,
                )
            dq_primitive_step, _ = self._update_active_primitive(dt)
            if dq_primitive_step is not None:
                final_raw_commanded_dq = dq_primitive_step
        elif self.current_orientation_control_mode == MODE_LEVELING:
            mode_leveling_ready = (
                self.current_filterworld_to_imu_raw is not None
                and self.current_joint_positions is not None
            )
            if mode_leveling_ready:
                dq_orientation = self._calculate_leveling_control_analytic_jacobian(dt)
                if dq_orientation is not None:
                    final_raw_commanded_dq = dq_orientation
            else:
                self.get_logger().warn(
                    "MODE_LEVELING: Prerequisites not met.", throttle_duration_sec=2.0
                )
        elif self.current_orientation_control_mode == MODE_FULL_ORIENTATION:
            is_calib_fresh = (
                self.last_external_calibration_time is not None
                and (now - self.last_external_calibration_time).nanoseconds / 1e9
                <= self.max_time_since_last_calibration_sec
            )
            mode_full_orientation_ready = (
                self.pin_model is not None
                and self.is_externally_calibrated
                and is_calib_fresh
                and self.target_orientation_jacobase is not None
                and self.current_RobotBase_to_IMUframe_calibrated is not None
                and self.current_joint_positions is not None
            )
            if mode_full_orientation_ready:
                dq_orientation = self._calculate_full_orientation_control(dt)
                if dq_orientation is not None:
                    final_raw_commanded_dq = dq_orientation
            else:
                self.get_logger().warn(
                    "MODE_FULL_ORIENTATION: Prerequisites not met.",
                    throttle_duration_sec=2.0,
                )
        if self.current_joint_positions is not None:
            final_dq_after_limits = self._enforce_joint_limits_predictive(
                self.current_joint_positions, final_raw_commanded_dq, dt
            )
            self._publish_command(final_dq_after_limits)
        elif np.any(final_raw_commanded_dq):
            self.get_logger().warn(
                "Commanding non-zero Articutool vels without current joint_states. Publishing raw.",
                throttle_duration_sec=5.0,
            )
            self._publish_command(final_raw_commanded_dq)
        elif (
            self.active_primitive_goal_handle is None
            and self.current_orientation_control_mode == MODE_DISABLED
        ):
            self._publish_zero_command()

    def _calculate_leveling_control_analytic_jacobian(
        self, dt: float
    ) -> Optional[np.ndarray]:
        if (
            self.current_filterworld_to_imu_raw is None
            or self.current_joint_positions is None
        ):
            return None
        try:
            theta_p_curr, theta_r_curr = self.current_joint_positions
            cp, sp = math.cos(theta_p_curr), math.sin(theta_p_curr)
            cr, sr = math.cos(theta_r_curr), math.sin(theta_r_curr)
            phi_o, psi_o = (
                self.current_pitch_offset_leveling,
                self.current_roll_offset_leveling,
            )
            c_phi_o, s_phi_o = math.cos(phi_o), math.sin(phi_o)
            c_psi_o, s_psi_o = math.cos(psi_o), math.sin(psi_o)
            y_eff = np.array([-s_psi_o * c_phi_o, c_psi_o * c_phi_o, s_phi_o])
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
                arbitrary_axis = (
                    np.array([1.0, 0.0, 0.0])
                    if np.linalg.norm(
                        np.cross(y_eff_in_FilterW_curr, np.array([1.0, 0.0, 0.0]))
                    )
                    > self.EPSILON
                    else np.array([0.0, 1.0, 0.0])
                )
                rotation_axis = np.cross(y_eff_in_FilterW_curr, arbitrary_axis)
                if np.linalg.norm(rotation_axis) > self.EPSILON:
                    error_vec_FilterW = (
                        rotation_axis / np.linalg.norm(rotation_axis)
                    ) * math.pi
                else:
                    error_vec_FilterW = np.array([math.pi, 0.0, 0.0])
            if np.allclose(error_vec_FilterW, 0.0, atol=self.EPSILON):
                omega_corr_FilterW = np.zeros(3)
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
                    f"Leveling Jacobian shape error: {J_eff_in_F0.shape}"
                )
                return None
            try:
                J_eff_in_F0_pinv = np.linalg.pinv(
                    J_eff_in_F0, rcond=self.JACOBIAN_PINV_RCOND
                )
            except np.linalg.LinAlgError:
                self.get_logger().warn("Leveling Jacobian pseudo-inverse failed.")
                return None
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
            self.get_logger().error(
                f"LvlCtrl Err: {e} {traceback.format_exc()}", throttle_duration_sec=5.0
            )
            return None

    def _calculate_full_orientation_control(self, dt: float) -> Optional[np.ndarray]:
        if (
            self.pin_model is None
            or not self.is_externally_calibrated
            or self.target_orientation_jacobase is None
            or self.current_RobotBase_to_IMUframe_calibrated is None
            or self.current_joint_positions is None
        ):
            return None
        try:
            q_jb_imu = self.current_RobotBase_to_IMUframe_calibrated
            q_pin_config = self._get_pinocchio_config()
            if q_pin_config is None:
                self.get_logger().warn(
                    "Pinocchio config is None in FullOrient.", throttle_duration_sec=2.0
                )
                return None
            R_imu_tooltip_current_pin = self._get_pinocchio_imu_tooltip_orientation(
                q_pin_config
            )
            if R_imu_tooltip_current_pin is None:
                self.get_logger().warn(
                    "R_imu_tooltip is None in FullOrient.", throttle_duration_sec=2.0
                )
                return None
            q_jb_tooltip_current = q_jb_imu * R_imu_tooltip_current_pin
            q_error_jacobase = (
                self.target_orientation_jacobase * q_jb_tooltip_current.inv()
            )
            error_vec_jacobase = q_error_jacobase.as_rotvec()
            if np.allclose(error_vec_jacobase, 0.0, atol=self.EPSILON):
                omega_desired_jacobase = np.zeros(3)
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
            return dq_desired
        except Exception as e:
            self.get_logger().error(
                f"FullOrient Err: {e} {traceback.format_exc()}",
                throttle_duration_sec=5.0,
            )
            return None

    def _get_pinocchio_config(self) -> Optional[np.ndarray]:
        if (
            self.pin_model is None
            or self.current_joint_positions is None
            or len(self.current_joint_positions) != len(self.articutool_joint_names)
        ):
            self.get_logger().warn(
                "_get_pinocchio_config: Preconditions not met.",
                throttle_duration_sec=5.0,
            )
            return None
        try:
            # self.get_logger().debug(f"Current Articutool Joint Positions for Pinocchio: {self.current_joint_positions}")
            q = pin.neutral(self.pin_model)
            if self.pin_model.nq > len(q):
                q_expanded = np.zeros(self.pin_model.nq)
                q_expanded[: len(q)] = q
                q = q_expanded
            for i, joint_name in enumerate(self.articutool_joint_names):
                if not self.pin_model.existJointName(joint_name):
                    self.get_logger().error(
                        f"Joint '{joint_name}' unexpectedly not in Pinocchio model for config."
                    )
                    return None
                joint_id = self.pin_model.getJointId(joint_name)
                joint_model = self.pin_model.joints[joint_id]
                idx_q, nq_joint = joint_model.idx_q, joint_model.nq
                val = self.current_joint_positions[i]
                if idx_q + nq_joint > len(q):
                    self.get_logger().error(
                        f"Pinocchio q vector too short for joint {joint_name}."
                    )
                    return None
                if nq_joint == 1:
                    q[idx_q] = val
                elif nq_joint == 2:
                    q[idx_q], q[idx_q + 1] = math.cos(val), math.sin(val)
                else:
                    q[idx_q] = val
            return q
        except Exception as e:
            self.get_logger().error(
                f"Pinocchio config error: {e} {traceback.format_exc()}"
            )
            return None

    def _get_pinocchio_imu_tooltip_orientation(
        self, q_pin_config: Optional[np.ndarray]
    ) -> Optional[R]:
        if (
            self.pin_model is None
            or self.pin_data is None
            or self.imu_frame_id_pin < 0
            or self.tooltip_frame_id_pin < 0
            or q_pin_config is None
        ):
            self.get_logger().warn(
                "_get_pinocchio_imu_tooltip_orientation: Preconditions not met.",
                throttle_duration_sec=5.0,
            )
            return None
        try:
            # self.get_logger().debug(f"_get_pinocchio_imu_tooltip_orientation: Using q_pin_config: {q_pin_config}")
            pin.forwardKinematics(self.pin_model, self.pin_data, q_pin_config)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            T_world_imu = self.pin_data.oMf[self.imu_frame_id_pin]
            T_world_tooltip = self.pin_data.oMf[self.tooltip_frame_id_pin]

            imu_rotation_matrix = T_world_imu.rotation
            tooltip_rotation_matrix = T_world_tooltip.rotation

            # Check if matrices are valid rotation matrices (determinant approx +1)
            is_imu_rot_valid = (
                abs(np.linalg.det(imu_rotation_matrix) - 1.0) < self.EPSILON
            )
            is_tooltip_rot_valid = (
                abs(np.linalg.det(tooltip_rotation_matrix) - 1.0) < self.EPSILON
            )

            if not (is_imu_rot_valid and is_tooltip_rot_valid):
                self.get_logger().warn(
                    "Invalid rotation matrix from Pinocchio FK (determinant not approx +1). "
                    f"IMU det: {np.linalg.det(imu_rotation_matrix):.4f} (trace: {imu_rotation_matrix.trace():.4f}), "
                    f"Tooltip det: {np.linalg.det(tooltip_rotation_matrix):.4f} (trace: {tooltip_rotation_matrix.trace():.4f})",
                    throttle_duration_sec=2.0,
                )
                return None

            T_imu_tooltip_se3 = T_world_imu.inverse() * T_world_tooltip
            return R.from_matrix(T_imu_tooltip_se3.rotation)
        except Exception as e:
            self.get_logger().error(f"Pinocchio FK error: {e} {traceback.format_exc()}")
            return None

    def _calculate_joint_velocities_pinocchio_jacobian(
        self,
        q_pin_config: Optional[np.ndarray],
        omega_desired_tooltip_local: np.ndarray,
    ) -> Optional[np.ndarray]:
        if (
            self.pin_model is None
            or self.pin_data is None
            or self.tooltip_frame_id_pin < 0
            or q_pin_config is None
        ):
            self.get_logger().warn(
                "_calculate_joint_velocities_pinocchio_jacobian: Preconditions not met.",
                throttle_duration_sec=5.0,
            )
            return None
        try:
            pin.computeJointJacobians(self.pin_model, self.pin_data, q_pin_config)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            J_full = pin.getFrameJacobian(
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
                    f"Articutool v_indices not setup correctly: {self.articutool_v_indices_pin}"
                )
                return None
            J_angular_tool = J_full[3:6, self.articutool_v_indices_pin]
            if J_angular_tool.shape != (3, 2):
                self.get_logger().error(f"Jacobian shape error: {J_angular_tool.shape}")
                return None
            try:
                J_pinv = np.linalg.pinv(J_angular_tool, rcond=self.JACOBIAN_PINV_RCOND)
            except np.linalg.LinAlgError:
                self.get_logger().warn("Jacobian pseudo-inverse failed.")
                return None
            dq = J_pinv @ omega_desired_tooltip_local
            return dq if dq.shape == (2,) else None
        except Exception as e:
            self.get_logger().error(
                f"Pinocchio Jacobian error: {e} {traceback.format_exc()}"
            )
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

    def _enforce_joint_limits_predictive(
        self, current_q: np.ndarray, desired_dq: np.ndarray, dt: float
    ) -> np.ndarray:
        final_dq = np.copy(desired_dq)
        if dt <= self.EPSILON:
            for i in range(len(self.articutool_joint_names)):
                q_i, dq_i, low, upp = (
                    current_q[i],
                    desired_dq[i],
                    self.joint_limits_lower[i],
                    self.joint_limits_upper[i],
                )
                if (dq_i < -self.EPSILON and q_i <= low + self.EPSILON) or (
                    dq_i > self.EPSILON and q_i >= upp - self.EPSILON
                ):
                    final_dq[i] = 0.0
            return final_dq
        for i in range(len(self.articutool_joint_names)):
            q_i, dq_i_des, low, upp = (
                current_q[i],
                desired_dq[i],
                self.joint_limits_lower[i],
                self.joint_limits_upper[i],
            )
            q_next_pred = q_i + dq_i_des * dt
            q_next_clamp = np.clip(q_next_pred, low, upp)
            final_dq[i] = (q_next_clamp - q_i) / dt if dt > self.EPSILON else 0.0
        return final_dq

    def _publish_command(self, joint_velocities: Optional[np.ndarray]):
        if joint_velocities is None:
            joint_velocities = np.zeros(len(self.articutool_joint_names))
        if len(joint_velocities) != len(self.articutool_joint_names):
            self.get_logger().error(
                f"Cmd length mismatch. Expected {len(self.articutool_joint_names)}, got {len(joint_velocities)}",
                throttle_duration_sec=5.0,
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
        node = ArticutoolController()
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        if node:
            node.get_logger().info(
                f"{node.get_name()} shutting down cleanly due to interrupt."
            )
    except ValueError as e:
        logger = (
            node.get_logger()
            if node
            else rclpy.logging.get_logger("articutool_controller_prerun_error")
        )
        logger.fatal(f"ArticutoolController ValueError: {e}\n{traceback.format_exc()}")
    except Exception as e:
        logger = (
            node.get_logger()
            if node
            else rclpy.logging.get_logger("articutool_controller_unhandled_error")
        )
        logger.fatal(
            f"Unhandled exception in ArticutoolController: {e}\n{traceback.format_exc()}"
        )
    finally:
        if node and node.executor is None:
            if (
                node.active_primitive_goal_handle is not None
                and node.active_primitive_goal_handle.is_active
            ):
                node.get_logger().info(
                    "Aborting active primitive action before shutdown."
                )
                node.active_primitive_goal_handle.abort()
            if node.current_orientation_control_mode != MODE_DISABLED:
                node.get_logger().info("Publishing zero command before shutdown.")
                node._publish_zero_command()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
