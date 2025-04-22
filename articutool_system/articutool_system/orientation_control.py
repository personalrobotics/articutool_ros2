# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from geometry_msgs.msg import Quaternion, QuaternionStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from articutool_interfaces.srv import SetOrientationControl

# Import math libraries
import numpy as np
from scipy.spatial.transform import Rotation as R

# Import Pinocchio and URDF parser
import pinocchio as pin

# Standard imports
import os
import tempfile
import subprocess
from typing import Optional
from ament_index_python.packages import get_package_share_directory


class OrientationControl(Node):
    """
    Controls Articutool's Roll/Pitch joints (J1/J2) to maintain a target
    orientation for the tool_tip frame, using orientation feedback for the
    atool_imu_frame and Pinocchio for kinematic calculations. Activated via service.
    """
    def __init__(self):
        super().__init__("orientation_control")

        # --- Parameters ---
        # --- Control gains ---
        p_gain_desc = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY, description='PID Proportional gains [Pitch, Roll]')
        i_gain_desc = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY, description='PID Integral gains [Pitch, Roll]')
        d_gain_desc = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY, description='PID Derivative gains [Pitch, Roll]')
        self.declare_parameter('pid_gains.p', [1.0, 1.0], p_gain_desc)
        self.declare_parameter('pid_gains.i', [0.1, 0.1], i_gain_desc)
        self.declare_parameter('pid_gains.d', [0.05, 0.05], d_gain_desc)
        self.declare_parameter('integral_clamp', 1.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, description='Max absolute value for integral term'))

        # --- Node Config ---
        loop_rate_desc = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, description='Control loop frequency (Hz)')
        feedback_topic_desc = ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='Topic for QuaternionStamped orientation feedback (of atool_imu_frame)')
        command_topic_desc = ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='Topic for publishing Float64MultiArray velocity commands')
        joint_state_topic_desc = ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='Topic for subscribing to Articutool joint states')
        self.declare_parameter('loop_rate', 50.0, loop_rate_desc)
        self.declare_parameter('feedback_topic', '/articutool/estimated_orientation', feedback_topic_desc)
        self.declare_parameter('command_topic', '/articutool/velocity_controller/commands', command_topic_desc)
        self.declare_parameter('joint_state_topic', '/articutool/joint_states', joint_state_topic_desc)

        # --- Model/Kinematics ---
        urdf_path_desc = ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='Path to the Articutool URDF/XACRO file')
        articutool_base_link_desc = ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='Base link name of the Articutool model (attaches to arm)')
        imu_link_desc = ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='Name of the link where orientation feedback is measured')
        tooltip_link_desc = ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='Name of the link whose orientation is controlled')
        joint_names_desc = ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY, description='Names of the actuated joints [Pitch, Roll]')
        self.declare_parameter('urdf_path', '', urdf_path_desc)
        self.declare_parameter('articutool_base_link', 'atool_handle', articutool_base_link_desc)
        self.declare_parameter('imu_link_frame', 'atool_imu_frame', imu_link_desc)
        self.declare_parameter('tooltip_frame', 'tool_tip', tooltip_link_desc)
        self.declare_parameter('joint_names', ['atool_joint1', 'atool_joint2'], joint_names_desc)

        # --- Get Parameters ---
        self.Kp = np.array(self.get_parameter('pid_gains.p').value)
        self.Ki = np.array(self.get_parameter('pid_gains.i').value)
        self.Kd = np.array(self.get_parameter('pid_gains.d').value)
        self.integral_max = self.get_parameter('integral_clamp').value
        self.rate = self.get_parameter('loop_rate').value
        self.feedback_topic = self.get_parameter('feedback_topic').value
        self.command_topic = self.get_parameter('command_topic').value
        self.joint_state_topic = self.get_parameter('joint_state_topic').value
        self.articutool_base_link = self.get_parameter('articutool_base_link').value
        self.imu_link = self.get_parameter('imu_link_frame').value
        self.tooltip_link = self.get_parameter('tooltip_frame').value
        self.joint_names = self.get_parameter('joint_names').value
        xacro_filename = self.get_parameter('urdf_path').value

        if len(self.joint_names) != 2:
            raise ValueError("Expecting exactly 2 joint names (Pitch, Roll)")
        if len(self.Kp) != 2 or len(self.Ki) != 2 or len(self.Kd) != 2:
             raise ValueError("PID gains must be provided as arrays of length 2")

        # --- Pinocchio Setup ---
        try:
            if not os.path.exists(xacro_filename):
                raise FileNotFoundError(f"Xacro file not found at {xacro_filename}")

            self.get_logger().info("Processing Xacro file")
            try:
                process = subprocess.run(
                    ["ros2", "run", "xacro", "xacro", xacro_filename],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                urdf_xml_string = process.stdout
                self.get_logger().info("XACRO processing successful.")
            except FileNotFoundError as e:
                self.get_logger().fatal(
                    f"Command 'ros2 run xacro ...' failed. Is xacro installed ('ros-{self.get_namespace().split('/')[-1]}-xacro') and ROS 2 sourced properly? Error: {e}"
                )
                raise RuntimeError("Failed to find/run xacro command") from e
            except subprocess.CalledProcessError as e:
                self.get_logger().fatal(
                    f"XACRO processing command failed with exit code {e.returncode}."
                )
                self.get_logger().error(f"XACRO stderr:\n{e.stderr}")
                raise RuntimeError("XACRO processing failed") from e

            # Create a temporary file and write the URDF string to it
            # We use delete=False because Pinocchio needs to open the file by path.
            # We MUST manually delete it in the finally block.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".urdf", delete=False
            ) as temp_urdf_file:
                temp_urdf_path = temp_urdf_file.name
                temp_urdf_file.write(urdf_xml_string)
                # File is flushed and closed automatically when 'with' block exits

            self.get_logger().info(f"Generated temporary URDF file: {temp_urdf_path}")

            # Load the model - assuming the URDF describes the articutool *only*,
            # potentially attached to a 'world' or its 'base_link' parameter.
            # If the URDF includes the Jaco arm, we need to build a reduced model.
            # For now, assume it's just the Articutool relative to its base_link.
            self.pin_model = pin.buildModelFromUrdf(temp_urdf_path)
            self.pin_data = self.pin_model.createData()
            self.get_logger().info(f"Pinocchio model loaded successfully from {temp_urdf_path}")

            # Get frame and joint IDs (handle potential errors)
            if not self.pin_model.existFrame(self.imu_link): raise ValueError(f"IMU frame '{self.imu_link}' not found in Pinocchio model")
            if not self.pin_model.existFrame(self.tooltip_link): raise ValueError(f"Tooltip frame '{self.tooltip_link}' not found in Pinocchio model")
            if not self.pin_model.existJoint(self.joint_names[0]): raise ValueError(f"Joint '{self.joint_names[0]}' not found")
            if not self.pin_model.existJoint(self.joint_names[1]): raise ValueError(f"Joint '{self.joint_names[1]}' not found")

            self.imu_frame_id = self.pin_model.getFrameId(self.imu_link)
            self.tooltip_frame_id = self.pin_model.getFrameId(self.tooltip_link)
            # Assuming the joints in Pinocchio model correspond directly to the names provided
            # Note: Pinocchio joint indices often start from 1 (0 is universe)
            self.joint1_id = self.pin_model.getJointId(self.joint_names[0])
            self.joint2_id = self.pin_model.getJointId(self.joint_names[1])
            # We need the velocity indices (nv) which correspond to the columns in the Jacobian
            # Assuming standard revolute joints, velocity index = joint index - 1 (due to universe)
            self.joint1_vel_idx = self.pin_model.joints[self.joint1_id].idx_v
            self.joint2_vel_idx = self.pin_model.joints[self.joint2_id].idx_v

        except Exception as e:
            self.get_logger().fatal(f"Failed to initialize Pinocchio model: {e}", exc_info=True)
            raise e # Prevent node from starting cleanly

        # --- State variables ---
        self.control_active = False
        self.target_orientation_world = R.identity()
        self.current_imu_orientation_world: Optional[R] = None
        self.current_joint_positions: Optional[np.ndarray] = None
        self.last_error = np.zeros(3)
        self.integral_error = np.zeros(3)
        self.last_time = self.get_clock().now()
        self.reference_frame = ""

        # --- ROS Comms ---
        self.srv = self.create_service(SetOrientationControl, '/articutool/set_orientation_control', self.set_orientation_control_callback)
        self.feedback_sub = self.create_subscription(QuaternionStamped, self.feedback_topic, self.feedback_callback, 1) # QoS=1 for latest
        self.joint_state_sub = self.create_subscription(JointState, self.joint_state_topic, self.joint_state_callback, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, self.command_topic, 10)
        self.timer = self.create_timer(1.0 / self.rate, self.control_loop)

        self.get_logger().info("Articutool Orientation Controller Node Started.")

    def joint_state_callback(self, msg):
        self.joint_states = msg
        # Update joint positions dictionary for easier access
        for i, name in enumerate(msg.name):
            if name in self.controlled_joint_names:
                # Make sure position array is long enough
                if i < len(msg.position):
                    self.joint_positions[name] = msg.position[i]
                else:
                    self.get_logger().warn(
                        f"Joint '{name}' exists in names but not in position array."
                    )

    def imu_orientation_callback(self, msg):
        # Assumes msg.orientation is the orientation of imu_frame relative to a world fixed frame
        # And that this world frame is consistent with the one `atool_handle` is defined in,
        # or that there's a TF link between them. Here, we assume the IMU directly gives
        # the orientation relative to the effective 'world' for control.
        try:
            q = msg.orientation
            # Store as scipy Rotation object relative to world
            self.current_rot_world_imu = R.from_quat([q.x, q.y, q.z, q.w])
        except Exception as e:
            self.get_logger().error(
                f"Exception in imu_orientation_callback: {e}", throttle_duration_sec=5
            )

    def desired_orientation_callback(self, msg):
        # Assumes msg is desired orientation of tip_frame relative to world_frame
        try:
            self.desired_rot_world_tip = R.from_quat([msg.x, msg.y, msg.z, msg.w])
        except Exception as e:
            self.get_logger().error(
                f"Exception in desired_orientation_callback: {e}",
                throttle_duration_sec=5,
            )

    def control_loop(self):
        """Main PID control calculation and publishing using Jacobian."""
        if not self.pinocchio_ready or self.model is None or self.data is None:
            self.get_logger().error(
                "Pinocchio not initialized, cannot run control loop.",
                throttle_duration_sec=10,
            )
            return
        if not self.joint_positions:
            self.get_logger().info(
                "Waiting for initial joint states...", throttle_duration_sec=5
            )
            return

        try:
            # 1. Get Current State (Orientations, Joint Positions)
            # -----------------------------------------------------
            current_rot_world_imu = (
                self.current_rot_world_imu
            )  # From imu_orientation_callback
            desired_rot_world_tip = (
                self.desired_rot_world_tip
            )  # From desired_orientation_callback

            # --- Get current joint positions in Pinocchio's expected order ---
            q = np.zeros(self.model.nq)
            all_joints_found = True
            for i, name in enumerate(self.pinocchio_joint_names_ordered):
                if name in self.joint_positions:
                    q[i] = self.joint_positions[name]
                else:
                    # Handle missing joint state - crucial for safety
                    self.get_logger().error(
                        f"Missing joint state for '{name}' required by Pinocchio. Cannot proceed.",
                        throttle_duration_sec=5,
                    )
                    all_joints_found = False
                    break

            if not all_joints_found:
                return

            # 2. Update Pinocchio Kinematics
            # --------------------------------
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # 3. Calculate Desired & Current Orientation relative to IMU frame
            # ------------------------------------------------------------------
            desired_rot_imu_tip = current_rot_world_imu.inv() * desired_rot_world_tip

            # Get current T(imu -> tip) transform from Pinocchio
            T_world_imu = self.data.oMf[
                self.imu_frame_id
            ]  # Transform from IMU frame to World frame
            T_world_tip = self.data.oMf[
                self.tip_frame_id
            ]  # Transform from Tip frame to World frame
            # T_imu_tip = T_imu_world * T_world_tip = T_world_imu^-1 * T_world_tip
            T_imu_tip = T_world_imu.inverse() * T_world_tip
            current_rot_imu_tip = R.from_matrix(T_imu_tip.rotation)

            # 4. Calculate Orientation Error & Run PID
            # -----------------------------------------
            error_rot_imu = desired_rot_imu_tip * current_rot_imu_tip.inv()
            error_vec_imu = error_rot_imu.as_rotvec()

            # Handle angle wrap-around
            angle = np.linalg.norm(error_vec_imu)
            if angle > math.pi:
                if angle > 1e-9:
                    error_vec_imu = error_vec_imu / angle * (angle - 2.0 * math.pi)
                else:
                    error_vec_imu = np.zeros(3)  # Avoid NaN if angle is exactly zero

            # PID Calculation
            self.integral_error += error_vec_imu * self.dt
            # Basic anti-windup - consider more advanced methods if needed
            integral_max = 1.0
            self.integral_error = np.clip(
                self.integral_error, -integral_max, integral_max
            )
            i_term = self.ki * self.integral_error

            derivative_error = (error_vec_imu - self.previous_error_vec) / self.dt
            d_term = self.kd * derivative_error

            self.previous_error_vec = error_vec_imu  # Update previous error

            # PID output: Desired angular velocity correction in IMU frame [wx, wy, wz]
            pid_output_imu = self.kp * error_vec_imu + i_term + d_term

            # Select desired correction for Roll (IMU X) and Pitch (IMU Y)
            omega_imu_desired_rp = np.array([pid_output_imu[0], pid_output_imu[1]])

            # 5. Calculate Joint Velocities using Jacobian
            # ---------------------------------------------
            # --- Calculate Jacobian ---
            # Compute Jacobian of the tip frame expressed in the world frame (6xNV)
            # NV = Number of velocity dimensions (usually == NQ for simple joints)
            J_world = pin.computeFrameJacobian(
                self.model, self.data, q, self.tip_frame_id
            )

            # Get rotation matrix from World frame TO IMU frame (R_imu_world)
            R_imu_world = T_world_imu.rotation.T

            # Transform angular part of Jacobian (rows 3,4,5) to be expressed in IMU frame
            J_omega_imu_full = R_imu_world @ J_world[3:6, :]

            # Extract columns corresponding to the controlled joints' velocity variables
            J_omega_imu_controlled = J_omega_imu_full[:, self.controlled_vel_indices]

            # Extract rows corresponding to desired control axes (Roll=IMU X=row 0, Pitch=IMU Y=row 1)
            J_omega_rp = J_omega_imu_controlled[[0, 1], :]

            # --- Inverse Differential Kinematics (Damped Pseudo-Inverse) ---
            try:
                Jt = J_omega_rp.T
                JJt = J_omega_rp @ Jt
                # Add damping term (lambda * I)
                lambda_eye = self.jacobian_damping * np.eye(
                    J_omega_rp.shape[0]
                )  # Size is 2x2

                # Solve (J*J^T + lambda*I) * x = omega_desired_rp
                # Then q_dot = J^T * x
                inv_term = np.linalg.solve(JJt + lambda_eye, omega_imu_desired_rp)
                q_dot = Jt @ inv_term

                # Check for NaN/Inf in result (can happen with extreme inputs/singularities)
                if np.isnan(q_dot).any() or np.isinf(q_dot).any():
                    self.get_logger().warn(
                        "NaN/Inf detected in Jacobian velocity calculation. Setting velocities to zero.",
                        throttle_duration_sec=1,
                    )
                    joint_velocities_raw = np.zeros(len(self.controlled_joint_names))
                else:
                    joint_velocities_raw = q_dot  # Result is [j1_vel, j2_vel]

                # Optional: Log condition number for singularity check
                # cond_num = np.linalg.cond(J_omega_rp)
                # self.get_logger().debug(f"Jacobian Condition Number: {cond_num:.2f}", throttle_duration_sec=1)

            except np.linalg.LinAlgError:
                self.get_logger().warn(
                    "Jacobian inverse calculation failed (LinAlgError). Setting velocities to zero.",
                    throttle_duration_sec=1,
                )
                joint_velocities_raw = np.zeros(len(self.controlled_joint_names))

            # 6. Apply Joint Limit Avoidance
            # --------------------------------
            joint_velocities_limited = self.apply_joint_limit_avoidance(
                joint_velocities_raw
            )

            # 7. Publish Command
            # -------------------
            self.publish_velocities(joint_velocities_limited)

            # 8. Optional Debug Logging
            # --------------------------
            # log_orientation_euler(self.get_logger(), current_rot_world_imu, self.imu_frame, "world", prefix="[CTRL] ")
            # log_orientation_euler(self.get_logger(), current_rot_imu_tip, self.tip_frame, self.imu_frame, prefix="[CTRL] ")
            # log_orientation_euler(self.get_logger(), desired_rot_imu_tip, f"Desired {self.tip_frame}", self.imu_frame, prefix="[CTRL] ")
            # log_orientation_euler(self.get_logger(), error_rot_imu, "ErrorRotation", self.imu_frame, prefix="[CTRL] ")
            # self.get_logger().debug(f"[CTRL] PID Output IMU (Wx,Wy,Wz): [{pid_output_imu[0]:.3f}, {pid_output_imu[1]:.3f}, {pid_output_imu[2]:.3f}]", throttle_duration_sec=0.1)
            # self.get_logger().debug(f"[CTRL] Omega Desired RP: [{omega_imu_desired_rp[0]:.3f}, {omega_imu_desired_rp[1]:.3f}]", throttle_duration_sec=0.1)
            # self.get_logger().debug(f"[CTRL] Jacobian J_omega_rp:\n{J_omega_rp}", throttle_duration_sec=0.1)
            # self.get_logger().debug(f"[CTRL] Raw Joint Vel Cmd (j1,j2): [{joint_velocities_raw[0]:.3f}, {joint_velocities_raw[1]:.3f}]", throttle_duration_sec=0.1)
            # self.get_logger().debug(f"[CTRL] Lim Joint Vel Cmd(j1,j2): [{joint_velocities_limited[0]:.3f}, {joint_velocities_limited[1]:.3f}]", throttle_duration_sec=0.1)

        except Exception as e:
            self.get_logger().error(f"Exception in control_loop: {e}")
            import traceback

            self.get_logger().error(traceback.format_exc())
            # Ensure motors stop if control loop fails critically
            self.publish_velocities([0.0, 0.0])

    def apply_joint_limit_avoidance(self, joint_velocities):
        """Reduces velocity smoothly near joint limits."""
        reduced_velocities = np.copy(joint_velocities)
        if len(self.controlled_joint_names) != len(joint_velocities):
            self.get_logger().warn(
                "Mismatch between controlled joints and velocity command length."
            )
            return joint_velocities  # Return unmodified

        # Ensure margin and gain are positive to avoid math errors
        margin = abs(math.radians(10.0))  # Use abs just in case, should be positive
        reduction_gain = (
            abs(self.kd) if abs(self.kd) > 1e-6 else 5.0
        )  # Example: link to kd or use fixed positive value
        if margin <= 1e-9 or reduction_gain <= 1e-9:
            self.get_logger().warn(
                "Invalid margin or reduction_gain for joint limit avoidance."
            )
            return joint_velocities  # Skip avoidance if parameters are invalid

        for i, joint_name in enumerate(self.controlled_joint_names):
            if (
                joint_name in self.joint_positions
                and joint_name in self.joint_limits_lower
                and joint_name in self.joint_limits_upper
            ):
                pos = self.joint_positions[joint_name]
                lower = self.joint_limits_lower[joint_name]
                upper = self.joint_limits_upper[joint_name]
                vel = joint_velocities[i]

                dist_to_lower = pos - lower
                dist_to_upper = upper - pos

                scale = 1.0

                # Approaching lower limit and commanding negative velocity
                if vel < 0 and dist_to_lower < margin:
                    # Clamp distance to be non-negative for calculation
                    # Use max(0.0, ...) to prevent negative base for exponent
                    clipped_dist_lower = max(0.0, dist_to_lower)
                    # Avoid division by zero if margin is tiny
                    base = clipped_dist_lower / margin if margin > 1e-9 else 0.0
                    # Calculate scaling factor safely
                    current_scale = base ** (1.0 / reduction_gain)
                    scale = min(scale, current_scale)

                # Approaching upper limit and commanding positive velocity
                elif vel > 0 and dist_to_upper < margin:
                    # Clamp distance to be non-negative for calculation
                    # Use max(0.0, ...) to prevent negative base for exponent
                    clipped_dist_upper = max(0.0, dist_to_upper)
                    # Avoid division by zero if margin is tiny
                    base = clipped_dist_upper / margin if margin > 1e-9 else 0.0
                    # Calculate scaling factor safely
                    current_scale = base ** (1.0 / reduction_gain)
                    scale = min(scale, current_scale)

                # Apply the calculated scale factor
                reduced_velocities[i] = vel * scale

            else:
                # Only warn periodically if limits/positions are missing
                self.get_logger().warn(
                    f"Joint position or limits missing for {joint_name}",
                    throttle_duration_sec=10,
                )

        return reduced_velocities

    def publish_velocities(self, velocities):
        """Publishes the calculated joint velocities."""
        if len(velocities) != len(self.controlled_joint_names):
            self.get_logger().error("Cannot publish velocities: length mismatch.")
            return

        velocity_command = Float64MultiArray()
        # Important: Ensure the data order matches the expected order by the velocity controller
        # This might be based on the order in self.controlled_joint_names or a fixed order.
        # Assuming the controller expects velocities in the order of self.controlled_joint_names:
        velocity_command.data = list(velocities)
        self.velocity_publisher.publish(velocity_command)


def log_orientation_euler(
    logger,
    rotation_input,  # Can be SciPy Rot, ROS Quat, or np array [x,y,z,w]
    target_frame: str,  # Frame whose orientation is being described
    reference_frame: str,  # Frame relative to which orientation is described
    prefix: str = "",
):  # Optional prefix for grouping logs
    """
    Logs a quaternion orientation as neatly formatted Euler angles (XYZ, degrees).

    Ensures alignment for easy reading even with rapid logging.

    Args:
        logger: The ROS2 node logger instance (e.g., self.get_logger()).
        rotation_input: The rotation to log. Accepts scipy.Rotation,
                         geometry_msgs.Quaternion, or numpy array [x,y,z,w].
        target_frame: Name of the frame whose orientation is being described.
        reference_frame: Name of the frame relative to which the orientation is given.
        level: The logging severity level (e.g., INFO, DEBUG).
        prefix: An optional string to prepend to the log message for context.
    """
    try:
        # --- Input Conversion to SciPy Rotation ---
        rotation = None
        if isinstance(rotation_input, R):
            rotation = rotation_input
        elif isinstance(rotation_input, RosQuaternion):
            # Ensure quaternion is normalized before conversion for robustness
            norm = math.sqrt(
                rotation_input.x**2
                + rotation_input.y**2
                + rotation_input.z**2
                + rotation_input.w**2
            )
            if abs(norm - 1.0) > 1e-6:  # Check if normalization is needed
                logger.debug(
                    f"Normalizing quaternion for {target_frame} in {reference_frame}",
                    throttle_duration_sec=10,
                )
                if norm > 1e-9:
                    x = rotation_input.x / norm
                    y = rotation_input.y / norm
                    z = rotation_input.z / norm
                    w = rotation_input.w / norm
                    rotation = R.from_quat([x, y, z, w])
                else:  # Avoid division by zero for zero quaternion
                    rotation = R.identity()
            else:
                rotation = R.from_quat(
                    [
                        rotation_input.x,
                        rotation_input.y,
                        rotation_input.z,
                        rotation_input.w,
                    ]
                )

        elif (
            isinstance(rotation_input, (np.ndarray, list, tuple))
            and len(rotation_input) == 4
        ):
            # Assuming [x, y, z, w] order based on common ROS/SciPy usage
            q = np.array(rotation_input)
            norm = np.linalg.norm(q)
            if abs(norm - 1.0) > 1e-6:  # Check if normalization is needed
                logger.debug(
                    f"Normalizing quaternion for {target_frame} in {reference_frame}",
                    throttle_duration_sec=10,
                )
                if norm > 1e-9:
                    q = q / norm
                else:  # Avoid division by zero
                    q = np.array([0.0, 0.0, 0.0, 1.0])  # Identity if zero quaternion
            rotation = R.from_quat(q)
        else:
            logger.warn(
                f"{prefix}ORIENT | Invalid input type for logging: {type(rotation_input)} for {target_frame}",
                throttle_duration_sec=5,
            )
            return

        # --- Convert to Euler Angles (intrinsic 'xyz', degrees) ---
        # Common alternatives: 'zyx' (often used for aircraft/NED)
        # Choose one convention and stick to it for consistency.
        euler_deg = rotation.as_euler("xyz", degrees=True)
        roll, pitch, yaw = euler_deg[0], euler_deg[1], euler_deg[2]

        # --- Format the Output String ---
        # :<15s reserves 15 characters for the string, left-aligned. Adjust width as needed.
        # :7.1f reserves 7 characters for the float, 1 decimal place, fixed-point.
        # This fixed width ensures alignment. Handles numbers like -123.4, 12.3, -0.1 etc.
        frame_width = 18  # Adjust as needed for your longest frame names
        log_message = (
            f"{prefix}ORIENT | {target_frame:<{frame_width}} in {reference_frame:<{frame_width}} | "
            f"R: {roll:7.1f} P: {pitch:7.1f} Y: {yaw:7.1f} deg"
        )

        # --- Log the Message ---
        logger.info(log_message)

    except Exception as e:
        # Log exceptions specifically from this function for easier debugging
        logger.error(
            f"{prefix}ORIENT | Failed during orientation logging for {target_frame}: {e}",
            throttle_duration_sec=5,
        )
        import traceback

        logger.debug(traceback.format_exc())  # Log full traceback at DEBUG level


def main(args=None):
    rclpy.init(args=args)
    try:
        control_node = OrientationControl()
        rclpy.spin(control_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error during node execution: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "control_node" in locals() and control_node:
            control_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
