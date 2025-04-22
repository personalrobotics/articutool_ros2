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
            # if not self.pin_model.existJoint(self.joint_names[0]): raise ValueError(f"Joint '{self.joint_names[0]}' not found")
            # if not self.pin_model.existJoint(self.joint_names[1]): raise ValueError(f"Joint '{self.joint_names[1]}' not found")

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
            self.get_logger().fatal(f"Failed to initialize Pinocchio model: {e}")
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
        # self.cmd_pub = self.create_publisher(Float64MultiArray, self.command_topic, 10)
        # self.timer = self.create_timer(1.0 / self.rate, self.control_loop)

        self.get_logger().info("Articutool Orientation Controller Node Started.")

    def feedback_callback(self, msg: QuaternionStamped):
        """Stores the latest orientation feedback."""
        if not self.reference_frame:
            self.reference_frame = msg.header.frame_id
            self.get_logger().info(f"Received first orientation feedback relative to frame: '{self.reference_frame}'")
        elif self.reference_frame != msg.header.frame_id:
            self.get_logger().warn(f"Orientation feedback frame changed from '{self.reference_frame}' to '{msg.header.frame_id}'!")
            self.reference_frame = msg.header.frame_id

        self.current_imu_orientation_world = R.from_quat([
            msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w
        ])

    def joint_state_callback(self, msg: JointState):
        """Stores the latest positions for the controlled joints."""
        if self.current_joint_positions is None:
             self.current_joint_positions = np.zeros(len(self.joint_names))

        found_count = 0
        for i, name in enumerate(msg.name):
            try:
                # Find index in our ordered list
                idx = self.joint_names.index(name)
                self.current_joint_positions[idx] = msg.position[i]
                found_count += 1
            except ValueError:
                continue

    def set_orientation_control_callback(self, request: SetOrientationControl.Request, response: SetOrientationControl.Response):
        """Handles service requests to enable/disable control and set target."""
        self.get_logger().info(f"SetOrientationControl Request: enable={request.enable}")
        if request.enable:
            try:
                # Store target as scipy Rotation object
                self.target_orientation_world = R.from_quat([
                    request.target_orientation.x, request.target_orientation.y,
                    request.target_orientation.z, request.target_orientation.w
                ])
                # Reset PID state
                self.integral_error.fill(0.0)
                self.last_error.fill(0.0)
                self.last_time = self.get_clock().now()
                self.control_active = True
                self.get_logger().info(f"Orientation control ENABLED.")
                response.success = True
                response.message = "Control enabled."
            except Exception as e:
                self.get_logger().error(f"Error setting target orientation: {e}")
                response.success = False
                response.message = f"Error setting target: {e}"
        else:
            self.control_active = False
            self._publish_zero_command()
            self.get_logger().info("Orientation control DISABLED.")
            response.success = True
            response.message = "Control disabled."
        return response

    def _publish_command(self, joint_velocities: np.ndarray):
        """Publishes velocity commands, assuming Float64MultiArray."""
        if len(joint_velocities) != len(self.joint_names):
            self.get_logger().error(f"Command length mismatch ({len(joint_velocities)}) vs expected ({len(self.joint_names)})")
            return

        cmd_msg = Float64MultiArray()
        cmd_msg.data = joint_velocities.tolist()
        self.cmd_pub.publish(cmd_msg)

    def _publish_zero_command(self):
        """Helper to send zero velocity commands."""
        self._publish_command(np.zeros(len(self.joint_names)))

# --- Main Execution ---
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = OrientationControl()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as e:
        # Catch initialization errors or other major issues
        if node: node.get_logger().fatal(f"Unhandled exception: {e}")
        else: print(f"Unhandled exception before node init: {e}")
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
             rclpy.shutdown()

if __name__ == '__main__':
    main()
