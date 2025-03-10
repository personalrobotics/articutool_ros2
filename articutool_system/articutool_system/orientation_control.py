import rclpy
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Quaternion, TransformStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from tf_transformations import quaternion_matrix, euler_from_matrix, quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Float64MultiArray  # Import Float64MultiArray
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import math

class OrientationControl(Node):

    def __init__(self):
        super().__init__('orientation_control')

        # Static rotation from atool_handle to tool_tip (90 degrees around X)
        self.static_rotation = R.from_euler('x', 90, degrees=True)
        self.static_rotation_inv = self.static_rotation.inv()

        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.orientation_subscription = self.create_subscription(
            Quaternion,
            '/estimated_orientation',
            self.orientation_callback,
            10
        )
        self.velocity_publisher = self.create_publisher(
            Float64MultiArray,
            '/velocity_controller/commands',
            10
        )
        self.desired_orientation_subscription = self.create_subscription(
            Quaternion, '/desired_orientation', self.desired_orientation_callback, 10
        )

        self.desired_orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.current_orientation = Quaternion()
        self.joint_states = JointState()

        # PID gains (to be tuned)
        self.kp = 5.0
        self.ki = 0.01
        self.kd = 0.1

        self.integral_error = np.array([0.0, 0.0, 0.0])
        self.previous_error = np.array([0.0, 0.0, 0.0])

        self.dt = 0.01 # 100 Hz
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.joint_limits_lower = np.array([-math.pi / 2, -math.pi / 2])
        self.joint_limits_upper = np.array([math.pi / 2, math.pi / 2])

    def joint_state_callback(self, msg):
        self.joint_states = msg

    def orientation_callback(self, msg):
        try:
            self.current_orientation = msg
        except Exception as e:
            self.get_logger().error(f"Exception in orientation_callback: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def desired_orientation_callback(self, msg):
        self.desired_orientation = msg

    def control_loop(self):
        self.calculate_control()

    def calculate_control(self):
        try:
            # Calculate fork tip orientation
            tool_tip_orientation = self.calculate_tool_tip_orientation()

            # Calculate orientation error
            error_quaternion = self.quaternion_multiply(self.desired_orientation, self.quaternion_conjugate(tool_tip_orientation))
            error_vector = self.quaternion_to_rotation_vector(error_quaternion)

            # PID control
            self.integral_error += error_vector * self.dt
            derivative_error = (error_vector - self.previous_error) / self.dt
            joint_velocities = self.kp * error_vector + self.ki * self.integral_error + self.kd * derivative_error

            self.previous_error = error_vector

            joint_velocities = self.apply_joint_limit_reduction(joint_velocities)

            # Publish joint velocities (only roll and pitch)
            velocity_command = Float64MultiArray()
            velocity_command.data = [joint_velocities[0], joint_velocities[2]]
            self.velocity_publisher.publish(velocity_command)
        except Exception as e:
            self.get_logger().error(f"Exception in calculate_control: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def calculate_tool_tip_orientation(self):
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                "root", "tool_tip", rclpy.time.Time()
            )
            tool_tip_raw = transform.transform.rotation

            # Convert geometry_msgs/Quaternion to NumPy array
            tool_tip_quat = np.array([tool_tip_raw.x, tool_tip_raw.y, tool_tip_raw.z, tool_tip_raw.w])

            # Apply the inverse static rotation
            tool_tip_rot = R.from_quat(tool_tip_quat)
            tool_tip_corrected_rot = self.static_rotation_inv * tool_tip_rot
            tool_tip_corrected_quat = tool_tip_corrected_rot.as_quat()

            return Quaternion(x=tool_tip_corrected_quat[0], y=tool_tip_corrected_quat[1], z=tool_tip_corrected_quat[2], w=tool_tip_corrected_quat[3])

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF error: {e}")
            return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0) # return identity quaternion on error

    def quaternion_multiply(self, q1, q2):
        # Quaternion multiplication
        w1, x1, y1, z1 = q1.w, q1.x, q1.y, q1.z
        w2, x2, y2, z2 = q2.w, q2.x, q2.y, q2.z

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return Quaternion(w=w, x=x, y=y, z=z)

    def quaternion_conjugate(self, q):
        # Quaternion conjugate
        return Quaternion(x=-q.x, y=-q.y, z=-q.z, w=q.w)

    def quaternion_to_rotation_vector(self, q):
        # Quaternion to rotation vector
        w, x, y, z = q.w, q.x, q.y, q.z  # Extract w, x, y, z
        angle = 2.0 * np.arccos(w)
        s = np.sqrt(1.0 - w * w)
        if s < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])  # Arbitrary axis if angle is close to 0
        else:
            axis = np.array([x, y, z]) / s
        return axis * angle

    def apply_joint_limit_reduction(self, joint_velocities):
        if not self.joint_states.position or not self.joint_states.velocity:
            return joint_velocities

        reduced_velocities = np.copy(joint_velocities)
        joint_positions = np.array(self.joint_states.position)

        for i in [0, 1]:
            joint_range = self.joint_limits_upper[i] - self.joint_limits_lower[i]
            joint_center = (self.joint_limits_upper[i] + self.joint_limits_lower[i]) / 2.0
            joint_width = joint_range / 2.0

            distance_from_center = abs(joint_positions[i] - joint_center)
            scaling_factor = math.exp(-0.5 * (distance_from_center / joint_width * 2.5) ** 2)

            reduced_velocities[i] *= scaling_factor

        return reduced_velocities

def main(args=None):
    rclpy.init(args=args)
    control_node = OrientationControl()
    rclpy.spin(control_node)
    control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
