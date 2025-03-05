import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import numpy as np
import pinocchio
from pinocchio.robot_wrapper import RobotWrapper
from articutool_orientation.pinocchio_ik import PinocchioIK
import os

class IKSolver(Node):
    def __init__(self):
        super().__init__('ik_solver')
        self.declare_parameter('urdf_path', 'articutool.urdf')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('end_effector_link', 'tool_tip')
        self.urdf_path = os.path.join(os.getcwd(), 'src/articutool_description/urdf/', self.get_parameter('urdf_path').value)
        self.base_link = self.get_parameter('base_link').value
        self.end_effector_link = self.get_parameter('end_effector_link').value

        self.ik = PinocchioIK(self.urdf_path, self.base_link, self.end_effector_link)
        self.joint_names = self.ik.get_joint_names()
        lower_limits, upper_limits = self.ik.get_joint_limits()
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits

        self.pose_subscription = self.create_subscription(Pose, 'target_pose', self.pose_callback, 10)
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.current_joint_state = JointState()

    def pose_callback(self, msg):
        target_pose = pinocchio.SE3(
            np.array([[msg.orientation.x, msg.orientation.y, msg.orientation.z],
                      [msg.orientation.y, msg.orientation.w, msg.orientation.x],
                      [msg.orientation.z, msg.orientation.x, msg.orientation.w]]),
            np.array([msg.position.x, msg.position.y, msg.position.z])
        )

        q_init = np.zeros(len(self.joint_names)) #initial joint state
        q, success = self.ik.compute_ik(target_pose, q_init)

        if success:
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            joint_state_msg.name = self.joint_names
            joint_state_msg.position = q.tolist()
            self.joint_state_publisher.publish(joint_state_msg)
        else:
            self.get_logger().warn("IK solver did not converge.")

def main(args=None):
    rclpy.init(args=args)
    ik_solver = IKSolver()
    rclpy.spin(ik_solver)
    ik_solver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
