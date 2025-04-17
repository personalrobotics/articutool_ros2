# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from articutool_control.controller_switcher import ControllerSwitcher


class SendTrajectoryActionClient(Node):
    def __init__(self):
        super().__init__("send_trajectory_action_client")
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/articutool/joint_trajectory_controller/follow_joint_trajectory",
        )
        self.controller_switcher = ControllerSwitcher()

    def send_trajectory_goal(self, joint_names, points):
        self.controller_switcher.switch_controllers(
            ["joint_trajectory_controller"], ["velocity_controller"]
        )

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names

        for point in points:
            trajectory_point = JointTrajectoryPoint()
            trajectory_point.positions = point["positions"]
            if "velocities" in point:
                trajectory_point.velocities = point["velocities"]
            if "accelerations" in point:
                trajectory_point.accelerations = point["accelerations"]
            trajectory_point.time_from_start = Duration(sec=point["time_from_start"])
            goal_msg.trajectory.points.append(trajectory_point)

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected :(")
            return

        self.get_logger().info("Goal accepted :)")

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result.error_code} - {result.error_string}")
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Received feedback: {feedback.actual.positions}")


def main(args=None):
    rclpy.init(args=args)
    action_client = SendTrajectoryActionClient()

    joint_names = ["atool_joint1", "atool_joint2"]

    # Define your trajectory points
    trajectory_points = [
        {"positions": [1.0, 1.0], "time_from_start": 2},
    ]

    action_client.send_trajectory_goal(joint_names, trajectory_points)

    rclpy.spin(action_client)


if __name__ == "__main__":
    main(args=args)
