#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import termios
import tty
import select
import sys

from control_msgs.msg import JointJog, JointTrajectoryControllerState
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from controller_manager_msgs.srv import SwitchController, ListControllers

INSTRUCTION_MSG = """
Control the Articutool!
---------------------------
Joint control (velocity):
  1-2: joint 1-2
  r: reverse the direction of joint movement
  v: toggle velocity control

Joint control (position):
  1-2: joint 1-2
  p: toggle position control

Orientation control:
  q,w,e,a,s,d: control orientation quaternion
  o: toggle orientation control

CTRL-C to quit
"""

JOINT_NAMES = [
    "atool_joint1",
    "atool_joint2",
]
JOINT_VEL_CMD = 0.5  # rad/s
JOINT_POS_CMD = 0.1  # rad
ORIENTATION_CMD = 0.1  # Quaternion change
COMMAND_KEYS = ["1", "2"]

NAMESPACE = "articutool"
CONTROLLER_MANAGER = f"/{NAMESPACE}/controller_manager"
VELOCITY_CONTROLLER_NAME = "velocity_controller"
JOINT_TRAJECTORY_CONTROLLER_NAME = "joint_trajectory_controller"
VELOCITY_CONTROLLER_TOPIC = f"/{NAMESPACE}/{VELOCITY_CONTROLLER_NAME}"
JOINT_TRAJECTORY_CONTROLLER_TOPIC = f"/{NAMESPACE}/{JOINT_TRAJECTORY_CONTROLLER_NAME}"
DESIRED_ORIENTATION_TOPIC = f"{NAMESPACE}/desired_orientation"


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def validate_key(key):
    return key in COMMAND_KEYS


class ControllerSwitcher(Node):
    def __init__(self):
        super().__init__("controller_switcher")
        self.switch_controller_client = self.create_client(
            SwitchController, f"{CONTROLLER_MANAGER}/switch_controller"
        )
        self.list_controllers_client = self.create_client(
            ListControllers, f"{CONTROLLER_MANAGER}/list_controllers"
        )

        while not self.switch_controller_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Switch controller service not available, waiting again..."
            )

        while not self.list_controllers_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "List controller service not available, waiting again..."
            )

    def switch_controllers(
        self, activate_controllers, deactivate_controllers, strictness=1
    ):
        request = SwitchController.Request()
        request.activate_controllers = activate_controllers
        request.deactivate_controllers = deactivate_controllers
        request.strictness = strictness
        future = self.switch_controller_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            if not future.result().ok:
                self.get_logger().error("Controller switch failed.")
        else:
            self.get_logger().error("Service call failed.")

    def list_controllers(self):
        request = ListControllers.Request()
        future = self.list_controllers_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            for controller in future.result().controller:
                self.get_logger().info(
                    f"Controller: {controller.name}, State: {controller.state}"
                )
        else:
            self.get_logger().error("Service call failed.")


def main(args=None):
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init(args=args)
    node = Node("articutool_keyboard_teleop")
    controller_switcher = ControllerSwitcher()
    controller_switcher.switch_controllers(
        [VELOCITY_CONTROLLER_NAME], [JOINT_TRAJECTORY_CONTROLLER_NAME]
    )

    velocity_pub = node.create_publisher(
        Float64MultiArray, f"{VELOCITY_CONTROLLER_TOPIC}/commands", 1
    )
    position_pub = node.create_publisher(
        JointTrajectory, f"{JOINT_TRAJECTORY_CONTROLLER_TOPIC}/joint_trajectory", 1
    )
    orientation_pub = node.create_publisher(Quaternion, DESIRED_ORIENTATION_TOPIC, 1)

    joint_direction = 1.0
    control_mode = "velocity"  # velocity, position, orientation

    desired_orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)  # Initial orientation

    try:
        node.get_logger().info(INSTRUCTION_MSG)
        controller_switcher.list_controllers()

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0)
            key = get_key(settings)

            if control_mode == "velocity":
                velocity_command = Float64MultiArray()
                velocity_command.data = [0.0] * len(JOINT_NAMES)
                if validate_key(key):
                    joint_index = int(key) - 1
                    velocity_command.data[joint_index] = JOINT_VEL_CMD * joint_direction
                velocity_pub.publish(velocity_command)

            elif control_mode == "joint_trajectory":
                try:
                    positions_str = input(
                        "Enter joint positions (radians, space-separated): "
                    )
                    positions = [float(p) for p in positions_str.split()]
                    duration = float(input("Enter trajectory duration (seconds): "))

                    if len(positions) != len(JOINT_NAMES):
                        raise ValueError(
                            f"Please enter {len(JOINT_NAMES)} joint positions."
                        )

                    position_command = JointTrajectory()
                    position_command.joint_names = JOINT_NAMES
                    point = JointTrajectoryPoint()
                    point.positions = positions
                    point.time_from_start = rclpy.duration.Duration(
                        seconds=duration
                    ).to_msg()

                    position_command.points.append(point)
                    position_pub.publish(position_command)

                    control_mode = "idle"
                    node.get_logger().info("Returning to idle control mode.")

                except ValueError as e:
                    node.get_logger().error(f"Invalid input: {e}")
                    control_mode = "velocity"
                    node.get_logger().info("Returning to velocity control mode.")

            if key == "r":
                joint_direction *= -1.0
                node.get_logger().info(f"Joint direction: {joint_direction}")
            elif key == "v":
                control_mode = "velocity"
                node.get_logger().info("Velocity control mode")
                controller_switcher.switch_controllers(
                    [VELOCITY_CONTROLLER_NAME], [JOINT_TRAJECTORY_CONTROLLER_NAME]
                )
            elif key == "j":
                control_mode = "joint_trajectory"
                node.get_logger().info("Joint trajectory control mode")
                controller_switcher.switch_controllers(
                    [JOINT_TRAJECTORY_CONTROLLER_NAME], [VELOCITY_CONTROLLER_NAME]
                )
            elif key == "o":
                control_mode = "orientation"
                node.get_logger().info("Orientation control mode")
                controller_switcher.switch_controllers(
                    [VELOCITY_CONTROLLER_NAME], [JOINT_TRAJECTORY_CONTROLLER_NAME]
                )
            elif key == "\x03":  # Ctrl+C
                break

    except Exception as exc:
        print(repr(exc))

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
