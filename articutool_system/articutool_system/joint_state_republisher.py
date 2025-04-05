#!/usr/bin/env python3

# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class JointStateRepublisher(Node):
    """
    A simple ROS 2 node that subscribes to JointState messages on one topic
    and republishes them to another topic.
    """

    def __init__(self):
        super().__init__("joint_state_republisher")

        # Declare parameters for input/output topics (optional, but good practice)
        self.declare_parameter("input_topic", "/articutool/joint_states")
        self.declare_parameter("output_topic", "/joint_states")

        input_topic = (
            self.get_parameter("input_topic").get_parameter_value().string_value
        )
        output_topic = (
            self.get_parameter("output_topic").get_parameter_value().string_value
        )

        # Define QoS profile - typically matches default for joint_states
        # (Reliable, Volatile - keeps only the last message)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # Keep only the latest message
        )
        # For broader compatibility or if issues arise, try KEEP_ALL or BEST_EFFORT:
        # qos_profile = QoSProfile(depth=10) # Default ROS 2 QoS

        # Create the publisher to the target topic
        self.publisher_ = self.create_publisher(
            JointState, output_topic, qos_profile  # Use defined QoS
        )

        # Create the subscriber to the source topic
        self.subscription_ = self.create_subscription(
            JointState,
            input_topic,
            self.listener_callback,
            qos_profile,  # Use defined QoS matching the publisher if possible
        )

        self.get_logger().info(
            f"JointStateRepublisher node started. "
            f"Subscribing to '{input_topic}', "
            f"republishing to '{output_topic}'."
        )

    def listener_callback(self, msg):
        """
        Callback function for the subscriber.
        Receives a JointState message and republishes it.
        """
        # Simply republish the received message
        self.publisher_.publish(msg)
        # self.get_logger().debug(f"Republished JointState message on {self.publisher_.topic}") # Optional debug log


def main(args=None):
    rclpy.init(args=args)
    try:
        republisher_node = JointStateRepublisher()
        rclpy.spin(republisher_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if "republisher_node" in locals():
            republisher_node.get_logger().error(f"Node crashed: {e}")
        else:
            print(f"Error during node initialization: {e}")
    finally:
        # Cleanup
        if "republisher_node" in locals() and rclpy.ok():
            republisher_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
