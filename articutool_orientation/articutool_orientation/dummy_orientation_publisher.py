#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025, Personal Robotics Laboratory & Google LLC
# License: BSD 3-Clause

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from geometry_msgs.msg import QuaternionStamped

class DummyOrientationPublisher(Node):
    """
    Node to listen to TF transforms, extract orientation, and publish as QuaternionStamped.

    Continuously looks up the transform between a specified source_frame and 
    target_frame, extracts the orientation (quaternion), and publishes it on 
    the '/articutool/estimated_orientation' topic
    """
    def __init__(self):
        super().__init__('dummy_orientation_publisher')

        # --- Parameters ---
        self.declare_parameter('target_frame', 'atool_imu_frame')
        self.declare_parameter('source_frame', 'world')
        self.declare_parameter('publish_rate', 50.0)

        self.target_frame_ = self.get_parameter('target_frame').get_parameter_value().string_value
        self.source_frame_ = self.get_parameter('source_frame').get_parameter_value().string_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        if not self.target_frame_ or not self.source_frame_:
            self.get_logger().error("'target_frame' and 'source_frame' parameters must be set.")
            raise ValueError("Missing required frame parameters")
            
        if publish_rate <= 0:
             self.get_logger().error("'publish_rate' must be positive.")
             raise ValueError("Invalid publish_rate parameter")

        self.get_logger().info(f"Configured to publish orientation of '{self.target_frame_}' "
                               f"relative to '{self.source_frame_}' at {publish_rate} Hz.")

        self.tf_buffer_ = Buffer() 
        self.tf_listener_ = TransformListener(self.tf_buffer_, self, spin_thread=True) 

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1 
        )
        self.publisher_ = self.create_publisher(
            QuaternionStamped, 
            'articutool/estimated_orientation',
            qos_profile
        )

        timer_period_sec = 1.0 / publish_rate
        self.timer_ = self.create_timer(timer_period_sec, self.timer_callback)

        self.get_logger().info('TF to Orientation Publisher node started.')

    def timer_callback(self):
        """
        Called periodically by the timer to look up the transform and publish.
        """
        now = self.get_clock().now()
        
        try:
            transform = self.tf_buffer_.lookup_transform(
                self.target_frame_,
                self.source_frame_,
                Time(seconds=0),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
        except TransformException as ex:
            self.get_logger().warn(
                f'Could not transform {self.target_frame_} to {self.source_frame_}: {ex}',
                throttle_duration_sec=1.0
            )
            return

        msg = QuaternionStamped()
        
        msg.header.stamp = transform.header.stamp
        msg.header.frame_id = self.source_frame_ 
        original_quat = transform.transform.rotation
        msg.quaternion.x = -original_quat.x
        msg.quaternion.y = -original_quat.y
        msg.quaternion.z = -original_quat.z
        msg.quaternion.w = original_quat.w

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DummyOrientationPublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except ValueError as e:
        if node:
            node.get_logger().error(f"Configuration error: {e}")
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
