#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, QuaternionStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Header

class OrientationRelay(Node):
    def __init__(self):
        super().__init__('orientation_relay')
        self.get_logger().info('Orientation Relay node starting...')

        # Get input/output topic names from parameters or use defaults
        self.declare_parameter('input_topic', '/articutool/imu_data_and_orientation')
        self.declare_parameter('output_topic', '/articutool/estimated_orientation')

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.publisher_ = self.create_publisher(
            QuaternionStamped,
            output_topic,
            10)

        self.subscription = self.create_subscription(
            Imu,
            input_topic,
            self.listener_callback,
            10)
        self.subscription

        self.get_logger().info(f"Relaying 'orientation' field from '{input_topic}' [{Imu.__name__}]")
        self.get_logger().info(f"Publishing to '{output_topic}' [{QuaternionStamped.__name__}]")

    def listener_callback(self, msg: Imu):
        try:
            if not hasattr(msg, 'header'):
                self.get_logger().warn(f"Input message on topic '{self.subscription.topic_name}' missing 'header' field. Skipping.", throttle_duration_sec=5)
                return
            if not hasattr(msg, 'orientation'):
                self.get_logger().warn(f"Input message on topic '{self.subscription.topic_name}' missing 'orientation' field. Skipping.", throttle_duration_sec=5)
                return
            if not isinstance(msg.orientation, Quaternion):
                 self.get_logger().warn(f"Field 'orientation' in input message is not a Quaternion. Type: {type(msg.orientation)}. Skipping.", throttle_duration_sec=5)
                 return

            stamped_quat_msg = QuaternionStamped()
            stamped_quat_msg.header = msg.header
            stamped_quat_msg.quaternion = msg.orientation
            self.publisher_.publish(stamped_quat_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing message: {e}")

def main(args=None):
    rclpy.init(args=args)
    orientation_relay = None
    try:
        orientation_relay = OrientationRelay()
        rclpy.spin(orientation_relay)
    except KeyboardInterrupt:
        if orientation_relay:
            orientation_relay.get_logger().info('KeyboardInterrupt, shutting down.')
    except Exception as e:
        if orientation_relay and orientation_relay.get_logger():
             orientation_relay.get_logger().error(f"Error during node execution: {e}")
        else:
             print(f"Error during node execution: {e}")
    finally:
        if orientation_relay:
            orientation_relay.destroy_node()
        if rclpy.ok():
             rclpy.shutdown()

if __name__ == '__main__':
    main()
