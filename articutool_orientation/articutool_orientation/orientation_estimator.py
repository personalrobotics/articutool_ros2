# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math


class OrientationEstimator(Node):
    def __init__(self):
        super().__init__("orientation_estimator")
        self.subscription = self.create_subscription(
            Imu, "imu_data", self.imu_callback, 10
        )
        self.publisher_ = self.create_publisher(Quaternion, "estimated_orientation", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

    def imu_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Extract accelerometer and gyroscope data
        accel_x = msg.linear_acceleration.x
        accel_y = msg.linear_acceleration.y
        accel_z = msg.linear_acceleration.z
        gyro_x = msg.angular_velocity.x
        gyro_y = msg.angular_velocity.y
        gyro_z = msg.angular_velocity.z

        # Simple complementary filter
        roll = math.atan2(accel_y, accel_z)
        pitch = math.atan2(-accel_x, math.sqrt(accel_y**2 + accel_z**2))
        yaw = 0.0  # yaw is not determined from accelerometer alone

        # Convert roll, pitch, yaw to quaternion
        qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(
            roll / 2
        ) * math.sin(pitch / 2) * math.sin(yaw / 2)
        qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(
            roll / 2
        ) * math.cos(pitch / 2) * math.sin(yaw / 2)
        qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(
            roll / 2
        ) * math.sin(pitch / 2) * math.cos(yaw / 2)
        qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(
            roll / 2
        ) * math.sin(pitch / 2) * math.sin(yaw / 2)

        orientation_quat = Quaternion(x=qx, y=qy, z=qz, w=qw)

        # Publish the quaternion
        self.publisher_.publish(orientation_quat)

        # Broadcast the transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "root"
        t.child_frame_id = "atool_handle"
        t.transform.rotation = orientation_quat
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    orientation_estimator = OrientationEstimator()
    rclpy.spin(orientation_estimator)
    orientation_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
