# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import rclpy
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Quaternion
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from filterpy.kalman import ExtendedKalmanFilter
import math
import tf_transformations
from tf_transformations import quaternion_from_euler, euler_from_quaternion


class OrientationEstimator(Node):
    def __init__(self):
        super().__init__("orientation_estimator")
        self.imu_subscription = self.create_subscription(
            Imu, "imu_data", self.imu_callback, 10
        )
        self.mag_subscription = self.create_subscription(
            MagneticField, "magnetic_field", self.mag_callback, 10
        )
        self.orientation_publisher = self.create_publisher(Quaternion, "estimated_orientation", 10)
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Initialize Kalman filter
        self.kf = ExtendedKalmanFilter(dim_x=6, dim_z=6)  # [roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot]
        self.kf.x = np.zeros((6, 1))  # Initial state
        self.kf.P *= 10.0  # Initial covariance
        self.kf.R = np.eye(6) * 0.0001  # Measurement noise
        self.kf.Q = np.eye(6) * 0.0001  # Process noise

        self.last_time = self.get_clock().now().nanoseconds / 1e9
        self.mag_data = None

    def imu_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.last_time
        self.last_time = current_time

        accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        self.predict(gyro, dt)
        self.update_accel(accel)
        if self.mag_data is not None:
            self.update_mag(self.mag_data)

        roll, pitch, yaw = self.kf.x[0, 0], self.kf.x[1, 0], self.kf.x[2, 0]
        quat_list = quaternion_from_euler(roll, pitch, yaw)
        orientation_quat = Quaternion(x=quat_list[0], y=quat_list[1], z=quat_list[2], w=quat_list[3])

        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = ["atool_root_to_roll", "atool_roll_to_pitch", "atool_pitch_to_yaw"]
        joint_state.position = [roll, pitch, yaw]

        self.joint_state_publisher.publish(joint_state)
        self.orientation_publisher.publish(orientation_quat)

    def mag_callback(self, msg):
        self.mag_data = np.array([msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z])

    def predict(self, gyro, dt):
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.kf.x = self.kf.F @ self.kf.x + np.array([[0], [0], [0], [gyro[0]], [gyro[1]], [gyro[2]]]) * dt
        self.kf.predict()

    def update_accel(self, accel):
        roll = math.atan2(accel[1], accel[2])
        pitch = math.atan2(-accel[0], math.sqrt(accel[1] ** 2 + accel[2] ** 2))

        def Hx_func(x):
            return np.array([[x[0, 0]], [x[1, 0]], [0], [0], [0], [0]])

        def H_jacobian(x):
            return np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ])

        z = np.array([[roll], [pitch], [0], [0], [0], [0]])

        self.kf.update(z, H_jacobian, Hx_func)

    def update_mag(self, mag):
        roll, pitch, yaw = self.kf.x[0, 0], self.kf.x[1, 0], self.kf.x[2, 0]
        mx, my, mz = mag
        expected_mx = mx * math.cos(yaw) + my * math.sin(yaw)
        yaw_mag = math.atan2(my, expected_mx)

        def Hx_func(x):
            return np.array([[0], [0], [x[2, 0]], [0], [0], [0]])

        def H_jacobian(x):
            return np.array([
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ])

        z = np.array([[0], [0], [yaw_mag], [0], [0], [0]])

        self.kf.update(z, H_jacobian, Hx_func)

def main(args=None):
    rclpy.init(args=args)
    orientation_estimator = OrientationEstimator()
    rclpy.spin(orientation_estimator)
    orientation_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
