#!/usr/bin/env python3

# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Mock IMU Publisher Node.
This node simulates IMU data by reading the TF transform between a robot base frame
and an IMU frame, and publishing this orientation as sensor_msgs/Imu messages
on specified topics. It's intended for simulation scenarios where actual IMU
hardware or detailed simulation is unavailable.
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3
import tf2_ros
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)
import numpy as np


class MockImuPublisher(Node):
    """
    Publishes mock Imu messages based on TF transforms.
    The orientation is taken from TF. Linear acceleration and angular velocity
    are set to zero, and covariances are set to small values indicating
    high confidence in the mock data.
    """

    def __init__(self):
        super().__init__("mock_imu_publisher")

        # Declare parameters
        self.declare_parameter(
            "robot_base_frame",
            "j2n6s200_link_base",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="The TF frame of the robot's base.",
            ),
        )
        self.declare_parameter(
            "articutool_imu_frame",
            "atool_imu_frame",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="The TF frame of the Articutool's IMU.",
            ),
        )
        self.declare_parameter(
            "uncalibrated_imu_topic",
            "/articutool/imu_data_and_orientation",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic for the 'uncalibrated' (FilterWorld-relative) mock IMU data.",
            ),
        )
        self.declare_parameter(
            "calibrated_imu_topic",
            "/articutool/imu_data_and_orientation_calibrated",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic for the 'calibrated' (RobotBase-relative) mock IMU data.",
            ),
        )
        self.declare_parameter(
            "publish_rate_hz",
            50.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Rate at which to publish mock IMU data.",
            ),
        )

        # Get parameters
        self.robot_base_frame = (
            self.get_parameter("robot_base_frame").get_parameter_value().string_value
        )
        self.articutool_imu_frame = (
            self.get_parameter("articutool_imu_frame")
            .get_parameter_value()
            .string_value
        )
        uncalibrated_topic = (
            self.get_parameter("uncalibrated_imu_topic")
            .get_parameter_value()
            .string_value
        )
        calibrated_topic = (
            self.get_parameter("calibrated_imu_topic")
            .get_parameter_value()
            .string_value
        )
        publish_rate = (
            self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        )

        if publish_rate <= 0:
            self.get_logger().error("Publish rate must be positive. Setting to 1.0 Hz.")
            publish_rate = 1.0

        self.get_logger().info(f"Mock IMU Publisher Configuration:")
        self.get_logger().info(f"  Robot Base Frame: {self.robot_base_frame}")
        self.get_logger().info(f"  Articutool IMU Frame: {self.articutool_imu_frame}")
        self.get_logger().info(f"  Uncalibrated Topic: {uncalibrated_topic}")
        self.get_logger().info(f"  Calibrated Topic: {calibrated_topic}")
        self.get_logger().info(f"  Publish Rate: {publish_rate} Hz")

        # TF Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self, spin_thread=True
        )  # Spin listener in a separate thread

        # Publishers
        self.uncalibrated_imu_pub = self.create_publisher(Imu, uncalibrated_topic, 10)
        self.calibrated_imu_pub = self.create_publisher(Imu, calibrated_topic, 10)

        # Timer for publishing
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_mock_imu)

        self.get_logger().info("Mock IMU publisher node started.")

    def publish_mock_imu(self):
        """
        Looks up the TF transform and publishes Imu messages.
        """
        now = self.get_clock().now()
        try:
            # Lookup the transform from robot_base_frame to articutool_imu_frame
            # This transform represents the pose of the IMU frame in the robot base frame.
            # For the "uncalibrated" topic, the FilterWorld is effectively the robot_base_frame.
            # For the "calibrated" topic, this is directly the orientation we want.
            transform_stamped = self.tf_buffer.lookup_transform(
                self.robot_base_frame,  # Target frame
                self.articutool_imu_frame,  # Source frame
                rclpy.time.Time(),  # Get the latest available transform
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(
                f"Could not get transform from '{self.robot_base_frame}' to "
                f"'{self.articutool_imu_frame}': {e}",
                throttle_duration_sec=5.0,  # Avoid spamming logs
            )
            return

        imu_msg = Imu()
        imu_msg.header.stamp = now.to_msg()
        imu_msg.header.frame_id = self.articutool_imu_frame  # Data is for this frame

        # Orientation from TF
        imu_msg.orientation = transform_stamped.transform.rotation

        # Mock linear acceleration (e.g., zero, or simulate gravity if IMU is Z-up aligned with base Z-down)
        # For simplicity, setting to zero as per user request.
        # If simulating gravity and assuming IMU Z is up and base Z is up:
        # R_base_imu = R.from_quat(...)
        # gravity_base = np.array([0,0,-9.81])
        # accel_imu = R_base_imu.inv().apply(gravity_base)
        # imu_msg.linear_acceleration.x = accel_imu[0] ...
        imu_msg.linear_acceleration.x = 0.0
        imu_msg.linear_acceleration.y = 0.0
        imu_msg.linear_acceleration.z = (
            0.0  # Or 9.81 if IMU's Z is typically up and you want to simulate gravity
        )

        # Mock angular velocity (zero)
        imu_msg.angular_velocity.x = 0.0
        imu_msg.angular_velocity.y = 0.0
        imu_msg.angular_velocity.z = 0.0

        # Covariances: Small values indicating high confidence (mock data)
        # Order: [xx, xy, xz, yx, yy, yz, zx, zy, zz]
        # Set diagonal elements to a small number, others to 0.
        small_covariance = 1e-9
        imu_msg.orientation_covariance[0] = small_covariance  # R
        imu_msg.orientation_covariance[4] = small_covariance  # P
        imu_msg.orientation_covariance[8] = small_covariance  # Y

        imu_msg.angular_velocity_covariance[0] = small_covariance
        imu_msg.angular_velocity_covariance[4] = small_covariance
        imu_msg.angular_velocity_covariance[8] = small_covariance

        imu_msg.linear_acceleration_covariance[0] = small_covariance
        imu_msg.linear_acceleration_covariance[4] = small_covariance
        imu_msg.linear_acceleration_covariance[8] = small_covariance

        # Publish on both topics
        # For the "uncalibrated" topic, this means FilterWorld effectively IS robot_base_frame.
        # The OrientationCalibrationService will then compute an identity transform for calibration.
        self.uncalibrated_imu_pub.publish(imu_msg)

        # For the "calibrated" topic, this is directly the orientation of IMU in robot_base_frame.
        # If OrientationCalibrationService also subscribes to this for its output,
        # it will effectively pass it through after its "identity" calibration.
        # However, the typical flow is that OrientationCalibrationService *produces* the calibrated topic.
        # The user request was to publish to *both* topics from this mock node.
        self.calibrated_imu_pub.publish(imu_msg)

        # self.get_logger().debug(f"Published mock IMU data for frame '{self.articutool_imu_frame}'",
        #                         throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MockImuPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info(
                "Mock IMU publisher shutting down due to KeyboardInterrupt."
            )
    except Exception as e:
        if node:
            node.get_logger().fatal(f"Unhandled exception in MockImuPublisher: {e}")
        else:
            print(f"Unhandled exception before node init for MockImuPublisher: {e}")
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
