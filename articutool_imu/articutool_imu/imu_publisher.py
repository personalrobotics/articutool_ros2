# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
import serial
import time
import math
from geometry_msgs.msg import Quaternion, Vector3
from std_msgs.msg import Header


class IMUPublisher(Node):
    def __init__(self):
        super().__init__("imu_publisher")
        self.declare_parameter("imu_port", "/dev/imu")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("frame_id", "atool_imu_frame")

        self.imu_port = (
            self.get_parameter("imu_port").get_parameter_value().string_value
        )
        self.baud_rate = (
            self.get_parameter("baud_rate").get_parameter_value().integer_value
        )
        self.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )

        self.imu_publisher_ = self.create_publisher(Imu, "articutool/imu_data", 10)
        self.mag_publisher_ = self.create_publisher(
            MagneticField, "articutool/magnetic_field", 10
        )

        timer_period = 0.01  # 100 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.ser = serial.Serial(self.imu_port)
        self.ser.baudrate = self.baud_rate
        self.get_logger().info(
            f"connected to: {self.ser.portstr} at {self.baud_rate} baud"
        )

        # ignore header information - wait for the empty line signifying header is over
        while True:
            line = str(self.ser.readline())
            if line == "b'\\r\\n'":
                break

    def timer_callback(self):
        self.ser.flushInput()
        self.ser.readline()
        line = str(self.ser.readline())
        data = list(map(str.strip, line.split(",")))

        if len(data) == 14:
            try:
                accel_x = float(data[2])
                accel_y = float(data[3])
                accel_z = float(data[4])
                gyro_x = math.radians(float(data[5]))
                gyro_y = math.radians(float(data[6]))
                gyro_z = math.radians(float(data[7]))
                mag_x = float(data[8])
                mag_y = float(data[9])
                mag_z = float(data[10])

                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = self.frame_id

                imu_msg = Imu()
                imu_msg.header = header
                imu_msg.linear_acceleration = Vector3(x=accel_x, y=accel_y, z=accel_z)
                imu_msg.angular_velocity = Vector3(x=gyro_x, y=gyro_y, z=gyro_z)
                imu_msg.orientation_covariance = [
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                imu_msg.angular_velocity_covariance = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
                imu_msg.linear_acceleration_covariance = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]

                mag_msg = MagneticField()
                mag_msg.header = header
                mag_msg.magnetic_field = Vector3(x=mag_x, y=mag_y, z=mag_z)
                mag_msg.magnetic_field_covariance = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]

                self.imu_publisher_.publish(imu_msg)
                self.mag_publisher_.publish(mag_msg)
            except ValueError:
                self.get_logger().warn(f"Could not convert data to float: {data}")
            except IndexError:
                self.get_logger().warn(f"Index error occurred with data: {data}")

        else:
            self.get_logger().warn(f"Incorrect number of data points: {data}")


def main(args=None):
    rclpy.init(args=args)
    imu_publisher = IMUPublisher()
    rclpy.spin(imu_publisher)
    imu_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
