# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    imu_port_arg = DeclareLaunchArgument(
        "imu_port",
        default_value="/dev/imu",
        description="USB port for the IMU",
    )
    imu_port = LaunchConfiguration("imu_port")

    return LaunchDescription(
        [
            imu_port_arg,
            Node(
                package="articutool_imu",
                executable="imu_publisher",
                name="imu_node",
                parameters=[{"imu_port": imu_port}],
                output="screen",
            )
        ]
    )
