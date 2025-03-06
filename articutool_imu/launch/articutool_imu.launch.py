# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("articutool_imu"), "config", "imu_params.yaml"
    )

    return LaunchDescription(
        [
            Node(
                package="articutool_imu",
                executable="imu_publisher",
                name="imu_node",
                parameters=[config],
                output="screen",
            )
        ]
    )
