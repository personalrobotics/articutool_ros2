# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import os
import launch
import launch.actions
import launch.substitutions
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_dir = os.path.join(
        get_package_share_directory("articutool_orientation"), "config"
    )

    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="imu_filter_madgwick",
                executable="imu_filter_madgwick_node",
                name="imu_filter",
                output="screen",
                remappings=[
                    ("imu/data_raw", "articutool/imu_data"),
                    ("imu/mag", "articutool/magnetic_field"),
                    ("imu/data", "articutool/estimated_orientation"),
                ],
                parameters=[os.path.join(config_dir, "imu_filter.yaml")],
            )
        ]
    )
