# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import os
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory # Keep if needed for future params

def generate_launch_description():
    
    articutool_orientation_directory = get_package_share_directory(
        "articutool_orientation" 
    )

    # --- Old Madgwick Node (Commented Out/Removed) ---
    # madgwick_orientation_node = launch_ros.actions.Node(
    #     package="imu_filter_madgwick",
    #     executable="imu_filter_madgwick_node",
    #     name="madgwick_orientation_publisher", # Renamed from real_orientation_publisher
    #     output="screen",
    #     remappings=[
    #         ("imu/data_raw", "articutool/imu_data"),
    #         ("imu/mag", "articutool/magnetic_field"),
    #         ("imu/data", "articutool/imu_data_and_orientation"), # Madgwick output
    #     ],
    #     parameters=[
    #         os.path.join(articutool_orientation_directory, "config", "imu_filter.yaml")
    #     ],
    # )

    ekf_orientation_node = launch_ros.actions.Node(
        package="articutool_orientation",
        executable="ekf_quaternion_orientation_estimator",
        name="ekf_orientation_estimator", # Node name
        output="screen",
    )

    return launch.LaunchDescription(
        [
            ekf_orientation_node,
        ]
    )
