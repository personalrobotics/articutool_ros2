# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import os
import launch
import launch.actions
import launch.conditions
import launch.substitutions
import launch_ros.actions
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    articutool_orientation_directory = get_package_share_directory("articutool_orientation")

    # Declare sim launch argument
    sim_arg = DeclareLaunchArgument(
        "sim",
        default_value="mock",
        description="Which sim to use:",
        choices=["mock", "real"],
    )

    sim = LaunchConfiguration("sim")

    real_orientation_node = launch_ros.actions.Node(
        package="imu_filter_madgwick",
        executable="imu_filter_madgwick_node",
        name="real_orientation_publisher",
        output="screen",
        condition=IfCondition(
            PythonExpression(
                ["'", sim, "' == 'real'"]
            )
        ), 
        remappings=[
            ("imu/data_raw", "articutool/imu_data"),
            ("imu/mag", "articutool/magnetic_field"),
            ("imu/data", "articutool/estimated_orientation"),
        ],
        parameters=[os.path.join(articutool_orientation_directory, "config", "imu_filter.yaml")],
    )

    mock_orientation_node = launch_ros.actions.Node(
        package="articutool_orientation",
        executable="dummy_orientation_publisher",
        name="mock_orientation_publisher",
        output="screen",
        condition=IfCondition(
            PythonExpression(
                ["'", sim, "' == 'mock'"]
            )
        ), 
        parameters=[{
            "target_frame": "atool_imu_frame",
            "source_frame": "world",
            "publish_rate": 50.0
        }],
    )

    return launch.LaunchDescription(
        [
            sim_arg,
            real_orientation_node,
            mock_orientation_node,
        ]
    )
