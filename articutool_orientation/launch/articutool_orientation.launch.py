# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import os
import launch
import launch_ros.actions
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import (
    LaunchConfiguration,
    PythonExpression,
    PathJoinSubstitution,
)
from launch.conditions import IfCondition
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument


def generate_launch_description():
    articutool_orientation_directory = get_package_share_directory(
        "articutool_orientation"
    )

    # Declare the launch argument 'filter_type'
    declare_filter_type_arg = DeclareLaunchArgument(
        "filter_type",
        default_value="complementary",
        description="Type of orientation filter to use.",
        choices=["ekf", "complementary", "madgwick"],
    )
    filter_type = LaunchConfiguration("filter_type")

    madgwick_orientation_node = launch_ros.actions.Node(
        package="imu_filter_madgwick",
        executable="imu_filter_madgwick_node",
        name="madgwick_orientation_publisher",
        output="screen",
        remappings=[
            ("imu/data_raw", "articutool/imu_data"),
            ("imu/mag", "articutool/magnetic_field"),
            ("imu/data", "articutool/imu_data_and_orientation"),
        ],
        parameters=[
            os.path.join(
                articutool_orientation_directory, "config", "madgwick_filter.yaml"
            )
        ],
        condition=IfCondition(PythonExpression(["'", filter_type, "' == 'madgwick'"])),
    )

    complementary_orientation_node = launch_ros.actions.Node(
        package="imu_complementary_filter",
        executable="complementary_filter_node",
        name="complementary_orientation_publisher",
        output="screen",
        remappings=[
            ("imu/data_raw", "articutool/imu_data"),
            ("imu/mag", "articutool/magnetic_field"),
            ("imu/data", "articutool/imu_data_and_orientation"),
        ],
        parameters=[
            os.path.join(
                articutool_orientation_directory, "config", "complementary_filter.yaml"
            )
        ],
        condition=IfCondition(
            PythonExpression(["'", filter_type, "' == 'complementary'"])
        ),
    )

    ekf_orientation_node = launch_ros.actions.Node(
        package="articutool_orientation",
        executable="ekf_quaternion_orientation_estimator",
        name="ekf_orientation_estimator",
        output="screen",
        condition=IfCondition(PythonExpression(["'", filter_type, "' == 'ekf'"])),
    )

    orientation_calibration_service_node = Node(
        package="articutool_orientation",
        executable="orientation_calibration_service",
        name="orientation_calibration_service",
        output="screen",
        parameters=[
            {"imu_input_topic": "/articutool/imu_data_and_orientation"},
            {"imu_output_topic": "/articutool/imu_data_and_orientation_calibrated"},
            {"robot_base_frame": "j2n6s200_link_base"},
            {"articutool_mount_frame": "atool_imu_frame"},
        ],
    )

    orientation_relay_node = Node(
        package="articutool_orientation",
        executable="orientation_relay_node",
        name="orientation_relay",
        output="screen",
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    filter_type,
                    "' == 'madgwick' or '",
                    filter_type,
                    "' == 'complementary'",
                ]
            )
        ),
    )

    return launch.LaunchDescription(
        [
            declare_filter_type_arg,
            ekf_orientation_node,
            madgwick_orientation_node,
            complementary_orientation_node,
            orientation_relay_node,
            orientation_calibration_service_node,
        ]
    )
