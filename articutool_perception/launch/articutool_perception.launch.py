# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import launch
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare arguments for configuration
    camera_name_arg = DeclareLaunchArgument(
        "camera_name", default_value="camera", description="Name of the camera node"
    )

    # Launch the realsense2_camera_node
    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name=LaunchConfiguration("camera_name"),
    )

    # Launch your RealSenseProcessor node
    processor_node = Node(
        package="articutool_perception",
        executable="realsense_processor",
        name="realsense_processor_node",
    )

    return launch.LaunchDescription(
        [
            camera_name_arg,
            realsense_node,
            processor_node,
        ]
    )
