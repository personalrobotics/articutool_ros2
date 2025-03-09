# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node  # Import Node action


def generate_launch_description():
    # Declare imu_port launch argument
    imu_port_arg = DeclareLaunchArgument(
        "imu_port",
        default_value="/dev/ttyUSB1",
        description="USB port for the IMU",
    )

    imu_port = LaunchConfiguration("imu_port")

    # Declare sim launch argument
    sim_arg = DeclareLaunchArgument(
        "sim",
        default_value="real",
        description="Which sim to use:",
        choices=["mock", "real"],
    )

    sim = LaunchConfiguration("sim")

    # Declare end_effector_tool launch argument
    end_effector_tool_arg = DeclareLaunchArgument(
        "end_effector_tool",
        default_value="fork",
        description="The end-effector tool being used",
        choices=["fork"],
    )

    end_effector_tool = LaunchConfiguration("end_effector_tool")

    # Declare controllers_file launch argument
    controllers_file_arg = DeclareLaunchArgument(
        "controllers_file",
        default_value=["controllers.yaml"],
        description="ROS2 Controller YAML configuration in config folder",
    )

    controllers_file = LaunchConfiguration("controllers_file")

    # Declare log_level launch argument
    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Logging level (debug, info, warn, error, fatal)",
    )

    log_level = LaunchConfiguration("log_level")

    # Include articutool_imu launch file
    imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("articutool_imu"),
                    "launch",
                    "articutool_imu.launch.py",
                )
            ]
        ),
        launch_arguments={"imu_port": imu_port}.items(),
    )

    # Include articutool_orientation launch file
    orientation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("articutool_orientation"),
                    "launch",
                    "articutool_orientation.launch.py",
                )
            ]
        ),
    )

    # Include articutool_moveit launch file
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("articutool_moveit"),
                    "launch",
                    "articutool_moveit.launch.py",
                )
            ]
        ),
        launch_arguments={
            "sim": sim,
            "end_effector_tool": end_effector_tool,
            "controllers_file": controllers_file,
            "log_level": log_level,
        }.items(),
    )

    # Launch orientation_control node
    orientation_control_node = Node(
        package="articutool_system",
        executable="orientation_control",
        name="orientation_control",
        output="screen",
    )

    return LaunchDescription(
        [
            imu_port_arg,
            sim_arg,
            end_effector_tool_arg,
            controllers_file_arg,
            log_level_arg,
            imu_launch,
            orientation_launch,
            moveit_launch,
            orientation_control_node,
        ]
    )
