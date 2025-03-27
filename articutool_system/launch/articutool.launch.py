# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    # Declare imu_port launch argument
    imu_port_arg = DeclareLaunchArgument(
        "imu_port",
        default_value="/dev/imu",
        description="USB port for the IMU",
    )

    imu_port = LaunchConfiguration("imu_port")

    # Declare u2d2_port launch argument
    u2d2_port_arg = DeclareLaunchArgument(
        "u2d2_port",
        default_value="/dev/u2d2",
        description="USB port for the U2D2",
    )

    u2d2_port = LaunchConfiguration("u2d2_port")

    # Declare sim launch argument
    sim_arg = DeclareLaunchArgument(
        "sim",
        default_value="mock",
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

    # Declare launch imu argument
    launch_imu_arg = DeclareLaunchArgument(
        "launch_imu",
        default_value="false",
        description="Launch IMU launch file.",
    )
    launch_imu = LaunchConfiguration("launch_imu")

    # Declare launch orientation argument
    launch_orientation_arg = DeclareLaunchArgument(
        "launch_orientation",
        default_value="false",
        description="Launch orientation launch file.",
    )
    launch_orientation = LaunchConfiguration("launch_orientation")

    # Declare launch moveit argument
    launch_moveit_arg = DeclareLaunchArgument(
        "launch_moveit",
        default_value="true",
        description="Launch MoveIt launch file.",
    )
    launch_moveit = LaunchConfiguration("launch_moveit")

    # Declare launch orientation control node argument
    launch_orientation_control_arg = DeclareLaunchArgument(
        "launch_orientation_control",
        default_value="false",
        description="Launch orientation control node.",
    )
    launch_orientation_control = LaunchConfiguration("launch_orientation_control")

    # Declare launch rviz argument
    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz",
        default_value="false",
        description="Launch MoveIt launch file.",
    )
    launch_rviz = LaunchConfiguration("launch_rviz")

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
        condition=IfCondition(launch_imu),
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
        condition=IfCondition(launch_orientation),
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
            "u2d2_port": u2d2_port,
            "launch_rviz": launch_rviz,
        }.items(),
        condition=IfCondition(launch_moveit),
    )

    # Launch orientation_control node
    orientation_control_node = Node(
        package="articutool_system",
        executable="orientation_control",
        name="orientation_control",
        output="screen",
        condition=IfCondition(launch_orientation_control),
    )

    return LaunchDescription(
        [
            imu_port_arg,
            u2d2_port_arg,
            sim_arg,
            end_effector_tool_arg,
            controllers_file_arg,
            log_level_arg,
            launch_imu_arg,
            launch_orientation_arg,
            launch_moveit_arg,
            launch_orientation_control_arg,
            launch_rviz_arg,
            imu_launch,
            orientation_launch,
            moveit_launch,
            orientation_control_node,
        ]
    )
