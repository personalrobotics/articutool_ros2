# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
    AnyLaunchDescriptionSource,
)
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    TextSubstitution,
)

from launch_ros.actions import Node

from srdfdom.srdf import SRDF

from moveit_configs_utils.launch_utils import (
    add_debuggable_node,
    DeclareBooleanLaunchArg,
)


def generate_launch_description():
    # Sim Launch Argument
    sim_da = DeclareLaunchArgument(
        "sim",
        default_value="real",
        description="Which sim to use:",
        choices=["mock", "real"]
    )
    sim = LaunchConfiguration("sim")

    # End-effector Tool Launch Argument
    eet_da = DeclareLaunchArgument(
        "end_effector_tool",
        default_value="fork",
        description="The end-effector tool being used",
        choices=["fork"],
    )
    end_effector_tool = LaunchConfiguration("end_effector_tool")

    # U2D2 USB Port Launch Argument
    u2d2_port_da = DeclareLaunchArgument(
        "u2d2_port",
        default_value="/dev/u2d2",
        description="The USB port corresponding to the U2D2",
    )
    u2d2_port = LaunchConfiguration("u2d2_port")

    # Controllers File
    ctrl_da = DeclareLaunchArgument(
        "controllers_file",
        default_value=["controllers.yaml"],
        description="ROS2 Controller YAML configuration in config folder",
    )
    controllers_file = LaunchConfiguration("controllers_file")

    # Log Level
    log_level_da = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Logging level (debug, info, warn, error, fatal)",
    )
    log_level = LaunchConfiguration("log_level")

    # Copy from generate_demo_launch
    ld = LaunchDescription()
    ld.add_action(sim_da)
    ld.add_action(eet_da)
    ld.add_action(ctrl_da)
    ld.add_action(log_level_da)
    ld.add_action(u2d2_port_da)

    # Get MoveIt Configs
    builder = MoveItConfigsBuilder("articutool", package_name="articutool_moveit")
    builder = builder.robot_description(
        mappings={"sim": sim, "end_effector_tool": end_effector_tool, "u2d2_port": u2d2_port}
    )
    builder = builder.robot_description_semantic(
        mappings={"end_effector_tool": end_effector_tool}
    )
    moveit_config = builder.to_moveit_configs()

    ld.add_action(
        DeclareBooleanLaunchArg(
            "debug",
            default_value=False,
            description="By default, we are not in debug mode",
        )
    )
    ld.add_action(DeclareBooleanLaunchArg("use_rviz", default_value=True))

    # If there are virtual joints, broadcast static tf by including virtual_joints launch
    virtual_joints_launch = (
        moveit_config.package_path / "launch/static_virtual_joint_tfs.launch.py"
    )
    if virtual_joints_launch.exists():
        ld.add_action(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(virtual_joints_launch)),
                launch_arguments={
                    "sim": sim,
                    "log_level": log_level,
                    "end_effector_tool": end_effector_tool,
                    "u2d2_port": u2d2_port,
                }.items(),
            )
        )

    # Given the published joint states, publish tf for the robot links
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(moveit_config.package_path / "launch/rsp.launch.py")
            ),
            launch_arguments={
                "sim": sim,
                "log_level": log_level,
                "end_effector_tool": end_effector_tool,
                "u2d2_port": u2d2_port,
            }.items(),
        )
    )

    # Launch the Move Group
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(moveit_config.package_path / "launch/move_group.launch.py")
            ),
            launch_arguments={
                "sim": sim,
                "log_level": log_level,
                "end_effector_tool": end_effector_tool,
                "u2d2_port": u2d2_port,
            }.items(),
        )
    )

    # Run Rviz and load the default config to see the state of the move_group node
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(moveit_config.package_path / "launch/moveit_rviz.launch.py")
            ),
            launch_arguments={
                "sim": sim,
                "log_level": log_level,
                "end_effector_tool": end_effector_tool,
                "u2d2_port": u2d2_port,
            }.items(),
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        )
    )

    robot_controllers = PathJoinSubstitution(
        [str(moveit_config.package_path), "config", controllers_file]
    )

    # Joint Controllers
    ld.add_action(
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[moveit_config.robot_description, robot_controllers],
            arguments=["--ros-args", "--log-level", log_level],
        )
    )

    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(moveit_config.package_path / "launch/spawn_controllers.launch.py")
            ),
            launch_arguments={
                "sim": sim,
                "log_level": log_level,
                "end_effector_tool": end_effector_tool,
                "u2d2_port": u2d2_port,
            }.items(),
        )
    )

    return ld
