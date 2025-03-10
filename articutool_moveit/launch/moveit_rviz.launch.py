# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.utilities import normalize_to_list_of_substitutions
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_moveit_rviz_launch


def generate_launch_description():
    # pylint: disable=duplicate-code
    # Launch arguments must be re-declared to be evaluated in context

    ld = LaunchDescription()

    # Sim Launch Argument
    sim_da = DeclareLaunchArgument(
        "sim",
        default_value="real",
        description="Which sim to use: 'mock', 'isaac', or 'real'",
    )
    sim = LaunchConfiguration("sim")
    ld.add_action(sim_da)

    # Log Level
    log_level_da = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Logging level (debug, info, warn, error, fatal)",
    )
    log_level = LaunchConfiguration("log_level")
    log_level_cmd_line_args = ["--ros-args", "--log-level", log_level]
    ld.add_action(log_level_da)

    # End-effector Tool Launch Argument
    eet_da = DeclareLaunchArgument(
        "end_effector_tool",
        default_value="fork",
        description="The end-effector tool being used",
        choices=["fork"],
    )
    end_effector_tool = LaunchConfiguration("end_effector_tool")
    ld.add_action(eet_da)

    # U2D2 USB Port Launch Argument
    u2d2_port_da = DeclareLaunchArgument(
        "u2d2_port",
        default_value="/dev/u2d2",
        description="The USB port corresponding to the U2D2",
    )
    u2d2_port = LaunchConfiguration("u2d2_port")
    ld.add_action(u2d2_port_da)

    # Get MoveIt Configs
    builder = MoveItConfigsBuilder("articutool", package_name="articutool_moveit")
    builder = builder.robot_description(
        mappings={"sim": sim, "end_effector_tool": end_effector_tool, "u2d2_port": u2d2_port}
    )
    moveit_config = builder.to_moveit_configs()

    entities = generate_moveit_rviz_launch(moveit_config).entities
    for entity in entities:
        if isinstance(entity, Node):
            entity.cmd.extend(
                [
                    normalize_to_list_of_substitutions(arg)
                    for arg in log_level_cmd_line_args
                ]
            )
        ld.add_action(entity)

    return ld
