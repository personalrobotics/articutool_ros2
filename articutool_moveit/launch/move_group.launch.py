# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.utilities import normalize_to_list_of_substitutions
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch


def get_move_group_launch(context):
    # pylint: disable=duplicate-code
    # Launch arguments must be re-declared to be evaluated in context

    """
    Gets the launch description for MoveGroup, after removing sensors_3d
    if sim is mock.

    Adapted from https://robotics.stackexchange.com/questions/104340/getting-the-value-of-launchargument-inside-python-launch-file
    """
    sim = LaunchConfiguration("sim").perform(context)
    log_level = LaunchConfiguration("log_level").perform(context)
    end_effector_tool = LaunchConfiguration("end_effector_tool").perform(context)
    u2d2_port = LaunchConfiguration("u2d2_port")

    # Get MoveIt Configs
    builder = MoveItConfigsBuilder("articutool", package_name="articutool_moveit")
    builder = builder.robot_description(
        mappings={
            "sim": sim,
            "end_effector_tool": end_effector_tool,
            "u2d2_port": u2d2_port,
        }
    )
    builder = builder.robot_description_semantic(
        mappings={"end_effector_tool": end_effector_tool}
    )
    moveit_config = builder.to_moveit_configs()

    # If sim is mock, set moveit_config.sensors_3d to an empty dictionary
    if sim == "mock":
        moveit_config.sensors_3d = {}

    entities = generate_move_group_launch(moveit_config).entities
    log_level_cmd_line_args = ["--ros-args", "--log-level", log_level]
    for entity in entities:
        if isinstance(entity, Node):
            entity.cmd.extend(
                [
                    normalize_to_list_of_substitutions(arg)
                    for arg in log_level_cmd_line_args
                ]
            )
    return entities


def generate_launch_description():
    # pylint: disable=duplicate-code
    # Launch arguments must be re-declared to be evaluated in context

    # Sim Launch Argument
    sim_da = DeclareLaunchArgument(
        "sim",
        default_value="real",
        description="Which sim to use: 'mock', 'isaac', or 'real'",
    )
    # Log Level
    log_level_da = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Logging level (debug, info, warn, error, fatal)",
    )
    eet_da = DeclareLaunchArgument(
        "end_effector_tool",
        default_value="fork",
        description="The end-effector tool being used",
        choices=["fork"],
    )
    # U2D2 USB Port Launch Argument
    u2d2_port_da = DeclareLaunchArgument(
        "u2d2_port",
        default_value="/dev/u2d2",
        description="The USB port corresponding to the U2D2",
    )

    ld = LaunchDescription()
    ld.add_action(sim_da)
    ld.add_action(log_level_da)
    ld.add_action(eet_da)
    ld.add_action(u2d2_port_da)
    ld.add_action(OpaqueFunction(function=get_move_group_launch))
    return ld
