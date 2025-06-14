# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    GroupAction,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
    AnyLaunchDescriptionSource,
)
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    TextSubstitution,
    PythonExpression,
)

from launch_ros.actions import Node, PushRosNamespace

from srdfdom.srdf import SRDF

from moveit_configs_utils.launch_utils import (
    add_debuggable_node,
    DeclareBooleanLaunchArg,
)
from moveit_configs_utils.launches import (
    generate_rsp_launch,
    generate_move_group_launch,
    generate_spawn_controllers_launch,
    generate_static_virtual_joint_tfs_launch,
    generate_moveit_rviz_launch,
)


def generate_launch_description():
    # Sim Launch Argument
    sim_da = DeclareLaunchArgument(
        "sim",
        default_value="mock",
        description="Which sim to use:",
        choices=["mock", "real"],
    )
    sim = LaunchConfiguration("sim")

    # End-effector Tool Launch Argument
    eet_da = DeclareLaunchArgument(
        "end_effector_tool",
        default_value="fork",
        description="The end-effector tool being used",
        choices=["fork", "spoon"],
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

    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz",
        default_value="false",
        description="Launch MoveIt launch file.",
    )
    launch_rviz = LaunchConfiguration("launch_rviz")

    launch_controllers_arg = DeclareLaunchArgument(
        "launch_controllers",
        default_value="true",
        description="Launch controllers",
    )
    launch_controllers = LaunchConfiguration("launch_controllers")

    # Copy from generate_demo_launch
    ld = LaunchDescription()
    ld.add_action(sim_da)
    ld.add_action(eet_da)
    ld.add_action(ctrl_da)
    ld.add_action(log_level_da)
    ld.add_action(u2d2_port_da)
    ld.add_action(launch_rviz_arg)
    ld.add_action(launch_controllers_arg)

    # Get MoveIt Configs
    builder = MoveItConfigsBuilder("articutool", package_name="articutool_moveit")
    builder = builder.robot_description(
        mappings={
            "sim": sim,
            "end_effector_tool": end_effector_tool,
            "u2d2_port": u2d2_port,
            "launch_controllers": launch_controllers,
        }
    )
    moveit_config = builder.to_moveit_configs()

    # If sim is mock, set moveit_config.sensors_3d to an empty dictionary
    if sim == "mock":
        moveit_config.sensors_3d = {}

    ld.add_action(
        DeclareBooleanLaunchArg(
            "debug",
            default_value=False,
            description="By default, we are not in debug mode",
        )
    )
    ld.add_action(
        DeclareBooleanLaunchArg("allow_trajectory_execution", default_value=True)
    )
    ld.add_action(
        DeclareBooleanLaunchArg("publish_monitored_planning_scene", default_value=True)
    )
    # load non-default MoveGroup capabilities (space separated)
    ld.add_action(
        DeclareLaunchArgument(
            "capabilities",
            default_value=moveit_config.move_group_capabilities["capabilities"],
        )
    )
    # inhibit these default MoveGroup capabilities (space separated)
    ld.add_action(
        DeclareLaunchArgument(
            "disable_capabilities",
            default_value=moveit_config.move_group_capabilities["disable_capabilities"],
        )
    )

    # do not copy dynamics information from /joint_states to internal robot monitoring
    # default to false, because almost nothing in move_group relies on this information
    ld.add_action(DeclareBooleanLaunchArg("monitor_dynamics", default_value=False))

    should_publish = LaunchConfiguration("publish_monitored_planning_scene")

    move_group_configuration = {
        "publish_robot_description_semantic": True,
        "allow_trajectory_execution": LaunchConfiguration("allow_trajectory_execution"),
        # Note: Wrapping the following values is necessary so that the parameter value can be the empty string
        "capabilities": ParameterValue(
            LaunchConfiguration("capabilities"), value_type=str
        ),
        "disable_capabilities": ParameterValue(
            LaunchConfiguration("disable_capabilities"), value_type=str
        ),
        # Publish the planning scene of the physical robot so that rviz plugin can know actual robot
        "publish_planning_scene": should_publish,
        "publish_geometry_updates": should_publish,
        "publish_state_updates": should_publish,
        "publish_transforms_updates": should_publish,
        "monitor_dynamics": False,
    }

    move_group_params = [
        moveit_config.to_dict(),
        move_group_configuration,
    ]

    actions = [
        PushRosNamespace("articutool"),
        # Static Virtual Joints
        *generate_static_virtual_joint_tfs_launch(moveit_config).entities,
        # Robot State Publisher
        *generate_rsp_launch(moveit_config).entities,
        # RViz
        GroupAction(
            actions=generate_moveit_rviz_launch(moveit_config).entities,
            condition=IfCondition(launch_rviz),
        ),
        # Joint Controllers
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[
                moveit_config.robot_description,
                PathJoinSubstitution(
                    [str(moveit_config.package_path), "config", controllers_file]
                ),
            ],
            arguments=["--ros-args", "--log-level", log_level],
            condition=IfCondition(launch_controllers),
        ),
        # Spawn Controllers
        GroupAction(
            actions=generate_spawn_controllers_launch(moveit_config).entities,
            condition=IfCondition(launch_controllers),
        ),
        # Move Group
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            respawn=False,
            parameters=move_group_params,
            remappings=[
                (
                    "/controller_manager/list_controllers",
                    "/articutool/controller_manager/list_controllers",
                ),
            ],
        ),
    ]

    articutool_group = GroupAction(actions=actions)
    ld.add_action(articutool_group)

    return ld
