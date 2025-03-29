# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch
from launch import LaunchDescription
from launch_ros.parameter_descriptions import ParameterValue
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

def generate_rsp_launch(moveit_config):
    """Launch file for robot state publisher (rsp)"""

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument("publish_frequency", default_value="15.0"))

    # Given the published joint states, publish tf for the robot links and the robot description
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        respawn=True,
        output="screen",
        parameters=[
            moveit_config.robot_description,
            {
                "publish_frequency": LaunchConfiguration("publish_frequency"),
            },
        ],
    )
    ld.add_action(rsp_node)

    return ld

def generate_moveit_rviz_launch(moveit_config):
    """Launch file for rviz"""
    ld = LaunchDescription()

    ld.add_action(DeclareBooleanLaunchArg("debug", default_value=False))
    ld.add_action(
        DeclareLaunchArgument(
            "rviz_config",
            default_value=str(moveit_config.package_path / "config/moveit.rviz"),
        )
    )

    rviz_parameters = [
        moveit_config.planning_pipelines,
        moveit_config.robot_description_kinematics,
    ]

    add_debuggable_node(
        ld,
        package="rviz2",
        executable="rviz2",
        output="log",
        respawn=False,
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=rviz_parameters,
    )

    return ld

def generate_setup_assistant_launch(moveit_config):
    """Launch file for MoveIt Setup Assistant"""
    ld = LaunchDescription()

    ld.add_action(DeclareBooleanLaunchArg("debug", default_value=False))
    add_debuggable_node(
        ld,
        package="moveit_setup_assistant",
        executable="moveit_setup_assistant",
        arguments=[["--config_pkg=", str(moveit_config.package_path)]],
    )

    return ld

def generate_static_virtual_joint_tfs_launch(moveit_config):
    ld = LaunchDescription()

    name_counter = 0

    for key, xml_contents in moveit_config.robot_description_semantic.items():
        srdf = SRDF.from_xml_string(xml_contents)
        for vj in srdf.virtual_joints:
            ld.add_action(
                Node(
                    package="tf2_ros",
                    executable="static_transform_publisher",
                    name=f"static_transform_publisher{name_counter}",
                    output="log",
                    arguments=[
                        "--frame-id",
                        vj.parent_frame,
                        "--child-frame-id",
                        vj.child_link,
                    ],
                )
            )
            name_counter += 1
    return ld

def generate_spawn_controllers_launch(moveit_config):
    controller_names = moveit_config.trajectory_execution.get(
        "moveit_simple_controller_manager", {}
    ).get("controller_names", [])
    ld = LaunchDescription()
    for controller in controller_names + ["joint_state_broadcaster"]:
        ld.add_action(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller],
                output="screen",
            )
        )
    return ld

def generate_move_group_launch(moveit_config):
    ld = LaunchDescription()

    ld.add_action(DeclareBooleanLaunchArg("debug", default_value=False))
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

    add_debuggable_node(
        ld,
        package="moveit_ros_move_group",
        executable="move_group",
        commands_file=str(moveit_config.package_path / "launch" / "gdb_settings.gdb"),
        output="screen",
        parameters=move_group_params,
        extra_debug_args=["--debug"],
        # Set the display variable, in case OpenGL code is used internally
        additional_env={"DISPLAY": os.environ["DISPLAY"]},
    )
    return ld

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

    actions = [
        PushRosNamespace("articutool"),
        # Robot State Publisher
        *generate_rsp_launch(moveit_config).entities,
        # Move Group
        *generate_move_group_launch(moveit_config).entities,
        # RViz
        GroupAction(
            actions=generate_moveit_rviz_launch(moveit_config).entities,
            condition=IfCondition(launch_rviz),
        ),
        # Spawn Controllers
        GroupAction(
            actions=generate_spawn_controllers_launch(moveit_config).entities,
            condition=IfCondition(launch_controllers),
        ),
        # Static Virtual Joints
        *generate_static_virtual_joint_tfs_launch(moveit_config).entities,
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
    ]

    articutool_group = GroupAction(actions=actions)
    ld.add_action(articutool_group)

    return ld
