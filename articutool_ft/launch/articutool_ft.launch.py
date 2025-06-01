# articutool_ft/launch/articutool_ft.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """
    Generates the launch description for the Articutool F/T sensor node.
    """

    # Declare launch arguments
    declare_serial_port_arg = DeclareLaunchArgument(
        "serial_port",
        default_value="/dev/resense_ft",
        description="Serial port for the Resense F/T sensor.",
    )

    declare_publish_rate_arg = DeclareLaunchArgument(
        "publish_rate_hz",
        default_value="100.0",
        description="Rate at which to read sensor and publish data.",
    )

    declare_wrench_topic_arg = DeclareLaunchArgument(
        "wrench_topic",
        default_value="ft_sensor/wrench_raw",
        description="Topic to publish WrenchStamped F/T data.",
    )

    declare_tare_service_arg = DeclareLaunchArgument(
        "tare_service_name",
        default_value="ft_sensor/tare",
        description="Service name to trigger sensor taring.",
    )

    declare_sensor_frame_id_arg = DeclareLaunchArgument(
        "sensor_frame_id",
        default_value="articutool_ft_sensor_link",
        description="TF frame ID for the WrenchStamped messages and sensor origin.",
    )

    # Node configuration
    resense_ft_node = Node(
        package="articutool_ft",
        executable="resense_ft_publisher",
        name="resense_ft_sensor_node",
        output="screen",
        parameters=[
            {
                "serial_port": LaunchConfiguration("serial_port"),
                "publish_rate_hz": LaunchConfiguration("publish_rate_hz"),
                "wrench_topic": LaunchConfiguration("wrench_topic"),
                "tare_service_name": LaunchConfiguration("tare_service_name"),
                "sensor_frame_id": LaunchConfiguration("sensor_frame_id"),
            }
        ],
    )

    return LaunchDescription(
        [
            declare_serial_port_arg,
            declare_publish_rate_arg,
            declare_wrench_topic_arg,
            declare_tare_service_arg,
            declare_sensor_frame_id_arg,
            resense_ft_node,
        ]
    )
