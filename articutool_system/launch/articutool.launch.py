# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare imu_port launch argument
    imu_port_arg = DeclareLaunchArgument(
        "imu_port",
        default_value="/dev/ttyUSB1",
        description="USB port for the IMU",
    )

    imu_port = LaunchConfiguration("imu_port")

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

    # Include articutool_description launch file
    description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("articutool_description"),
                    "launch",
                    "articutool_description.launch.py",
                )
            ]
        ),
    )

    return LaunchDescription(
        [
            imu_port_arg,
            imu_launch,
            orientation_launch,
            description_launch,
        ]
    )
