from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
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

    return LaunchDescription(
        [
            imu_launch,
            orientation_launch,
        ]
    )
