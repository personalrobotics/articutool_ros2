import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('articutool_orientation'),
        'config',
        'orientation_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='articutool_orientation',
            executable='orientation_estimator',
            name='orientation_estimator',
            parameters=[config],
            output='screen'
        ),
        Node(
            package='articutool_orientation',
            executable='ik_solver',
            name='ik_solver',
            parameters=[config],
            output='screen'
        )
    ])
