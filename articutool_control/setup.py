from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'articutool_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='regulus',
    maintainer_email='jose33jaime@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'send_trajectory_action_client = articutool_control.send_trajectory_action_client:main',
        ],
    },
)
