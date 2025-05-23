# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

from setuptools import setup
import os
from glob import glob

package_name = "articutool_system"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
        (
            os.path.join("share", package_name, "scripts"),
            glob(os.path.join("scripts", "*.py")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="charles",
    maintainer_email="charles@todo.todo",
    description="System level launch files for the Articutool",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "orientation_control = articutool_system.orientation_control:main",
            "articutool_keyboard_teleop = scripts.articutool_keyboard_teleop:main",
            "joint_state_republisher = articutool_system.joint_state_republisher:main",
        ],
    },
)
