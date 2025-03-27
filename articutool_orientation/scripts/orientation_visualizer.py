# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3d
import numpy as np
import transforms3d.quaternions as quaternions


class Orientation3DVisualizer(Node):
    def __init__(self):
        super().__init__("orientation_3d_visualizer")
        self.subscription = self.create_subscription(
            Imu, "/articutool/estimated_orientation", self.listener_callback, 10
        )
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("3D Orientation")
        self.quaternion = [1.0, 0.0, 0.0, 0.0]  # Initial quaternion
        self.draw_frame()
        plt.show(block=False)  # Non-blocking show

    def listener_callback(self, msg):
        self.quaternion = [
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
        ]
        self.draw_frame()
        plt.pause(0.001)

    def draw_frame(self):
        self.ax.cla()
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("3D Orientation")

        # Create frame axes
        frame_axes = np.array(
            [
                [1, 0, 0],  # X-axis
                [0, 1, 0],  # Y-axis
                [0, 0, 1],  # Z-axis
            ]
        )

        # Rotate frame axes using quaternion
        rotated_axes = np.array(
            [quaternions.rotate_vector(axis, self.quaternion) for axis in frame_axes]
        )  # Corrected line.

        # Draw frame axes
        origin = [0, 0, 0]
        self.ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            rotated_axes[0, 0],
            rotated_axes[0, 1],
            rotated_axes[0, 2],
            color="r",
            length=0.5,
            arrow_length_ratio=0.1,
        )
        self.ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            rotated_axes[1, 0],
            rotated_axes[1, 1],
            rotated_axes[1, 2],
            color="g",
            length=0.5,
            arrow_length_ratio=0.1,
        )
        self.ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            rotated_axes[2, 0],
            rotated_axes[2, 1],
            rotated_axes[2, 2],
            color="b",
            length=0.5,
            arrow_length_ratio=0.1,
        )


def main(args=None):
    rclpy.init(args=args)
    visualizer = Orientation3DVisualizer()
    rclpy.spin(visualizer)
    visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
