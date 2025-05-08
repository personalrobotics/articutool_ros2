# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import QuaternionStamped 
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3d
import numpy as np
import transforms3d.quaternions as quaternions

class Orientation3DVisualizer(Node):
    def __init__(self):
        super().__init__("orientation_3d_visualizer")

        # Subscribe to the QuaternionStamped topic
        self.subscription = self.create_subscription(
            QuaternionStamped, # Changed message type here
            "/articutool/estimated_orientation", 
            self.listener_callback, 
            10
        )

        # --- Matplotlib Setup ---
        self.fig = plt.figure()
        # Make sure you have mpl_toolkits.mplot3d imported as a3d
        # self.ax = self.fig.add_subplot(111, projection="3d") # Standard way
        # Or potentially if axes_3d is needed explicitly (less common now):
        try:
            self.ax = self.fig.add_subplot(111, projection='3d')
        except ImportError:
             # Fallback or error if projection='3d' isn't directly supported this way
             # Might need Axes3D from mpl_toolkits.mplot3d depending on mpl version
             from mpl_toolkits.mplot3d import Axes3D # Ensure Axes3D is imported
             self.ax = Axes3D(self.fig) # Older way

        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("3D Orientation (IMU Frame)")

        # Initial quaternion [w, x, y, z] (identity)
        self.quaternion = [1.0, 0.0, 0.0, 0.0] 

        # Initial draw and non-blocking display
        self.draw_frame()
        plt.show(block=False) 

    # Change the type hint for msg
    def listener_callback(self, msg: QuaternionStamped): 
        # Access the quaternion field from QuaternionStamped message
        self.quaternion = [
            msg.quaternion.w,
            msg.quaternion.x,
            msg.quaternion.y,
            msg.quaternion.z,
        ]
        # No need to normalize here if the source already provides normalized quaternions
        # self.quaternion = normalize_quaternion(self.quaternion) # Optional safety check

        # Update the plot
        self.draw_frame()
        # Use plt.draw() and plt.pause() for non-blocking updates
        plt.draw() 
        plt.pause(0.001) # Small pause to allow plot to update

    def draw_frame(self):
        # Clear previous axes (keeps labels/title if not cleared fully)
        self.ax.cla() 

        # Re-apply limits and labels after clearing
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("3D Orientation (IMU Frame)")

        # Create standard basis vectors (representing the IMU frame's axes initially aligned with world)
        frame_axes = np.identity(3) # [[1,0,0], [0,1,0], [0,0,1]]

        # Rotate these basis vectors using the current quaternion
        # transforms3d expects quaternion as [w, x, y, z]
        try:
            rotated_axes = np.array(
                [quaternions.rotate_vector(axis, self.quaternion) for axis in frame_axes]
            )
        except Exception as e:
             self.get_logger().error(f"Error rotating vector: {e}. Quaternion: {self.quaternion}")
             return # Don't draw if rotation fails

        # Origin point
        origin = [0, 0, 0]

        # Draw the rotated axes using quiver
        # X-axis (Red)
        self.ax.quiver(
            origin[0], origin[1], origin[2], # Start point
            rotated_axes[0, 0], rotated_axes[0, 1], rotated_axes[0, 2], # Direction vector (rotated X)
            color="r", length=0.5, arrow_length_ratio=0.1,
        )
        # Y-axis (Green)
        self.ax.quiver(
            origin[0], origin[1], origin[2], # Start point
            rotated_axes[1, 0], rotated_axes[1, 1], rotated_axes[1, 2], # Direction vector (rotated Y)
            color="g", length=0.5, arrow_length_ratio=0.1,
        )
        # Z-axis (Blue)
        self.ax.quiver(
            origin[0], origin[1], origin[2], # Start point
            rotated_axes[2, 0], rotated_axes[2, 1], rotated_axes[2, 2], # Direction vector (rotated Z)
            color="b", length=0.5, arrow_length_ratio=0.1,
        )

        # Add labels for axes near the tips
        self.ax.text(rotated_axes[0, 0] * 0.55, rotated_axes[0, 1] * 0.55, rotated_axes[0, 2] * 0.55, "X", color="r")
        self.ax.text(rotated_axes[1, 0] * 0.55, rotated_axes[1, 1] * 0.55, rotated_axes[1, 2] * 0.55, "Y", color="g")
        self.ax.text(rotated_axes[2, 0] * 0.55, rotated_axes[2, 1] * 0.55, rotated_axes[2, 2] * 0.55, "Z", color="b")

        # Set aspect ratio to be equal
        self.ax.set_aspect('equal', adjustable='box')


def main(args=None):
    rclpy.init(args=args)
    visualizer = Orientation3DVisualizer()
    try:
        # Need to keep the script alive for plt.pause() to work
        while rclpy.ok():
             rclpy.spin_once(visualizer, timeout_sec=0.01)
             # Check if the figure is still open
             if not plt.fignum_exists(visualizer.fig.number):
                 visualizer.get_logger().info("Plot closed, shutting down.")
                 break
    except KeyboardInterrupt:
        visualizer.get_logger().info("Keyboard interrupt, shutting down.")
    except Exception as e:
        visualizer.get_logger().error(f"Unhandled exception: {e}", exc_info=True)
    finally:
        if visualizer.context.ok():
             visualizer.destroy_node()
        if rclpy.ok():
             rclpy.shutdown()
        plt.close('all') # Ensure plot is closed on exit

if __name__ == "__main__":
    main()
