#!/usr/bin/env python3

# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
ROS 2 Node for reading data from a Resense F/T sensor,
performing software taring, and publishing as WrenchStamped messages.
It also provides a service (SetBool) to re-trigger the taring procedure.
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

import serial
import struct
import time
import numpy as np
import sys  # For stdout in tare progress

from geometry_msgs.msg import WrenchStamped, Vector3 as Vector3Msg
from std_srvs.srv import SetBool


class ResenseFtNode(Node):
    """
    ROS 2 Node to interface with the Resense F/T sensor.
    """

    DEFAULT_SERIAL_PORT = "/dev/resense_ft"
    BAUD_RATE = 2000000
    DATA_PACKET_SIZE = 28  # 7 floats * 4 bytes/float
    NUM_VALUES = 7
    STRUCT_FORMAT = "<fffffff"  # Little-endian for 7 floats
    SAMPLES_FOR_TARE = 100

    def __init__(self):
        super().__init__("resense_ft_publisher_node")

        self.declare_parameter(
            "serial_port",
            self.DEFAULT_SERIAL_PORT,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Serial port for the Resense F/T sensor.",
            ),
        )
        self.declare_parameter(
            "publish_rate_hz",
            100.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Rate at which to read sensor and publish data.",
            ),
        )
        self.declare_parameter(
            "wrench_topic",
            "~/ft_data",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic to publish WrenchStamped F/T data.",
            ),
        )
        self.declare_parameter(
            "tare_service_name",
            "~/tare_sensor",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Service name to trigger sensor taring (SetBool type).",
            ),
        )
        self.declare_parameter(
            "sensor_frame_id",
            "resense_ft_sensor_link",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="TF frame ID for the WrenchStamped messages.",
            ),
        )

        self.serial_port_name = (
            self.get_parameter("serial_port").get_parameter_value().string_value
        )
        publish_rate_hz = (
            self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        )
        self.wrench_topic_name = (
            self.get_parameter("wrench_topic").get_parameter_value().string_value
        )
        tare_service_name = (
            self.get_parameter("tare_service_name").get_parameter_value().string_value
        )
        self.sensor_frame_id = (
            self.get_parameter("sensor_frame_id").get_parameter_value().string_value
        )

        if publish_rate_hz <= 0:
            self.get_logger().warn(
                "Publish rate must be positive. Setting to default 100.0 Hz."
            )
            publish_rate_hz = 100.0

        self.get_logger().info(f"Resense F/T Node Configuration:")
        self.get_logger().info(f"  Serial Port: {self.serial_port_name}")
        self.get_logger().info(f"  Publish Rate: {publish_rate_hz} Hz")
        self.get_logger().info(f"  Wrench Topic: {self.wrench_topic_name}")
        self.get_logger().info(
            f"  Tare Service ({tare_service_name}) Type: std_srvs/SetBool"
        )
        self.get_logger().info(f"  Sensor Frame ID: {self.sensor_frame_id}")

        self.force_offsets = np.array([0.0, 0.0, 0.0])
        self.torque_offsets = np.array([0.0, 0.0, 0.0])
        self.is_tared = False
        self.tare_in_progress = False
        self.serial_connection: Optional[serial.Serial] = None

        self.wrench_publisher = self.create_publisher(
            WrenchStamped, self.wrench_topic_name, 10
        )
        self.tare_service = self.create_service(
            SetBool,
            tare_service_name,
            self.handle_tare_request,
        )

        if self._connect_sensor():
            self._perform_tare_routine()

        self.read_publish_timer = self.create_timer(
            1.0 / publish_rate_hz, self.read_and_publish_data
        )
        self.get_logger().info("Resense F/T sensor node started.")

    def _connect_sensor(self) -> bool:
        if self.serial_connection and self.serial_connection.is_open:
            self.get_logger().info("Sensor already connected.")
            return True
        try:
            self.serial_connection = serial.Serial(
                port=self.serial_port_name,
                baudrate=self.BAUD_RATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
            )
            self.get_logger().info(
                f"Successfully connected to sensor on {self.serial_port_name}."
            )
            return True
        except serial.SerialException as e:
            self.get_logger().error(
                f"Failed to connect to sensor on {self.serial_port_name}: {e}"
            )
            self.serial_connection = None
            return False

    def _perform_tare_routine(self) -> bool:
        if not self.serial_connection or not self.serial_connection.is_open:
            self.get_logger().error("Cannot perform tare: Sensor not connected.")
            return False
        if self.tare_in_progress:
            self.get_logger().warn("Tare already in progress. Ignoring request.")
            return False

        self.tare_in_progress = True
        self.is_tared = False

        self.get_logger().info(
            "Initiating software tare: Please ensure sensor is UNLOADED."
        )
        self.get_logger().info(f"Collecting {self.SAMPLES_FOR_TARE} samples...")

        temp_forces = []
        temp_torques = []
        samples_collected = 0

        if self.serial_connection:
            self.serial_connection.reset_input_buffer()
        else:
            self.get_logger().error("Serial connection lost before taring.")
            self.tare_in_progress = False
            return False

        time.sleep(0.2)

        start_tare_time = self.get_clock().now()
        max_tare_duration = rclpy.duration.Duration(seconds=5.0)

        while samples_collected < self.SAMPLES_FOR_TARE:
            if (self.get_clock().now() - start_tare_time) > max_tare_duration:
                self.get_logger().error(
                    f"Tare timed out. Collected {samples_collected}/{self.SAMPLES_FOR_TARE} samples."
                )
                self.tare_in_progress = False
                return False

            if (
                self.serial_connection
                and self.serial_connection.in_waiting >= self.DATA_PACKET_SIZE
            ):
                raw_data = self.serial_connection.read(self.DATA_PACKET_SIZE)
                if len(raw_data) == self.DATA_PACKET_SIZE:
                    try:
                        unpacked_data = struct.unpack(self.STRUCT_FORMAT, raw_data)
                        fx, fy, fz, mx, my, mz, _ = unpacked_data
                        temp_forces.append([fx, fy, fz])
                        temp_torques.append([mx, my, mz])
                        samples_collected += 1
                        print(
                            f"  Tare sample {samples_collected}/{self.SAMPLES_FOR_TARE}",
                            end="\r",
                        )
                        sys.stdout.flush()
                    except struct.error:
                        self.get_logger().warn(
                            "Struct error during tare sample collection, skipping."
                        )
                        if self.serial_connection:
                            self.serial_connection.reset_input_buffer()
            else:
                time.sleep(0.001)

        if samples_collected == self.SAMPLES_FOR_TARE:
            self.force_offsets = np.mean(np.array(temp_forces), axis=0)
            self.torque_offsets = np.mean(np.array(temp_torques), axis=0)
            self.is_tared = True
            print("\n" + "=" * 30)
            self.get_logger().info("Software taring complete.")
            self.get_logger().info(
                f"  Force Offsets (Fx,Fy,Fz)[N]: {np.round(self.force_offsets, 4).tolist()}"
            )
            self.get_logger().info(
                f"  Torque Offsets (Mx,My,Mz)[mNm]: {np.round(self.torque_offsets, 4).tolist()}"
            )
            print("=" * 30 + "\n")
            self.tare_in_progress = False
            return True
        else:
            self.get_logger().error(
                "Failed to collect enough samples for taring (should have timed out)."
            )
            self.tare_in_progress = False
            return False

    def handle_tare_request(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        """Handles requests to the tare service."""
        self.get_logger().info(f"Tare service called with data: {request.data}")

        if request.data:
            if self._perform_tare_routine():
                response.success = True
                response.message = "Sensor taring successful."
                self.get_logger().info("Tare service: Succeeded.")
            else:
                response.success = False
                response.message = "Sensor taring failed during routine. Check logs."
                self.get_logger().error("Tare service: Failed.")
        else:  # request.data is false
            response.success = (
                True  # Or False depending on desired behavior for "data: false"
            )
            response.message = "Tare request with data=false received. No action taken."
            self.get_logger().info(response.message)
            # If data:false should clear tare or have another meaning, implement here.
            # For now, it's a no-op but reports success for the service call itself.

        return response

    def read_and_publish_data(self):
        if not self.serial_connection or not self.serial_connection.is_open:
            if not self._connect_sensor():
                self.get_logger().warn(
                    "Sensor not connected. Cannot read data.", throttle_duration_sec=5.0
                )
                return
            else:
                if not self.is_tared and not self.tare_in_progress:
                    self.get_logger().info(
                        "Sensor reconnected, attempting initial tare."
                    )
                    self._perform_tare_routine()

        if (
            self.serial_connection
            and self.serial_connection.in_waiting >= self.DATA_PACKET_SIZE
        ):
            raw_data_bytes = self.serial_connection.read(self.DATA_PACKET_SIZE)
            if len(raw_data_bytes) == self.DATA_PACKET_SIZE:
                try:
                    unpacked_data = struct.unpack(self.STRUCT_FORMAT, raw_data_bytes)
                    fx_raw, fy_raw, fz_raw, mx_raw, my_raw, mz_raw, temp_raw = (
                        unpacked_data
                    )

                    self.get_logger().debug(
                        f"Raw Unpacked FT: Fx={fx_raw:.3f}, Fy={fy_raw:.3f}, Fz={fz_raw:.3f}, "
                        f"Mx={mx_raw:.3f}, My={my_raw:.3f}, Mz={mz_raw:.3f}, Temp={temp_raw:.1f}"
                    )

                    fx_tared, fy_tared, fz_tared = fx_raw, fy_raw, fz_raw
                    mx_tared, my_tared, mz_tared = mx_raw, my_raw, mz_raw

                    if self.is_tared:
                        fx_tared -= self.force_offsets[0]
                        fy_tared -= self.force_offsets[1]
                        fz_tared -= self.force_offsets[2]
                        mx_tared -= self.torque_offsets[0]
                        my_tared -= self.torque_offsets[1]
                        mz_tared -= self.torque_offsets[2]

                    wrench_msg = WrenchStamped()
                    wrench_msg.header.stamp = self.get_clock().now().to_msg()
                    wrench_msg.header.frame_id = self.sensor_frame_id
                    wrench_msg.wrench.force.x = float(fx_tared)
                    wrench_msg.wrench.force.y = float(fy_tared)
                    wrench_msg.wrench.force.z = float(fz_tared)
                    wrench_msg.wrench.torque.x = float(mx_tared) / 1000.0
                    wrench_msg.wrench.torque.y = float(my_tared) / 1000.0
                    wrench_msg.wrench.torque.z = float(mz_tared) / 1000.0
                    self.wrench_publisher.publish(wrench_msg)

                except struct.error as e:
                    self.get_logger().warn(
                        f"Struct unpacking error: {e}. Data: {raw_data_bytes.hex()}",
                        throttle_duration_sec=1.0,
                    )
                    if self.serial_connection:
                        self.serial_connection.reset_input_buffer()
                except Exception as e:
                    self.get_logger().error(
                        f"Unexpected error in read_and_publish_data: {e}", exc_info=True
                    )
            # else: self.get_logger().warn("Incomplete packet read.", throttle_duration_sec=5.0)
        # else: self.get_logger().debug("No full packet available for reading.", throttle_duration_sec=1.0)

    def destroy_node(self):
        self.get_logger().info("Shutting down Resense F/T sensor node.")
        if (
            hasattr(self, "read_publish_timer") and self.read_publish_timer
        ):  # Check if timer exists
            self.read_publish_timer.cancel()
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ResenseFtNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info(
                f"{node.get_name()} shutting down due to KeyboardInterrupt."
            )
    except Exception as e:
        temp_logger = rclpy.logging.get_logger("resense_ft_node_main_exception")
        temp_logger.fatal(f"Unhandled exception in main: {e}", exc_info=True)
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
