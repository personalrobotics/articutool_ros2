#!/usr/bin/env python3

# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
ROS 2 Node for reading data from a Resense F/T sensor (real mode)
or publishing dummy data (mock/dummy mode), performing software taring,
and publishing as WrenchStamped messages.
It also provides a service (SetBool) to re-trigger the taring procedure.
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy.duration

import serial
import struct
import time
import numpy as np
import sys
import threading
from typing import Optional

from geometry_msgs.msg import WrenchStamped, Vector3 as Vector3Msg
from std_srvs.srv import SetBool


class ResenseFtNode(Node):
    DEFAULT_SERIAL_PORT = "/dev/resense_ft"
    BAUD_RATE = 2000000
    DATA_PACKET_SIZE = 28
    NUM_VALUES = 7
    STRUCT_FORMAT = "<fffffff"
    SAMPLES_FOR_TARE = 100

    def __init__(self):
        super().__init__("resense_ft_node")

        # --- Core Parameters ---
        self.declare_parameter(
            "sim",
            "real",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Operation mode: 'real' for hardware sensor, 'mock' for dummy data.",
            ),
        )
        self.sim = self.get_parameter("sim").get_parameter_value().string_value
        if self.sim not in ["real", "mock"]:
            self.get_logger().warn(
                f"Invalid sim '{self.sim}'. Defaulting to 'real'. "
                "Valid options are 'real' or 'mock'."
            )
            self.sim = "real"

        self.declare_parameter(
            "publish_rate_hz",
            100.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Rate at which to publish data (real or mock).",
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

        # --- Parameters for Real Sensor Mode ---
        self.declare_parameter(
            "serial_port",
            self.DEFAULT_SERIAL_PORT,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Serial port for the Resense F/T sensor (real mode only).",
            ),
        )

        # --- Parameters for Mock Sensor Mode ---
        self.declare_parameter(
            "mock_mean",
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="Mean of the F/T data (mock mode only).",
            ),
        )
        self.declare_parameter(
            "mock_std",
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="Std dev of the F/T data (mock mode only).",
            ),
        )
        self.declare_parameter(
            "mock_is_on",
            True,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="Simulates if the sensor is 'on' (publishing) (mock mode only).",
            ),
        )

        # Get parameters
        self.publish_rate_hz = (
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

        if self.publish_rate_hz <= 0:
            self.get_logger().warn(
                "Publish rate must be positive. Setting to default 100.0 Hz."
            )
            self.publish_rate_hz = 100.0

        self.get_logger().info(
            f"Resense F/T Node Operating in '{self.sim.upper()}' mode."
        )
        self.get_logger().info(f"  Publish Rate: {self.publish_rate_hz} Hz")
        self.get_logger().info(f"  Wrench Topic: {self.wrench_topic_name}")
        self.get_logger().info(
            f"  Tare Service ({tare_service_name}) Type: std_srvs/SetBool"
        )
        self.get_logger().info(f"  Sensor Frame ID: {self.sensor_frame_id}")

        self.force_offsets = np.array([0.0, 0.0, 0.0])
        self.torque_offsets = np.array([0.0, 0.0, 0.0])
        self.is_tared = False
        self.tare_in_progress = False

        self.mock_set_bias_request_time: Optional[rclpy.time.Time] = None
        self.mock_messages_since_bias_request = 0
        self.mock_set_bias_request_time_lock = threading.Lock()

        self.serial_port_name = (
            self.get_parameter("serial_port").get_parameter_value().string_value
        )
        self.serial_connection: Optional[serial.Serial] = None

        if self.sim == "real":
            self.get_logger().info(
                f"  Real Sensor Serial Port: {self.serial_port_name}"
            )
            if self._connect_sensor():
                self._perform_tare_routine()
        elif self.sim == "mock":
            self.mock_mean_val = (
                self.get_parameter("mock_mean").get_parameter_value().double_array_value
            )
            self.mock_std_val = (
                self.get_parameter("mock_std").get_parameter_value().double_array_value
            )
            # self.mock_is_on_val is read dynamically in publish loop
            self.get_logger().info(f"  Mock Mode Mean: {self.mock_mean_val}")
            self.get_logger().info(f"  Mock Mode Std: {self.mock_std_val}")
            # Initial dummy_is_on state logged when read_and_publish_data first runs in mock mode

        self.wrench_publisher = self.create_publisher(
            WrenchStamped, self.wrench_topic_name, 10
        )
        self.tare_service = self.create_service(
            SetBool, tare_service_name, self.handle_tare_request
        )
        self.read_publish_timer = self.create_timer(
            1.0 / self.publish_rate_hz, self.read_and_publish_data
        )
        self.get_logger().info(f"{self.get_name()} initialized.")

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
            self.get_logger().error("Cannot perform real tare: Sensor not connected.")
            return False
        if self.tare_in_progress:
            self.get_logger().warn("Real tare already in progress.")
            return False

        self.tare_in_progress = True
        self.is_tared = False
        self.get_logger().info(
            "Initiating REAL sensor software tare: Please ensure sensor is UNLOADED."
        )
        self.get_logger().info(f"Collecting {self.SAMPLES_FOR_TARE} samples...")
        temp_forces, temp_torques, samples_collected = [], [], 0
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
                    f"Real tare timed out. Collected {samples_collected}/{self.SAMPLES_FOR_TARE} samples."
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
                            f"  Real Tare sample {samples_collected}/{self.SAMPLES_FOR_TARE}",
                            end="\r",
                        )
                        sys.stdout.flush()
                    except struct.error:
                        self.get_logger().warn(
                            "Struct error during real tare sample collection, skipping."
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
            self.get_logger().info("REAL sensor software taring complete.")
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
            self.get_logger().error("Failed to collect enough samples for real taring.")
            self.tare_in_progress = False
            return False

    def handle_tare_request(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        self.get_logger().info(f"Tare service called with data: {request.data}")
        if self.sim == "mock":
            if request.data:
                with self.mock_set_bias_request_time_lock:
                    self.mock_set_bias_request_time = self.get_clock().now()
                    self.mock_messages_since_bias_request = 0
                response.success = True
                response.message = "Mock tare initiated (simulating delay)."
                self.get_logger().info(response.message)
            else:
                response.success = True
                response.message = "Mock tare request with data=false received."
                self.get_logger().info(response.message)
        elif self.sim == "real":
            if request.data:
                if self._perform_tare_routine():
                    response.success = True
                    response.message = "Real sensor taring successful."
                else:
                    response.success = False
                    response.message = "Real sensor taring failed. Check logs."
            else:
                response.success = True
                response.message = (
                    "Real sensor tare request with data=false. No action taken."
                )
            self.get_logger().info(f"Tare service response: {response.message}")
        else:  # Should not happen due to init check
            response.success = False
            response.message = f"Unknown sim: {self.sim}"
            self.get_logger().error(response.message)
        return response

    def read_and_publish_data(self):
        if self.sim == "mock":
            mock_is_on = (
                self.get_parameter("mock_is_on").get_parameter_value().bool_value
            )
            if not mock_is_on:
                return

            with self.mock_set_bias_request_time_lock:
                if self.mock_set_bias_request_time is not None and (
                    self.get_clock().now() - self.mock_set_bias_request_time
                ) < rclpy.duration.Duration(seconds=0.75):
                    self.mock_messages_since_bias_request += 1
                    if (self.mock_messages_since_bias_request % 10) != 0:
                        return
                else:
                    if self.mock_set_bias_request_time is not None:
                        self.get_logger().info("Mock tare delay period finished.")
                    self.mock_set_bias_request_time = None

            current_mock_mean = (
                self.get_parameter("mock_mean").get_parameter_value().double_array_value
            )
            current_mock_std = (
                self.get_parameter("mock_std").get_parameter_value().double_array_value
            )

            ft_data = np.random.normal(current_mock_mean, current_mock_std)

            wrench_msg = WrenchStamped()
            wrench_msg.header.stamp = self.get_clock().now().to_msg()
            wrench_msg.header.frame_id = self.sensor_frame_id
            wrench_msg.wrench.force.x = float(ft_data[0])
            wrench_msg.wrench.force.y = float(ft_data[1])
            wrench_msg.wrench.force.z = float(ft_data[2])
            wrench_msg.wrench.torque.x = float(ft_data[3])
            wrench_msg.wrench.torque.y = float(ft_data[4])
            wrench_msg.wrench.torque.z = float(ft_data[5])
            self.wrench_publisher.publish(wrench_msg)

        elif self.sim == "real":
            if not self.serial_connection or not self.serial_connection.is_open:
                if not self._connect_sensor():
                    self.get_logger().warn(
                        "Sensor not connected. Cannot read data.",
                        throttle_duration_sec=5.0,
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
                        unpacked_data = struct.unpack(
                            self.STRUCT_FORMAT, raw_data_bytes
                        )
                        fx_raw, fy_raw, fz_raw, mx_raw, my_raw, mz_raw, _ = (
                            unpacked_data
                        )
                        self.get_logger().debug(
                            f"Raw FT: Fx={fx_raw:.3f} Fy={fy_raw:.3f} Fz={fz_raw:.3f} Mx={mx_raw:.3f} My={my_raw:.3f} Mz={mz_raw:.3f}"
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
                            f"Unexpected error in read_and_publish_data: {e}",
                        )

    def destroy_node(self):
        self.get_logger().info(f"Shutting down {self.get_name()} node.")
        if hasattr(self, "read_publish_timer") and self.read_publish_timer:
            self.read_publish_timer.cancel()
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ResenseFtNode()
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except KeyboardInterrupt:
        if node:
            node.get_logger().info(
                f"{node.get_name()} shutting down due to KeyboardInterrupt."
            )
    except Exception as e:
        temp_logger = rclpy.logging.get_logger("resense_ft_node_main_exception")
        temp_logger.fatal(f"Unhandled exception in main: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
