# articutool_control/primitives.py

import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any


class PrimitiveAction:
    """Abstract base class for all Articutool primitive actions."""

    def __init__(self, node_logger, params: List[float]):
        """
        Constructor for a primitive action.

        :param node_logger: The logger object from the calling ROS node.
        :param params: A list of float parameters from the action goal.
        """
        self.logger = node_logger
        self.params = params
        self._is_finished = False
        self._was_successful = False

    def start(self, current_joint_positions: np.ndarray) -> None:
        """
        Called once when the primitive action begins.
        Use this to initialize internal state based on the robot's current state.

        :param current_joint_positions: The current [pitch, roll] of the Articutool.
        """
        self._is_finished = False
        self._was_successful = False
        self.logger.info(
            f"Starting primitive: {self.__class__.__name__} with params: {self.params}"
        )

    def update(
        self, dt: float, current_joint_positions: np.ndarray
    ) -> Tuple[np.ndarray, str, float]:
        """
        Called on every tick of the controller loop. This is where the main logic lives.

        :param dt: Time delta since the last update.
        :param current_joint_positions: The current [pitch, roll] of the Articutool.
        :return: A tuple of (dq_command, feedback_string, percent_complete).
                 dq_command is a numpy array [d_pitch, d_roll].
        """
        raise NotImplementedError("Each primitive must implement its own update logic.")

    @property
    def is_finished(self) -> bool:
        """Returns True if the action has completed (succeeded or failed)."""
        return self._is_finished

    @property
    def was_successful(self) -> bool:
        """Returns True if the action finished successfully, False otherwise."""
        return self._was_successful


class TwirlPrimitive(PrimitiveAction):
    """Primitive for continuous rotation (twirling)."""

    def start(self, current_joint_positions: np.ndarray) -> None:
        super().start(current_joint_positions)
        if len(self.params) < 2:
            self.logger.error(
                f"[{self.__class__.__name__}] requires 2 parameters: [target_rotations, speed_rad_per_sec]. Got {len(self.params)}."
            )
            self._is_finished = True
            self._was_successful = False
            return

        self.target_total_delta_rad = self.params[0] * 2 * math.pi
        self.speed_rad_per_sec = abs(self.params[1])
        self.accumulated_rad = 0.0

    def update(
        self, dt: float, current_joint_positions: np.ndarray
    ) -> Tuple[np.ndarray, str, float]:
        dq_command = np.zeros(2)
        remaining_delta = self.target_total_delta_rad - self.accumulated_rad

        if abs(remaining_delta) < 1e-3:
            self._is_finished = True
            self._was_successful = True
            feedback_string = "Twirl completed."
            percent_complete = 1.0
        else:
            velocity = np.sign(remaining_delta) * self.speed_rad_per_sec
            if abs(velocity * dt) > abs(remaining_delta):
                velocity = remaining_delta / dt if dt > 1e-6 else 0.0

            dq_command[1] = velocity  # Roll is joint index 1
            self.accumulated_rad += dq_command[1] * dt

            percent_complete = (
                abs(self.accumulated_rad / self.target_total_delta_rad)
                if self.target_total_delta_rad != 0
                else 1.0
            )
            feedback_string = f"Twirling: Accum={self.accumulated_rad:.2f} / Target={self.target_total_delta_rad:.2f} rad"

        return dq_command, feedback_string, min(1.0, max(0.0, percent_complete))


class VibratePrimitive(PrimitiveAction):
    """Primitive for vibrating the roll joint."""

    def start(self, current_joint_positions: np.ndarray) -> None:
        super().start(current_joint_positions)
        if len(self.params) < 3:
            self.logger.error(
                f"[{self.__class__.__name__}] requires 3 parameters: [frequency_hz, amplitude_rad, duration_sec]."
            )
            self._is_finished = True
            self._was_successful = False
            return

        self.frequency_hz = self.params[0]
        self.amplitude_rad = self.params[1]
        self.duration_sec = self.params[2]
        self.time_elapsed_sec = 0.0

    def update(
        self, dt: float, current_joint_positions: np.ndarray
    ) -> Tuple[np.ndarray, str, float]:
        dq_command = np.zeros(2)
        self.time_elapsed_sec += dt

        if self.time_elapsed_sec >= self.duration_sec:
            self._is_finished = True
            self._was_successful = True
            feedback_string = "Vibration completed."
            percent_complete = 1.0
        else:
            current_phase = 2 * math.pi * self.frequency_hz * self.time_elapsed_sec
            dq_command[1] = (
                self.amplitude_rad
                * (2 * math.pi * self.frequency_hz)
                * math.cos(current_phase)
            )
            percent_complete = self.time_elapsed_sec / self.duration_sec
            feedback_string = (
                f"Vibrating: {self.time_elapsed_sec:.2f} / {self.duration_sec:.2f} sec"
            )

        return dq_command, feedback_string, min(1.0, max(0.0, percent_complete))


class DepositBitePrimitive(PrimitiveAction):
    """Primitive to deposit food with a timed roll and return."""

    def start(self, current_joint_positions: np.ndarray) -> None:
        super().start(current_joint_positions)
        # Params: [roll_angle_deg, roll_speed_rps, pause_sec]
        if len(self.params) < 3:
            self.logger.error(
                f"[{self.__class__.__name__}] requires 3 parameters: [roll_angle_deg, roll_speed_rps, pause_sec]."
            )
            self._is_finished = True
            self._was_successful = False
            return

        self.target_roll_angle_rad = math.radians(self.params[0])
        self.roll_speed_rps = self.params[1]
        self.pause_duration = self.params[2]

        self.state = "ROLLING_OUT"  # States: ROLLING_OUT, PAUSING, ROLLING_BACK
        self.accumulated_rad = 0.0
        self.pause_timer = 0.0

    def update(
        self, dt: float, current_joint_positions: np.ndarray
    ) -> Tuple[np.ndarray, str, float]:
        dq_command = np.zeros(2)
        feedback_string = ""
        percent_complete = 0.0

        if self.state == "ROLLING_OUT":
            feedback_string = "Depositing..."
            remaining_delta = self.target_roll_angle_rad - self.accumulated_rad
            if abs(remaining_delta) < 0.05:  # Small tolerance
                self.state = "PAUSING"
                self.accumulated_rad = 0.0  # Reset for return trip
            else:
                velocity = np.sign(remaining_delta) * self.roll_speed_rps
                dq_command[1] = velocity
                self.accumulated_rad += velocity * dt
            percent_complete = (
                abs(self.accumulated_rad / self.target_roll_angle_rad) / 3.0
            )

        elif self.state == "PAUSING":
            feedback_string = "Pausing during deposit..."
            self.pause_timer += dt
            if self.pause_timer >= self.pause_duration:
                self.state = "ROLLING_BACK"
            percent_complete = 1 / 3 + (self.pause_timer / self.pause_duration) / 3.0

        elif self.state == "ROLLING_BACK":
            feedback_string = "Returning from deposit..."
            # Target is now back to 0
            # current_joint_positions[1] is the roll joint
            remaining_delta = 0 - current_joint_positions[1]
            if abs(remaining_delta) < 0.05:
                self._is_finished = True
                self._was_successful = True
            else:
                velocity = np.sign(remaining_delta) * self.roll_speed_rps
                dq_command[1] = velocity
            percent_complete = (
                2 / 3 + (1.0 - abs(remaining_delta / self.target_roll_angle_rad)) / 3.0
            )

        return dq_command, feedback_string, min(1.0, max(0.0, percent_complete))


class SettleTossPrimitive(PrimitiveAction):
    """
    Primitive to settle food in a spoon with a controlled pitch-flick motion.
    This is designed to be more effective than simple vibration.
    """

    def start(self, current_joint_positions: np.ndarray) -> None:
        super().start(current_joint_positions)
        # Params: [tilt_angle_deg, tilt_speed_rps, flick_speed_rps]
        if len(self.params) < 3:
            self.logger.error(
                f"[{self.__class__.__name__}] requires 3 parameters: "
                "[tilt_angle_deg, tilt_speed_rps, flick_speed_rps]."
            )
            self._is_finished = True
            self._was_successful = False
            return

        self.tilt_angle_rad = math.radians(self.params[0])
        self.tilt_speed_rps = abs(self.params[1])
        self.flick_speed_rps = abs(self.params[2])

        # Store the starting position to return to it
        self.start_pitch_rad = current_joint_positions[0]

        # State machine for the motion
        self.state = "TILT_BACK"  # States: TILT_BACK, FLICK_FORWARD, RETURN_TO_LEVEL
        self.tolerance = 0.05  # Radians

    def update(
        self, dt: float, current_joint_positions: np.ndarray
    ) -> Tuple[np.ndarray, str, float]:
        dq_command = np.zeros(2)
        feedback_string = ""
        percent_complete = 0.0
        current_pitch = current_joint_positions[0]

        if self.state == "TILT_BACK":
            feedback_string = "Settling: Tilting back..."
            target_pitch = self.start_pitch_rad - self.tilt_angle_rad
            error = target_pitch - current_pitch

            if abs(error) < self.tolerance:
                self.state = "FLICK_FORWARD"
            else:
                velocity = -self.tilt_speed_rps  # Move in negative direction
                dq_command[0] = velocity
            percent_complete = (
                abs(current_pitch - self.start_pitch_rad)
                / abs(self.tilt_angle_rad)
                / 3.0
            )

        elif self.state == "FLICK_FORWARD":
            feedback_string = "Settling: Flicking forward..."
            # Target is slightly past the start position to create a jolt
            target_pitch = self.start_pitch_rad + self.tolerance * 2
            error = target_pitch - current_pitch

            if error < 0:  # We have passed the target
                self.state = "RETURN_TO_LEVEL"
            else:
                velocity = self.flick_speed_rps  # Move in positive direction
                dq_command[0] = velocity
            percent_complete = (
                1 / 3
                + abs(current_pitch - (self.start_pitch_rad - self.tilt_angle_rad))
                / abs(self.tilt_angle_rad)
                / 3.0
            )

        elif self.state == "RETURN_TO_LEVEL":
            feedback_string = "Settling: Returning to start..."
            target_pitch = self.start_pitch_rad
            error = target_pitch - current_pitch

            if abs(error) < self.tolerance:
                self._is_finished = True
                self._was_successful = True
                dq_command[0] = 0.0  # Ensure we stop
            else:
                # Return slowly and smoothly
                velocity = np.sign(error) * self.tilt_speed_rps
                dq_command[0] = velocity
            percent_complete = 2 / 3 + (1.0 - abs(error / self.tilt_angle_rad)) / 3.0

        return dq_command, feedback_string, min(1.0, max(0.0, percent_complete))
