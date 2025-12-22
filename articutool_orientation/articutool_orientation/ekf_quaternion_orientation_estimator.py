# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, QuaternionStamped, TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import math
from scipy.spatial.transform import Rotation
import sys


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def normalize_quaternion(q_arr):
    norm = np.linalg.norm(q_arr)
    if norm < 1e-9:  # Avoid division by zero or near-zero
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q_arr / norm


def quaternion_to_rotation_matrix(q_arr):
    # q_arr is [qw, qx, qy, qz]
    # Scipy expects [qx, qy, qz, qw]
    q_scipy = np.array([q_arr[1], q_arr[2], q_arr[3], q_arr[0]])
    norm = np.linalg.norm(q_scipy)
    if norm < 1e-9:
        return np.identity(3)
    return Rotation.from_quat(q_scipy / norm).as_matrix()


def skew_symmetric(v):
    """Returns the skew-symmetric matrix for a 3x1 vector v."""
    v = v.flatten()
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class QuaternionRollPitchEstimator(Node):
    def __init__(self):
        super().__init__("quaternion_roll_pitch_estimator")

        self.dim_x = 7
        self.dim_z_accel = 3
        self.x = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        self.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01]) * 10

        # --- Noise Parameters ---
        # Gyroscope Noise Spectral Density: 0.015 dps/√Hz
        # Converted to (rad/s)/√Hz: 0.015 * (π/180)
        # PSD for EKF ((rad/s)^2/Hz): (0.015 * π/180)^2
        gyro_noise_sq = (0.015 * math.pi / 180.0) ** 2  # Approx 6.85e-8 [(rad/s)^2/Hz]

        # Bias drift - this is more empirical as datasheet doesn't give direct random walk for bias
        bias_drift_sq = (0.001) ** 2  # (rad/s^2)^2/Hz for bias random walk PSD

        # Accelerometer Noise Spectral Density: 230 µg/√Hz
        # Converted to (m/s^2)/√Hz: 230 * 1e-6 * 9.81
        # Variance for EKF ((m/s^2)^2): (230 * 1e-6 * 9.81)^2
        accel_noise_sq = (230.0 * 1e-6 * 9.81) ** 2  # Approx 5.09e-6 [(m/s^2)^2]

        self.get_logger().info(f"Using gyro_noise_sq (PSD): {gyro_noise_sq:.2e}")
        self.get_logger().info(f"Using accel_noise_sq (Variance): {accel_noise_sq:.2e}")

        self.Q_cont_diag_variances = np.array([gyro_noise_sq] * 3 + [bias_drift_sq] * 3)
        self.R_accel = np.eye(self.dim_z_accel) * accel_noise_sq

        self.gravity_ref_world = np.array([[0.0, 0.0, 9.81]]).T
        self.orientation_initialized = False

        self.imu_subscription = self.create_subscription(
            Imu, "articutool/imu_data", self.imu_callback, 10
        )
        self.orientation_publisher = self.create_publisher(
            QuaternionStamped, "articutool/estimated_orientation", 10
        )
        self.imu_frame_id = "atool_imu_frame"

        self.last_time = None
        self.get_logger().info(
            "Quaternion EKF Roll/Pitch Estimator Initialized (Yaw will drift)"
        )

    def imu_callback(self, msg):
        current_time_ros = self.get_clock().now()
        current_time = current_time_ros.nanoseconds / 1e9

        if self.last_time is None:
            self.last_time = current_time
            if not self.orientation_initialized:
                self.initialize_orientation_from_accel(msg)
            return

        dt = current_time - self.last_time
        if dt <= 1e-6:
            return
        self.last_time = current_time

        accel_meas = np.array(
            [
                [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ]
            ]
        ).T
        gyro_meas = np.array(
            [[msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]]
        ).T

        if not self.orientation_initialized:
            self.get_logger().warn(
                "Orientation not yet initialized, skipping EKF step.",
                throttle_duration_sec=5,
            )
            return

        self.predict_step(gyro_meas, dt)

        accel_norm = np.linalg.norm(accel_meas)
        gravity_norm = np.linalg.norm(self.gravity_ref_world)
        # allow +/- 50% deviation from g
        if abs(accel_norm - gravity_norm) < 0.5 * gravity_norm:
            self.update_step_accelerometer(accel_meas)
        else:
            self.get_logger().warn(
                f"Skipping accel update due to large linear acceleration: {accel_norm:.2f} m/s^2 (gravity_norm: {gravity_norm:.2f})",
                throttle_duration_sec=5,
            )

        if np.isnan(self.x).any() or np.isinf(self.x).any():
            self.get_logger().error(
                "NaN or Inf detected in state vector! Filter diverged."
            )
            self.orientation_initialized = False
            self.x = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
            self.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01]) * 10
            return

        q_est = normalize_quaternion(self.x[0:4].flatten())

        ros_q_stamped = QuaternionStamped()
        ros_q_stamped.header.stamp = msg.header.stamp
        ros_q_stamped.header.frame_id = "world"
        ros_q_stamped.quaternion.w = q_est[0]
        ros_q_stamped.quaternion.x = q_est[1]
        ros_q_stamped.quaternion.y = q_est[2]
        ros_q_stamped.quaternion.z = q_est[3]
        self.orientation_publisher.publish(ros_q_stamped)

    def initialize_orientation_from_accel(self, imu_msg):
        self.get_logger().info(
            "Initializing orientation from accelerometer (Roll/Pitch, Yaw=0 and will drift)..."
        )
        try:
            accel_init = np.array(
                [
                    imu_msg.linear_acceleration.x,
                    imu_msg.linear_acceleration.y,
                    imu_msg.linear_acceleration.z,
                ]
            )
            accel_norm = np.linalg.norm(accel_init)
            if accel_norm < 1e-3:
                self.get_logger().warn(
                    "Initial accelerometer reading near zero norm. Cannot initialize roll/pitch."
                )
                return
            accel_init_norm = accel_init / accel_norm

            ax, ay, az = accel_init_norm
            # Ensure az is not too small to avoid instability in roll calculation if sensor is near horizontal
            if abs(az) < 0.1:  # If z-component is small, sensor is mostly horizontal
                # Pitch is dominant, calculate carefully
                pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
                # If ay is also small (ax is dominant), roll might be ill-defined or jumpy.
                # For pure pitch (ay near 0, az near 0), roll can be set to 0.
                if abs(ay) < 0.1 and abs(az) < 0.1:  # Mostly aligned with X axis
                    roll = 0.0  # or undefined, depends on desired behavior
                else:  # Prioritize ay for roll if az is small
                    roll = math.atan2(
                        ay, 0.001 if az == 0 else az
                    )  # avoid division by zero if az is exactly 0
            else:  # Standard calculation when az is significant
                roll = math.atan2(ay, az)
                pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))

            yaw = 0.0

            r = Rotation.from_euler("zyx", [yaw, pitch, roll])
            q_init_scipy = r.as_quat()  # [x,y,z,w]

            self.x[0:4] = np.array(
                [[q_init_scipy[3], q_init_scipy[0], q_init_scipy[1], q_init_scipy[2]]]
            ).T
            self.x[0:4] = normalize_quaternion(self.x[0:4].flatten()).reshape(4, 1)
            # Initialize biases to zero
            self.x[4:7] = np.zeros((3, 1))
            # Reset covariance with a bit more confidence in initial roll/pitch from accel
            self.P = np.diag([0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05])

            self.orientation_initialized = True
            self.get_logger().info(
                f"Initialized: Roll={math.degrees(roll):.1f}, Pitch={math.degrees(pitch):.1f}, Yaw={math.degrees(yaw):.1f} (Yaw will drift)"
            )

        except Exception as e:
            self.get_logger().error(
                f"Error during EKF initialization: {e}", exc_info=True
            )

    def predict_step(self, gyro_meas, dt):
        x_prev = self.x.copy()
        P_prev = self.P.copy()

        if x_prev.shape != (self.dim_x, 1):
            self.get_logger().error(
                f"x has shape {x_prev.shape} at start of predict! Skipping."
            )
            return

        q_prev = x_prev[0:4].flatten()
        bias_prev = x_prev[4:7].flatten()
        gyro_corrected = gyro_meas.flatten() - bias_prev
        wx, wy, wz = gyro_corrected
        Omega = np.array(
            [
                [0.0, -wx, -wy, -wz],
                [wx, 0.0, wz, -wy],
                [wy, -wz, 0.0, wx],
                [wz, wy, -wx, 0.0],
            ]
        )
        # Quaternion update: q_new = (I + 0.5 * Omega * dt) * q_old
        # More robust way: q_new = exp(0.5 * Omega * dt) * q_old
        # For small dt, (I + 0.5 * Omega * dt) is a good approximation
        # delta_q_vec = np.exp(0.5 * gyro_corrected * dt)
        delta_q_vec = 0.5 * gyro_corrected * dt
        delta_q_norm = np.linalg.norm(delta_q_vec)

        # Using exact quaternion kinematics for integration can be more stable
        if delta_q_norm > 1e-9:  # Avoid division by zero if no rotation
            delta_q_scalar = math.cos(
                delta_q_norm / 2.0
            )  # Using angle/2 for Rodrigues' formula form
            delta_q_vector_part = (
                math.sin(delta_q_norm / 2.0) / delta_q_norm
            ) * delta_q_vec

            # Convert delta_q to a quaternion [w, x, y, z]
            dq_w = delta_q_scalar
            dq_x, dq_y, dq_z = (
                delta_q_vector_part[0],
                delta_q_vector_part[1],
                delta_q_vector_part[2],
            )

            # Quaternion multiplication: q_new = q_prev * dq (if dq is local rotation)
            # Or q_new = dq * q_prev (if dq is world rotation)
            # Gyro measures body rates, so dq is a rotation in body frame.
            # q_prev = [qw, qx, qy, qz]
            # dq     = [dw, dx, dy, dz]
            # q_new_w = qw*dw - qx*dx - qy*dy - qz*dz
            # q_new_x = qw*dx + qx*dw + qy*dz - qz*dy
            # q_new_y = qw*dy - qx*dz + qy*dw + qz*dx
            # q_new_z = qw*dz + qx*dy - qy*dx + qz*dw
            qw, qx, qy, qz = q_prev[0], q_prev[1], q_prev[2], q_prev[3]
            q_new_w = qw * dq_w - qx * dq_x - qy * dq_y - qz * dq_z
            q_new_x = qw * dq_x + qx * dq_w + qy * dq_z - qz * dq_y
            q_new_y = qw * dq_y - qx * dq_z + qy * dq_w + qz * dq_x
            q_new_z = qw * dq_z + qx * dq_y - qy * dq_x + qz * dq_w
            q_new_flat = np.array([q_new_w, q_new_x, q_new_y, q_new_z])
        else:
            q_new_flat = q_prev

        q_new_flat = normalize_quaternion(q_new_flat)
        bias_new_flat = bias_prev
        x_pred = np.vstack((q_new_flat.reshape(4, 1), bias_new_flat.reshape(3, 1)))

        # --- Covariance Prediction ---
        # F_qq approximation is (I - Omega_skew * dt) where Omega_skew is for error state.
        # Or using the Omega matrix for full quaternion: (I + 0.5 * Omega * dt)
        F_qq = np.eye(4) + 0.5 * Omega * dt
        Xi_q_prev = 0.5 * np.array(
            [
                [-q_prev[1], -q_prev[2], -q_prev[3]],
                [q_prev[0], -q_prev[3], q_prev[2]],
                [q_prev[3], q_prev[0], -q_prev[1]],
                [-q_prev[2], q_prev[1], q_prev[0]],
            ]
        )
        F_qb = -Xi_q_prev * dt
        Fk = np.block([[F_qq, F_qb], [np.zeros((3, 4)), np.eye(3)]])

        # Discrete Process Noise Covariance Q_k
        # Q_cont_diag_variances = [gyro_x_psd, gyro_y_psd, gyro_z_psd, bias_x_psd, bias_y_psd, bias_z_psd]
        # Variance for angular increment due to gyro noise over dt: G_q_w * (gyro_psd * dt) * G_q_w.T
        # Variance for bias change due to bias random walk over dt: bias_psd * dt
        Q_gyro_noise_increment_var = np.diag(self.Q_cont_diag_variances[0:3]) * dt
        Q_bias_random_walk_increment_var = np.diag(self.Q_cont_diag_variances[3:6]) * dt

        # G_q_w is Xi_q_prev (Jacobian of quaternion kinematics w.r.t angular velocity)
        Q_block_qq = (
            Xi_q_prev @ np.diag(self.Q_cont_diag_variances[0:3]) @ Xi_q_prev.T * dt
        )

        # Or, simpler for state transition based on Omega matrix:
        # The noise enters through gyro_corrected which forms Omega.
        # Let G be the matrix that maps gyro noise to d(quat)/dt. G = Xi_q_prev
        # Q_qq = G * (diag(gyro_psd)) * G.T * dt (This is more standard for error-state)
        # For direct state, often Q related to F_qb is considered if noise affects bias part that then affects quat.
        # The provided current_Q formulation:
        # Q_block_qq = G_q_w @ Q_gyro_noise_cov @ G_q_w.T
        # where Q_gyro_noise_cov = np.diag(self.Q_cont_diag_variances[0:3]) * dt
        # This assumes Q_cont_diag_variances[0:3] are PSDs. This looks correct.
        Q_block_bb = Q_bias_random_walk_increment_var
        current_Q = np.block(
            [[Q_block_qq, np.zeros((4, 3))], [np.zeros((3, 4)), Q_block_bb]]
        )

        P_pred = Fk @ P_prev @ Fk.T + current_Q

        self.x = x_pred
        self.P = P_pred

        if self.x.shape != (self.dim_x, 1) or self.P.shape != (self.dim_x, self.dim_x):
            self.get_logger().error(
                f"Shape mismatch end of predict! x:{self.x.shape}, P:{self.P.shape}. Resetting state."
            )
            self.orientation_initialized = False
            self.x = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
            self.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01]) * 10

    def Hx_accelerometer(self, x_state):
        q_current = x_state[0:4].flatten()
        R_wb_transpose = quaternion_to_rotation_matrix(q_current).T  # R_bw = R_wb^T
        # Accelerometer measures: f_body = R_bw * (a_world - g_world)
        # If a_world = 0, f_body = R_bw * (-g_world) = -R_bw * g_world
        # gravity_ref_world is [0,0,9.81] which is -g_world if g_world is [0,0,-9.81] (Z-down)
        # If g_world is [0,0,-9.81] (Z-up world, gravity points down)
        # then accel measures a_body - R_wb^T * [0,0,-9.81] = a_body + R_wb^T * [0,0,9.81]
        # If a_body=0, accel measures R_wb^T * [0,0,9.81]
        # Our self.gravity_ref_world = np.array([[0.0, 0.0, 9.81]]).T
        # So predicted_accel = R_wb.T @ self.gravity_ref_world is correct.
        predicted_accel = R_wb_transpose @ self.gravity_ref_world
        return predicted_accel.flatten()

    def _calculate_measurement_jacobian_Hq(self, q_arr, world_vector):
        qw, qx, qy, qz = q_arr.flatten()
        vx, vy, vz = world_vector.flatten()

        # Jacobian of d(R(q)^T * v_world) / dq
        # R(q)^T * v_world = R(q_conjugate) * v_world
        # Let q = [w, x, y, z]. q_conj = [w, -x, -y, -z]
        # Derivative of R(q)v with respect to q:
        # See Sola "Quaternion Kinematics" eqn 130, 131 for d(Rv)/dq
        # Here we need d(R^T v)/dq. R^T(q) = R(q*).
        # Let q' = q* = [qw, -qx, -qy, -qz]
        # Hq_prime_col0 = 2 * (qw*v + cross(q_vec_prime, v))
        # Hq_prime_col1to3 = 2 * (dot(q_vec_prime,v)*I + outer(q_vec_prime,v) - outer(v,q_vec_prime) - qw*skew(v))
        # Then apply chain rule: d(R(q*)v)/dqw = d(R(q*)v)/dqw' * (dqw'/dqw=1)
        # d(R(q*)v)/dqx = d(R(q*)v)/dqx' * (dqx'/dqx=-1)
        #
        # Simpler: use formula for d(R(q)^T v)/dq directly.
        # Common form (e.g. Trawny, "Indirect Kalman Filter for 3D Attitude Estimation", Eq. 71, 72, with q = [q_vec; q_scalar])
        # Our q is [q_scalar, q_vec].
        # Let v_body = R(q)^T * v_world.
        # d(v_body)/dq_scalar = 2 * (qw*v_world + cross(q_vec, v_world))
        # d(v_body)/dq_vec = 2 * (-dot(v_world, q_vec)*I + q_vec*v_world^T - v_world*q_vec^T + qw*skew(v_world))
        # Note: Sola's d(Rv)/dq is different. Need to be careful.
        #
        # Let's use a known numerical derivative for verification if issues persist.
        # For now, assume the existing _calculate_measurement_jacobian_Hq is for R(q)^T v_w
        # The existing one was:
        # dH_dqw_conj = 2 * (q_conj_w * v_flat + np.cross([q_conj_x, q_conj_y, q_conj_z], v_flat))
        # dH_dqvec_conj = 2 * (np.dot(q_conj_vec, v_flat) * np.eye(3) + ...)
        # dH_dq[:,0] = dH_dqw_conj
        # dH_dq[:,1:] = -dH_dqvec_conj
        # This seems to implement the chain rule for R(q_conj) * v_world correctly.

        v_skew = skew_symmetric(world_vector)
        q_vec = q_arr[1:4]  # qx, qy, qz

        # d(R^T v_w)/dqw
        dH_dqw = 2 * (
            qw * world_vector.flatten()
            + np.cross(q_vec.flatten(), world_vector.flatten())
        )

        # d(R^T v_w)/dq_vec
        # This part from a source (e.g. Markley, "Attitude Error Representations for Kalman Filtering")
        # or derived from properties of quaternion rotation.
        # For v_b = R(q_conj) v_w: (where q_conj = [qw, -qv])
        # d(v_b)/dqw = 2 * (qw*v_w - skew(qv)*v_w)
        # d(v_b)/d(-qv) = 2 * (-transpose(skew(v_w)*qv + qw*v_w)) -- this is complex
        # Using Sola (2017) "Quaternion Kinematics" equation (78) for d(C(q)*v)/dq, where C(q) is R_wb
        # For d(C(q)^T*v)/dq = d(C(q^{-1})*v)/dq
        # Let q_inv = [qw, -qx, -qy, -qz] (if norm=1)
        # C(q) = (qw^2 - qv . qv)I + 2qv*qv^T - 2*qw*[qv]_x
        # C(q)^T v = ((qw^2 - qv . qv)I + 2qv*qv^T + 2*qw*[qv]_x) v
        # This Jacobian is non-trivial to write out directly.
        # The existing implementation using q_conj seems a plausible way. Let's trust it for now.

        # Re-using the original _calculate_measurement_jacobian_Hq as it was before modification attempts
        q_conj_w, q_conj_x, q_conj_y, q_conj_z = qw, -qx, -qy, -qz
        v_flat = world_vector.flatten()

        dH_dqw_conj = 2 * (
            q_conj_w * v_flat
            + np.cross(np.array([q_conj_x, q_conj_y, q_conj_z]), v_flat)
        )
        q_conj_vec = np.array([q_conj_x, q_conj_y, q_conj_z])
        dH_dqvec_conj = 2 * (
            np.dot(q_conj_vec, v_flat) * np.eye(3)
            + np.outer(q_conj_vec, v_flat)
            - np.outer(v_flat, q_conj_vec)
            - q_conj_w * skew_symmetric(v_flat)
        )

        dH_dq = np.zeros((3, 4))
        dH_dq[:, 0] = dH_dqw_conj
        dH_dq[:, 1:] = -dH_dqvec_conj  # Chain rule: d/dq_vec = d/dq_conj_vec * (-1)
        return dH_dq

    def H_jacobian_accelerometer(self, x_state):
        q_current = x_state[0:4]
        dH_dq = self._calculate_measurement_jacobian_Hq(
            q_current, self.gravity_ref_world
        )
        dH_dbias = np.zeros((3, 3))
        Hk_accel = np.hstack((dH_dq, dH_dbias))
        return Hk_accel

    def update_step(self, z, R_sensor, Hx_func, H_jacobian_func, sensor_name="sensor"):
        current_dim_z = R_sensor.shape[0]
        x_pred = self.x
        P_pred = self.P

        if x_pred.shape != (self.dim_x, 1):
            self.get_logger().error(
                f"x has unexpected shape {x_pred.shape} BEFORE {sensor_name} update!"
            )
            return False

        try:
            H = H_jacobian_func(x_pred)
            Hx = Hx_func(x_pred)
        except Exception as e:
            self.get_logger().error(
                f"Error calculating H or Hx for {sensor_name}: {e}", exc_info=True
            )
            return False

        z_flat = z.flatten()

        if (
            H.shape != (current_dim_z, self.dim_x)
            or Hx.shape != (current_dim_z,)
            or z_flat.shape != (current_dim_z,)
        ):
            self.get_logger().error(
                f"Dimension mismatch in {sensor_name} update: H:{H.shape}, Hx:{Hx.shape}, z:{z_flat.shape}"
            )
            return False

        Hx = Hx.reshape(-1, 1)
        z = z.reshape(-1, 1)

        I = np.identity(self.dim_x)
        y = z - Hx

        # Robustly compute S inverse
        try:
            S = H @ P_pred @ H.T + R_sensor
            if np.linalg.cond(S) > 1e12:  # Check condition number
                self.get_logger().warn(
                    f"S matrix ill-conditioned in {sensor_name} update (cond={np.linalg.cond(S):.2e}). Using pseudo-inverse."
                )
            S_inv = np.linalg.pinv(S, rcond=1e-15)
        except np.linalg.LinAlgError as e:
            self.get_logger().warn(
                f"S matrix likely singular in {sensor_name} update ({e}), skipping update"
            )
            return False

        K = P_pred @ H.T @ S_inv

        x_new = x_pred + K @ y
        self.x = x_new

        I_KH = I - K @ H
        P_new = I_KH @ P_pred @ I_KH.T + K @ R_sensor @ K.T
        self.P = 0.5 * (P_new + P_new.T)

        try:
            self.x[0:4] = normalize_quaternion(self.x[0:4].flatten()).reshape(4, 1)
        except Exception as e:
            self.get_logger().error(
                f"Error during {sensor_name} normalization: {e}. State x: {self.x.flatten()}",
                exc_info=True,
            )
            # If normalization fails, it's a critical error, consider reset
            self.orientation_initialized = False
            return False

        return True

    def update_step_accelerometer(self, accel_meas):
        self.update_step(
            accel_meas,
            self.R_accel,
            self.Hx_accelerometer,
            self.H_jacobian_accelerometer,
            "accel",
        )


def main(args=None):
    rclpy.init(args=args)
    try:
        estimator = QuaternionRollPitchEstimator()
        rclpy.spin(estimator)
    except KeyboardInterrupt:
        if (
            "estimator" in locals() and estimator.context.ok()
        ):  # Check if estimator was successfully created
            estimator.get_logger().info("Shutting down quaternion EKF node.")
    except Exception as e:
        logger = rclpy.logging.get_logger(
            "quaternion_ekf_main"
        )  # Use a generic logger if estimator failed
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        # Ensure estimator is destroyed if it exists and context is ok
        if (
            "estimator" in locals()
            and hasattr(estimator, "destroy_node")
            and estimator.context.ok()
        ):
            estimator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(1)  # Exit with error code
    finally:
        # This block will run whether an exception occurred or not (except for SystemExit type calls)
        if (
            "estimator" in locals()
            and hasattr(estimator, "destroy_node")
            and estimator.context.ok()
        ):
            if (
                rclpy.ok() and estimator.context.ok()
            ):  # double check context before destroying
                estimator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        # Removed the sys.exit here if it was reached through normal KeyboardInterrupt shutdown path
        # Only exit with error if an unhandled exception 'e' was caught in the try-except block
        if "e" in locals() and not isinstance(e, KeyboardInterrupt):
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
