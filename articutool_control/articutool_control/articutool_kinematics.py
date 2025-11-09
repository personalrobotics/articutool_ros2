import numpy as np
import math
from typing import List, Tuple


class ArticutoolAnalyticalKinematics:
    """
    Encapsulates the 2-DOF analytical kinematic model for the Articutool.

    Defines the mapping between joint angles (pitch, roll) and the
    tool-tip's Y-axis ("up" vector) in the Articutool's base frame.

    Kinematic Model:
    - p: pitch angle, r: roll angle
    - Up Vector (v): [vx, vy, vz]^T

    Forward Kinematics (FK):
    - vx = cos(p) * cos(r)
    - vy = sin(r)
    - vz = sin(p) * cos(r)

    Inverse Kinematics (IK):
    - r = asin(vy)
    - p = atan2(vz, vx)
    """

    def __init__(self, epsilon: float = 1e-6):
        """
        Initializes the kinematic model.

        Args:
            epsilon: Small value for floating point comparisons.
        """
        self.EPSILON = epsilon
        self.WORLD_Z_UP_VECTOR = np.array([0.0, 0.0, 1.0])

    def _normalize_angle(self, angle: float) -> float:
        """Normalizes an angle to the range [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def compute_fk_vector(self, theta_p: float, theta_r: float) -> np.ndarray:
        """
        Computes the Forward Kinematics "up" vector.

        Args:
            theta_p: The pitch joint angle.
            theta_r: The roll joint angle.

        Returns:
            The 3D "up" vector (tool-tip Y-axis) in the base frame.
        """
        cp, sp = math.cos(theta_p), math.sin(theta_p)
        cr, sr = math.cos(theta_r), math.sin(theta_r)

        # vx = cp * cr
        # vy = sr
        # vz = sp * cr
        return np.array([cp * cr, sr, sp * cr])

    def compute_fk_matrix(self, theta_p: float, theta_r: float) -> np.ndarray:
        """
        Computes the full 3x3 rotation matrix from tooltip to base frame.

        Args:
            theta_p: The pitch joint angle.
            theta_r: The roll joint angle.

        Returns:
            The 3x3 rotation matrix (R_base_tooltip).
        """
        cp, sp = math.cos(theta_p), math.sin(theta_p)
        cr, sr = math.cos(theta_r), math.sin(theta_r)

        # This matrix is defined such that its second column is the
        # "up" vector [cp*cr, sr, sp*cr]^T from compute_fk_vector.
        return np.array([[cp * sr, cp * cr, -sp], [-cr, sr, 0], [sp * sr, sp * cr, cp]])

    def compute_fk_partials(
        self, theta_p: float, theta_r: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the partial derivatives of the FK rotation matrix.

        Args:
            theta_p: The pitch joint angle.
            theta_r: The roll joint angle.

        Returns:
            A tuple (dR_dtheta_p, dR_dtheta_r), where each element
            is a 3x3 matrix.
        """
        cp, sp = math.cos(theta_p), math.sin(theta_p)
        cr, sr = math.cos(theta_r), math.sin(theta_r)

        # Partial derivative of R_base_tooltip w.r.t. theta_p
        dR_dtheta_p_mat = np.array(
            [[-sp * sr, -sp * cr, -cp], [0, 0, 0], [cp * sr, cp * cr, -sp]]
        )

        # Partial derivative of R_base_tooltip w.r.t. theta_r
        dR_dtheta_r_mat = np.array(
            [[cp * cr, -cp * sr, 0], [sr, cr, 0], [sp * cr, -sp * sr, 0]]
        )

        return dR_dtheta_p_mat, dR_dtheta_r_mat

    def compute_jacobian(self, theta_p: float, theta_r: float) -> np.ndarray:
        """
        Computes the 3x2 analytical Jacobian.

        This matrix maps joint velocities [p_dot, r_dot] to the
        linear velocity of the "up" vector [vx_dot, vy_dot, vz_dot].

        Args:
            theta_p: The pitch joint angle.
            theta_r: The roll joint angle.

        Returns:
            The 3x2 Jacobian matrix.
        """
        cp, sp = math.cos(theta_p), math.sin(theta_p)
        cr, sr = math.cos(theta_r), math.sin(theta_r)

        # Column 1: Partial derivatives w.r.t. theta_p
        j11 = -sp * cr
        j21 = 0
        j31 = cp * cr

        # Column 2: Partial derivatives w.r.t. theta_r
        j12 = -cp * sr
        j22 = cr
        j32 = -sp * sr

        return np.array([[j11, j12], [j21, j22], [j31, j32]])

    def solve_ik_for_leveling(
        self, target_y_axis_in_atool_base: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Solves the analytical Inverse Kinematics for a target "up" vector.

        Finds the (pitch, roll) angles required to align the
        tool-tip's Y-axis with the given target vector.

        Args:
            target_y_axis_in_atool_base: The desired 3D "up" vector
                expressed in the Articutool's base frame.

        Returns:
            A list of (pitch, roll) solution tuples.
        """
        vx, vy, vz = target_y_axis_in_atool_base
        solutions: List[Tuple[float, float]] = []

        # From sin(r) = vy
        asin_arg_for_tr = vy
        if not (-1.0 - self.EPSILON <= asin_arg_for_tr <= 1.0 + self.EPSILON):
            return []  # No real solution

        asin_arg_for_tr_clipped = np.clip(asin_arg_for_tr, -1.0, 1.0)
        theta_r_sol1 = math.asin(asin_arg_for_tr_clipped)
        theta_r_sol2 = self._normalize_angle(math.pi - theta_r_sol1)

        candidate_thetas_r = [theta_r_sol1]
        if not math.isclose(theta_r_sol1, theta_r_sol2, abs_tol=self.EPSILON):
            candidate_thetas_r.append(theta_r_sol2)

        for theta_r in candidate_thetas_r:
            cos_theta_r = math.cos(theta_r)

            # Check for singularity (cos(r) is near zero)
            if math.isclose(cos_theta_r, 0.0, abs_tol=self.EPSILON):
                # If cos(r) is 0, vx and vz must also be 0 for a solution
                if math.isclose(vx, 0.0, abs_tol=self.EPSILON) and math.isclose(
                    vz, 0.0, abs_tol=self.EPSILON
                ):
                    # p is indeterminate, any value works. Choose 0.
                    solutions.append((0.0, self._normalize_angle(theta_r)))
                continue

            # Regular case: p = atan2(vz, vx)
            theta_p_sol = math.atan2(vz, vx)
            solutions.append(
                (self._normalize_angle(theta_p_sol), self._normalize_angle(theta_r))
            )
        return solutions
