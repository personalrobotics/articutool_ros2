import numpy as np
import pinocchio
from pinocchio.robot_wrapper import RobotWrapper

class PinocchioIK:
    def __init__(self, urdf_path, base_link, end_effector_link):
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [base_link])
        self.model = self.robot.model
        self.data = self.model.createData()
        self.base_link_id = self.model.getFrameId(base_link)
        self.end_effector_id = self.model.getFrameId(end_effector_link)

    def compute_ik(self, target_pose, q_init, eps=1e-4, IT_MAX=1000, DT=1e-1, damp=1e-12):
        q = q_init.copy()
        i = 0
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)
            current_pose = self.robot.framePlacement(q, self.end_effector_id)
            dMi = target_pose.actInv(current_pose)
            err = pinocchio.log(dMi).vector
            if np.linalg.norm(err) < eps:
                return q, True
            if i >= IT_MAX:
                return q, False
            J = pinocchio.computeFrameJacobian(self.model, self.data, q, self.end_effector_id, pinocchio.ReferenceFrame.LOCAL)
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pinocchio.integrate(self.model, q, v * DT)
            i += 1

    def get_joint_names(self):
        return self.model.names[2:] #remove "world" and "universe"

    def get_joint_limits(self):
      lower_limits = self.model.lowerPositionLimit[7:]
      upper_limits = self.model.upperPositionLimit[7:]
      return lower_limits, upper_limits
