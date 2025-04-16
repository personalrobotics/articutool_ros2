import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Time
from articutool_control.srv import ExecuteJointTrajectory

class SendTrajectoryClient(Node):
    def __init__(self):
        super().__init__('send_trajectory_client')
        self.client = self.create_client(ExecuteJointTrajectory, 'execute_joint_trajectory')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the trajectory execution service...')
        self.req = ExecuteJointTrajectory.Request()

    def send_trajectory(self, joint_names, points, timeout=10.0):
        self.req.trajectory.joint_names = joint_names
        self.req.trajectory.points = points
        self.req.execution_timeout = timeout

        self.future = self.client.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        if self.future.result() is not None:
            response = self.future.result()
            self.get_logger().info(f'Trajectory execution result: Success={response.success}, Message="{response.message}"')
            return response.success, response.message
        else:
            self.get_logger().error('Service call failed %r' % (self.future.exception(),))
            return False, "Service call failed."

def main(args=None):
    rclpy.init(args=args)
    client = SendTrajectoryClient()

    # Example trajectory
    joint_names = ['joint1', 'joint2']
    point1 = JointTrajectoryPoint()
    point1.positions = [1.0, 2.0]
    point1.time_from_start.sec = 2
    point2 = JointTrajectoryPoint()
    point2.positions = [2.0, 1.5]
    point2.time_from_start.sec = 5
    points = [point1, point2]

    success, message = client.send_trajectory(joint_names, points, timeout=15.0)

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
