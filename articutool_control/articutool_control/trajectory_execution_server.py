import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from articutool_interfaces.srv import ExecuteJointTrajectory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time

class TrajectoryExecutionServer(Node):
    def __init__(self):
        super().__init__('trajectory_execution_server')

        # QoS profile for status subscriptions
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/articutool/joint_trajectory_controller/joint_trajectory',
            10  # QoS depth
        )

        self.service = self.create_service(
            ExecuteJointTrajectory,
            '/articutool/execute_joint_trajectory',
            self.execute_trajectory_callback)

        # Variables to track execution status
        self._goal_handle = None
        self._is_executing = False
        self._execution_result = None
        self._status_subscription = None

        self.get_logger().info('Trajectory execution service started.')

    def _goal_response_callback(self, goal_future):
        self._goal_handle = goal_future.result()
        if not self._goal_handle.accepted:
            self.get_logger().warn('Trajectory goal was rejected!')
            self._is_executing = False
            self._execution_result = False, 'Trajectory goal rejected by controller.'
            return

        self.get_logger().info('Trajectory goal accepted by controller.')
        self._is_executing = True
        self._execution_result = None
        self._get_result_future = self._goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._get_result_callback)

    def _get_result_callback(self, result_future):
        if result_future.result():
            status = result_future.result().status
            if status == 3:  # GoalStatus.SUCCEEDED
                self._execution_result = True, 'Trajectory execution succeeded.'
                self.get_logger().info('Trajectory execution finished successfully.')
            else:
                self._execution_result = False, f'Trajectory execution failed with status: {status}'
                self.get_logger().warn(f'Trajectory execution failed with status: {status}')
        else:
            self._execution_result = False, 'Failed to get trajectory execution result.'
            self.get_logger().error('Failed to get trajectory execution result.')
        self._is_executing = False

    def execute_trajectory_callback(self, request, response):
        self.get_logger().info('Received trajectory execution request.')
        trajectory = request.trajectory
        execution_timeout = request.execution_timeout if request.execution_timeout > 0 else float('inf')

        # Publish the trajectory to the controller
        self.trajectory_publisher.publish(trajectory)
        self.get_logger().info('Published trajectory to the controller.')

        start_time = self.get_clock().now().to_sec()

        # Wait for the trajectory to complete or timeout
        while self._is_executing is True and (self.get_clock().now().to_sec() - start_time < execution_timeout):
            rclpy.spin_once(self, timeout_sec=0.1)  # Check for status updates

        if self._execution_result is not None:
            response.success, response.message = self._execution_result
        elif self._is_executing:
            response.success = False
            response.message = "Trajectory execution timed out (waiting for controller result)."
            self.get_logger().warn(response.message)
        else:
            # This case might happen if the goal was rejected immediately
            response.success = False
            response.message = "Trajectory execution failed (no result received from controller)."
            self.get_logger().warn(response.message)

        return response

def main(args=None):
    rclpy.init(args=args)
    trajectory_execution_server = TrajectoryExecutionServer()
    executor = SingleThreadedExecutor() # Use a single threaded executor for simpler state management
    executor.add_node(trajectory_execution_server)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        trajectory_execution_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
