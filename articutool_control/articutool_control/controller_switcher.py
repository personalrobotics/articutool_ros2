import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import SwitchController, ListControllers

class ControllerSwitcher(Node):
    def __init__(self):
        super().__init__("controller_switcher")
        self.switch_controller_client = self.create_client(
            SwitchController, "/articutool/controller_manager/switch_controller"
        )
        self.list_controllers_client = self.create_client(
            ListControllers, f"/articutool/controller_manager/list_controllers"
        )

        while not self.switch_controller_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Switch controller service not available, waiting again..."
            )

        while not self.list_controllers_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "List controller service not available, waiting again..."
            )

    def switch_controllers(
        self, activate_controllers, deactivate_controllers, strictness=1
    ):
        request = SwitchController.Request()
        request.activate_controllers = activate_controllers
        request.deactivate_controllers = deactivate_controllers
        request.strictness = strictness
        future = self.switch_controller_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            if not future.result().ok:
                self.get_logger().error("Controller switch failed.")
        else:
            self.get_logger().error("Service call failed.")

    def list_controllers(self):
        request = ListControllers.Request()
        future = self.list_controllers_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            for controller in future.result().controller:
                self.get_logger().info(
                    f"Controller: {controller.name}, State: {controller.state}"
                )
        else:
            self.get_logger().error("Service call failed.")
