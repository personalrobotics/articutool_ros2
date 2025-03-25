import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class RealSenseProcessor(Node):
    def __init__(self):
        super().__init__('realsense_processor')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.process_image,
            10
        )

    def process_image(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
            return

        # Display the image (optional, for debugging)
        cv2.imshow('Image', cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    realsense_processor = RealSenseProcessor()
    rclpy.spin(realsense_processor)
    realsense_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
