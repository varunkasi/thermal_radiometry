#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ThermalCalibrator(Node):
    def __init__(self):
        super().__init__('thermal_calibrator')
        
        # Get robot name from environment variable
        self.robot_name = os.environ.get('ROBOT_NAME', 'spot1')
        self.get_logger().info(f"Initializing thermal calibrator for {self.robot_name}")
        
        # Declare parameters with default values
        self.declare_parameter('input_topic', 'image_raw')
        self.declare_parameter('output_topic', 'image_calibrated')
        self.declare_parameter(f'calibration.{self.robot_name}.a', -5.74228169e-8)
        self.declare_parameter(f'calibration.{self.robot_name}.b', 6.43838380e-3)
        self.declare_parameter(f'calibration.{self.robot_name}.c', -90.2560653)
        
        # Get parameters
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        
        # Get calibration parameters and handle special cases
        self.a = self.get_parameter(f'calibration.{self.robot_name}.a').value
        self.b = self.get_parameter(f'calibration.{self.robot_name}.b').value
        self.c = self.get_parameter(f'calibration.{self.robot_name}.c').value
        
        # Special handling for spot2 parameter which is stored as a string in the config
        if self.robot_name == 'spot2' and isinstance(self.a, str) and self.a == '0000008152':
            self.a = 8.152e-6
            self.get_logger().info(f"Converted spot2 parameter a from string '0000008152' to float {self.a}")
        
        self.get_logger().info(f"Using calibration parameters: a={self.a}, b={self.b}, c={self.c}")
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Track message counts
        self.received_count = 0
        self.published_count = 0
        
        # Create publisher for calibrated thermal image
        self.calibrated_pub = self.create_publisher(
            Image, 
            output_topic, 
            10
        )
        
        # Subscribe to raw thermal image
        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self.calibration_callback,
            10
        )
        
        self.get_logger().info("Thermal calibrator initialized")
        self.get_logger().info(f"Input topic: {input_topic}")
        self.get_logger().info(f"Output topic: {output_topic}")
    
    def apply_calibration(self, raw_image):
        """Apply calibration to the raw image using CPU"""
        return self.a * (raw_image**2) + self.b * raw_image + self.c
    
    def calibration_callback(self, msg):
        try:
            # Track message count
            self.received_count += 1
            
            # Convert ROS Image message to NumPy array
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
            
            # Apply calibration (using float64 for better precision)
            raw_img = cv_image.astype(np.float64)
            calibrated_img = self.apply_calibration(raw_img)
            
            # Convert to float32 for ROS message
            calibrated_img = calibrated_img.astype(np.float32)
            
            # Create and publish calibrated image message
            calibrated_msg = self.bridge.cv2_to_imgmsg(calibrated_img, encoding='32FC1')
            calibrated_msg.header = msg.header
            self.calibrated_pub.publish(calibrated_msg)
            
            # Track published count
            self.published_count += 1
            
            # Log occasionally for progress tracking
            if self.published_count % 100 == 0:
                self.get_logger().info(f"Processed {self.published_count} images")
            
        except Exception as e:
            self.get_logger().error(f"Error processing thermal image: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    
    try:
        calibrator = ThermalCalibrator()
        rclpy.spin(calibrator)
    except Exception as e:
        print(f"Error in thermal_calibrator: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()