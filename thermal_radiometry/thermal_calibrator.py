#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
import numpy as np
import cupy as cp
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
        self.declare_parameter('output_topic', f'/{os.getenv("ROBOT_NAME")}/hand/sensor/thermal/image_calibrated')
        self.declare_parameter(f'calibration.{self.robot_name}.a', -5.74228169e-8)
        self.declare_parameter(f'calibration.{self.robot_name}.b', 6.43838380e-3)
        self.declare_parameter(f'calibration.{self.robot_name}.c', -90.2560653)
        
        # Get parameters
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.a = self.get_parameter(f'calibration.{self.robot_name}.a').value
        self.b = self.get_parameter(f'calibration.{self.robot_name}.b').value
        self.c = self.get_parameter(f'calibration.{self.robot_name}.c').value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Pre-allocate GPU memory buffers for reuse
        self.gpu_buffer = None
        self.last_shape = None
        
        # Create publisher for calibrated thermal image
        self.calibrated_pub = self.create_publisher(
            Image, 
            output_topic, 
            1
        )
        
        # Subscribe to raw thermal image
        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self.calibration_callback,
            1
        )
        
        self.get_logger().info("Thermal calibrator initialized")
        self.get_logger().info(f"Input topic: {input_topic}")
        self.get_logger().info(f"Output topic: {output_topic}")
    
    def apply_calibration_gpu(self, raw_gpu):
        """Apply calibration using GPU acceleration"""
        return self.a * (raw_gpu**2) + self.b * raw_gpu + self.c
    
    def calibration_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
            
            # Check if we need to allocate new GPU buffer
            if self.gpu_buffer is None or self.last_shape != cv_image.shape:
                self.last_shape = cv_image.shape
                self.gpu_buffer = cp.empty(cv_image.shape, dtype=cp.float32)
            
            # Transfer image to GPU
            gpu_image = cp.asarray(cv_image, dtype=cp.float64)
            
            # Apply calibration on GPU
            calibrated_gpu = self.apply_calibration_gpu(gpu_image)
            
            # Transfer result back to CPU
            calibrated_cpu = cp.asnumpy(calibrated_gpu.astype(cp.float32))
            
            # Create and publish calibrated image message
            calibrated_msg = self.bridge.cv2_to_imgmsg(calibrated_cpu, encoding='32FC1')
            calibrated_msg.header = msg.header
            self.calibrated_pub.publish(calibrated_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing thermal image: {e}")

def main(args=None):
    rclpy.init(args=args)
    calibrator = ThermalCalibrator()
    rclpy.spin(calibrator)
    calibrator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

