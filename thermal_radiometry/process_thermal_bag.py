#!/usr/bin/env python3
"""
Thermal Bag Processor

This script processes a thermal bag file to extract and convert:
1. Raw thermal images (mono16)
2. Calibrated temperature images
3. 8-bit visualization images

It uses the algorithms from thermal_radiometry and mono16_converter packages
but processes the entire bag in one go, storing results for later visualization.
"""

import os
import sys
import time
import json
import numpy as np
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.serialization import deserialize_message
# Replace the deprecated import
from rosidl_runtime_py.utilities import get_message
import argparse
from pathlib import Path
import yaml
import concurrent.futures
from tqdm import tqdm
import rosbag2_py

# Check for GPU support
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        GPU_AVAILABLE = True
        print("GPU acceleration available. Will use CUDA for image processing.")
    else:
        GPU_AVAILABLE = False
        print("No CUDA-capable GPU found. Using CPU processing.")
except:
    GPU_AVAILABLE = False
    print("OpenCV not built with CUDA support. Using CPU processing.")

class ThermalBagProcessor:
    """Processes thermal bag files and stores the processed data for visualization"""
    
    def __init__(self, bag_path, robot_name, output_dir=None, use_gpu=True):
        """Initialize the processor with the bag file path and robot name"""
        self.bag_path = bag_path
        self.robot_name = robot_name
        
        # Set output directory
        if output_dir is None:
            bag_name = os.path.basename(bag_path)
            if bag_name.endswith('.bag') or bag_name.endswith('.mcap'):
                bag_name = os.path.splitext(bag_name)[0]
            self.output_dir = f"{bag_name}_processed"
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize processing utilities
        self.bridge = CvBridge()
        
        # Set GPU usage flag
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Configure topic names
        self.raw_topic = f"/{self.robot_name}/hand/sensor/thermal/image_raw"
        
        # Load calibration parameters
        self.load_calibration_params()
        
        # Store processed frame data
        self.frames = {
            'raw': [],          # Original raw mono16 frames
            'calibrated': [],   # Calibrated temperature frames
            'mono8': [],        # 8-bit visualization frames
            'timestamps': [],   # Frame timestamps
            'metadata': {}      # Processing metadata
        }
        
        print(f"Processing bag: {self.bag_path}")
        print(f"Robot name: {self.robot_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        
    def load_calibration_params(self):
        """Load the calibration parameters for the specified robot"""
        # Default parameters in case config file is not found
        if self.robot_name == 'spot1':
            self.calib_params = {
                'a': -5.74228169e-8,
                'b': 6.43838380e-3,
                'c': -90.2560653
            }
        elif self.robot_name == 'spot2':
            self.calib_params = {
                'a': 8.152e-7,  # Converted from string '0000008152'
                'b': -0.0294304271,
                'c': 279.0969265845
            }
        else:
            # Default to spot1 if unknown robot
            self.calib_params = {
                'a': -5.74228169e-8,
                'b': 6.43838380e-3,
                'c': -90.2560653
            }
            
        print(f"Using calibration parameters for {self.robot_name}: {self.calib_params}")
    
    def apply_calibration(self, raw_image):
        """Apply thermal calibration to convert raw values to temperature"""
        a = self.calib_params['a']
        b = self.calib_params['b']
        c = self.calib_params['c']
        
        ambient_temp_adjustment = 2  # -9.8 for night mode, +2 for day mode
        # Apply calibration formula: ax² + bx + c
        return ((a * raw_image**2 + b * raw_image + c) + ambient_temp_adjustment + 1.08) / 1.089
    
    def normalize_to_uint8(self, image):
        """Efficiently convert to 8-bit with full dynamic range"""
        img_min = image.min()
        img_max = image.max()
        if img_max == img_min:
            return np.zeros(image.shape, dtype=np.uint8)
        return ((image - img_min) * 255.0 / (img_max - img_min)).astype(np.uint8)

    def process_mono16_to_mono8(self, mono16_image):
        """
        Convert a mono16 image to mono8 with enhanced visualization
        Enhanced with advanced processing techniques from mono16_to_mono8_converter.py, including wavelet denoising.
        """
        try:
            # Convert to float for processing
            img_float = mono16_image.astype(np.float32)

            # Step 1: Normalized contrast stretching using percentile clipping
            p_low = np.percentile(img_float, 2)
            p_high = np.percentile(img_float, 98)

            if p_high <= p_low:
                p_high = p_low + 1

            normalized = np.clip((img_float - p_low) / (p_high - p_low), 0, 1)
            img_8bit = (normalized * 255).astype(np.uint8)

            # Step 2: Apply wavelet denoising if available
            try:
                import pywt
                coeffs = pywt.wavedec2(img_8bit.astype(np.float32), 'db4', level=1)
                new_coeffs = [coeffs[0]]

                detail_coeffs = coeffs[1]
                sigma = np.median(np.abs(detail_coeffs[0])) / 0.6745
                threshold = sigma * 1.0  # Gentle thresholding to preserve details

                for i in range(1, len(coeffs)):
                    coeff_details = []
                    for j in range(len(coeffs[i])):
                        processed = pywt.threshold(coeffs[i][j], threshold, mode='soft')
                        coeff_details.append(processed)

                    new_coeffs.append(tuple(coeff_details))

                denoised = pywt.waverec2(new_coeffs, 'db4')

                if denoised.shape != img_8bit.shape:
                    denoised = cv2.resize(denoised, (img_8bit.shape[1], img_8bit.shape[0]))

                denoised = np.clip(denoised, 0, 255).astype(np.uint8)
                working_img = denoised

            except ImportError:
                print("PyWavelets not installed. Proceeding without wavelet denoising.")
                working_img = img_8bit
            except Exception as e:
                print(f"Error in wavelet denoising: {e}")
                working_img = img_8bit

            # Step 3: Apply Sobel filters for gradient calculation
            sobelx = cv2.Sobel(working_img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(working_img, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)

            # Normalize gradient magnitude
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Identify strong gradients (important features/edges)
            _, strong_edges = cv2.threshold(magnitude, 30, 255, cv2.THRESH_BINARY)

            # Apply morphological operations to connect nearby edges
            kernel = np.ones((3, 3), np.uint8)
            connected_edges = cv2.morphologyEx(strong_edges, cv2.MORPH_CLOSE, kernel)

            # Dilate to include areas around important features
            feature_mask = cv2.dilate(connected_edges, kernel, iterations=1) > 0

            # Step 4: Final bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(working_img, 5, 50, 50)

            # Combine enhanced features with the filtered image
            final_img = np.copy(filtered)
            final_img[feature_mask] = cv2.addWeighted(
                working_img[feature_mask].astype(np.float32), 1.5,  # Increase contrast for features
                filtered[feature_mask].astype(np.float32), -0.5, 0
            ).astype(np.uint8)

            return final_img

        except Exception as e:
            print(f"Error in process_mono16_to_mono8: {e}")
            # Fallback to basic normalization
            return self.normalize_to_uint8(mono16_image)
    
    def process_frame(self, raw_cv_image, timestamp):
        """Process a single frame and return all versions"""
        # Store the raw image
        raw_data = raw_cv_image.copy()
        
        # Create calibrated temperature image
        calibrated_data = self.apply_calibration(raw_cv_image.astype(np.float64)).astype(np.float32)
        
        # Create mono8 visualization image
        mono8_data = self.process_mono16_to_mono8(raw_cv_image)
        
        return {
            'raw': raw_data,
            'calibrated': calibrated_data,
            'mono8': mono8_data,
            'timestamp': timestamp
        }
    
    def extract_messages_from_bag(self):
        """Extract and count messages from the bag file"""
        # Initialize ROS context if needed
        context = rclpy.Context()
        if not context.ok():
            rclpy.init(context=context)
        
        # Detect storage format based on file extension
        storage_id = 'mcap' if self.bag_path.endswith('.mcap') else 'sqlite3'
        print(f"Using {storage_id} reader for bag file")
        
        try:
            # First pass: count messages
            reader = rosbag2_py.SequentialReader()
            reader.open(
                rosbag2_py.StorageOptions(uri=self.bag_path, storage_id=storage_id),
                rosbag2_py.ConverterOptions(
                    input_serialization_format='cdr',
                    output_serialization_format='cdr')
            )
            
            topic_types = reader.get_all_topics_and_types()
            type_map = {topic_type.name: topic_type.type for topic_type in topic_types}
            
            # Count messages in raw topic
            message_count = 0
            while reader.has_next():
                topic, data, t = reader.read_next()
                if topic == self.raw_topic:
                    message_count += 1
                    
            # Second pass: extract messages
            reader = rosbag2_py.SequentialReader()
            reader.open(
                rosbag2_py.StorageOptions(uri=self.bag_path, storage_id=storage_id),
                rosbag2_py.ConverterOptions(
                    input_serialization_format='cdr',
                    output_serialization_format='cdr')
            )
            
            # Process all messages with a progress bar
            raw_messages = []
            print(f"Extracting {message_count} messages from topic: {self.raw_topic}")
            
            with tqdm(total=message_count, desc="Extracting frames") as pbar:
                while reader.has_next():
                    topic, data, timestamp_ns = reader.read_next()
                    if topic == self.raw_topic:
                        # Get message type using get_message instead of get_interface
                        msg_type = get_message(type_map[topic])
                        msg = deserialize_message(data, msg_type)
                        timestamp = timestamp_ns / 1.0e9  # Convert to seconds
                        raw_messages.append((msg, timestamp))
                        pbar.update(1)
            
            return raw_messages
            
        except Exception as e:
            print(f"Error reading bag file: {e}")
            # Try alternative storage ID if first attempt failed
            if storage_id == 'mcap':
                print("Trying sqlite3 storage format...")
                self.bag_path = self.bag_path.replace('.mcap', '')
                storage_id = 'sqlite3'
            else:
                print("Trying mcap storage format...")
                storage_id = 'mcap'
                
            try:
                # Retry with alternative format
                reader = rosbag2_py.SequentialReader()
                reader.open(
                    rosbag2_py.StorageOptions(uri=self.bag_path, storage_id=storage_id),
                    rosbag2_py.ConverterOptions(
                        input_serialization_format='cdr',
                        output_serialization_format='cdr')
                )
                
                topic_types = reader.get_all_topics_and_types()
                type_map = {topic_type.name: topic_type.type for topic_type in topic_types}
                
                raw_messages = []
                while reader.has_next():
                    topic, data, timestamp_ns = reader.read_next()
                    if topic == self.raw_topic:
                        msg_type = get_message(type_map[topic])
                        msg = deserialize_message(data, msg_type)
                        timestamp = timestamp_ns / 1.0e9  # Convert to seconds
                        raw_messages.append((msg, timestamp))
                
                print(f"Successfully extracted {len(raw_messages)} messages with {storage_id} reader")
                return raw_messages
                
            except Exception as e2:
                print(f"Failed to read bag file with alternative format: {e2}")
                raise RuntimeError(f"Could not read bag file in either format. Original error: {e}, Alternative error: {e2}")
    
    def process_bag(self):
        """Process the entire bag file and store results"""
        start_time = time.time()
        
        # Extract all raw messages from the bag
        raw_messages = self.extract_messages_from_bag()
        
        print(f"Processing {len(raw_messages)} frames...")
        
        # Process each frame
        with tqdm(total=len(raw_messages), desc="Processing") as pbar:
            for msg, timestamp in raw_messages:
                try:
                    # Convert ROS Image to OpenCV image
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
                    
                    # Process the frame
                    result = self.process_frame(cv_image, timestamp)
                    
                    # Store results
                    self.frames['raw'].append(result['raw'])
                    self.frames['calibrated'].append(result['calibrated'])
                    self.frames['mono8'].append(result['mono8'])
                    self.frames['timestamps'].append(timestamp)
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
        
        # Convert lists to numpy arrays for more efficient storage
        self.frames['raw'] = np.array(self.frames['raw'], dtype=np.uint16)
        self.frames['calibrated'] = np.array(self.frames['calibrated'], dtype=np.float32)
        self.frames['mono8'] = np.array(self.frames['mono8'], dtype=np.uint8)
        self.frames['timestamps'] = np.array(self.frames['timestamps'], dtype=np.float64)
        
        # Add metadata
        self.frames['metadata'] = {
            'robot_name': self.robot_name,
            'source_bag': self.bag_path,
            'frame_count': len(self.frames['timestamps']),
            'processing_time': time.time() - start_time,
            'calibration_params': self.calib_params,
            'raw_shape': self.frames['raw'].shape,
            'calibrated_shape': self.frames['calibrated'].shape,
            'mono8_shape': self.frames['mono8'].shape,
        }
        
        print(f"Processing completed in {self.frames['metadata']['processing_time']:.2f} seconds")
        print(f"Processed {self.frames['metadata']['frame_count']} frames")
        
        return self.frames
    
    def save_processed_data(self):
        """Save the processed data to the output directory"""
        print(f"Saving processed data to {self.output_dir}...")
        
        # Save each data array as a separate file to avoid memory issues
        np.save(os.path.join(self.output_dir, 'raw_frames.npy'), self.frames['raw'])
        np.save(os.path.join(self.output_dir, 'calibrated_frames.npy'), self.frames['calibrated'])
        np.save(os.path.join(self.output_dir, 'mono8_frames.npy'), self.frames['mono8'])
        np.save(os.path.join(self.output_dir, 'timestamps.npy'), self.frames['timestamps'])
        
        # Save metadata as JSON
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(self.frames['metadata'], f, indent=2)
            
        print(f"Data saved successfully to {self.output_dir}")
        
        # Return the path to the processed data
        return self.output_dir

def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description='Process a thermal bag file')
    parser.add_argument('bag_path', help='Path to the thermal bag file')
    parser.add_argument('robot_name', help='Robot name (e.g., spot1, spot2)')
    parser.add_argument('--output', help='Output directory for processed data')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # Create the processor
    processor = ThermalBagProcessor(
        args.bag_path,
        args.robot_name,
        output_dir=args.output,
        use_gpu=not args.no_gpu
    )
    
    # Process the bag
    processor.process_bag()
    
    # Save the results
    output_path = processor.save_processed_data()
    
    print(f"Processing complete. Results saved to: {output_path}")
    return output_path

if __name__ == '__main__':
    main()