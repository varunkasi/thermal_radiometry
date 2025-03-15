import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, EnvironmentVariable

def launch_setup(context, *args, **kwargs):
    # Get environment variables with defaults
    robot_name = os.environ.get('ROBOT_NAME', 'robot1')
    position = os.environ.get('THERMAL_POSITION', 'front')
    sensor_name = os.environ.get('THERMAL_SENSOR_NAME', 'thermal')
    
    # Construct namespace
    namespace = f"{robot_name}/{position}/sensor/{sensor_name}"
    
    # Load configuration
    config_file = os.path.join(
        get_package_share_directory('thermal_radiometry'),
        'config',
        'thermal_calibrator_params.yaml'
    )
    
    node = Node(
        package='thermal_radiometry',
        executable='thermal_calibrator',
        name='thermal_calibrator',
        namespace=namespace,
        parameters=[config_file],
        remappings=[
            ('image_raw', f'/{namespace}/image_raw'),
            ('image_calibrated', f'/{namespace}/image_calibrated'),
        ]
    )
    
    return [node]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])

