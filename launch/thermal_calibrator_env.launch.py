import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import EnvironmentVariable, PathJoinSubstitution

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('thermal_radiometry'),
        'config',
        'thermal_calibrator_params.yaml'
    )
    
    # Construct namespace using environment variables with defaults
    namespace = PathJoinSubstitution([
        EnvironmentVariable('ROBOT_NAME', default_value='spot1'),
        EnvironmentVariable('THERMAL_POSITION', default_value='hand'),
        'sensor',
        EnvironmentVariable('THERMAL_SENSOR_NAME', default_value='thermal')
    ])
    
    return LaunchDescription([
        Node(
            package='thermal_radiometry',
            executable='thermal_calibrator',
            name='thermal_calibrator',
            namespace=namespace,
            parameters=[config_file],
            remappings=[
                ('image_raw', [namespace, '/image_raw']),
                ('image_calibrated', [namespace, '/image_calibrated']),
            ]
        )
    ])

