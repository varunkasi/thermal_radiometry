from setuptools import setup
import os
from glob import glob

package_name = 'thermal_radiometry'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'opencv-python', 'tqdm'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Thermal radiometry package for calibrating thermal camera data',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'thermal_calibrator = thermal_radiometry.thermal_calibrator:main',
            'process_thermal_bag = thermal_radiometry.process_thermal_bag:main',
        ],
    },
)