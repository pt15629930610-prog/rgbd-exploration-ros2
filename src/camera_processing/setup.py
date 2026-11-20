
from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'camera_processing'

setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(exclude=['test']),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'worlds', 'meshes'), glob('worlds/meshes/*.dae')),
    ],

    install_requires=['setuptools'],

    zip_safe=True,

    maintainer='yahboom',
    maintainer_email='yahboom@todo.todo',

    description='RGB-D perception and obstacle processing',

    license='TODO: License declaration',

    tests_require=['pytest'],

    entry_points={
        'console_scripts': [
            'rgbd_subscriber = camera_processing.rgbd_subscriber:main',

            'obstacle_clustering = camera_processing.obstacle_clustering:main',

            'open3d_viewer = camera_processing.open3d_viewer:main',
            'map_accumulator = camera_processing.map_accumulator:main',
            'frontier_explorer = camera_processing.frontier_explorer:main',
            'door_snapshot = camera_processing.door_snapshot:main',
            'door_navigator = camera_processing.door_navigator:main',
            'door_console = camera_processing.door_console:main',
            'route_console = camera_processing.route_console:main',
            'route_navigator = camera_processing.route_navigator:main',
            'occ_mapping = camera_processing.occ_mapping:main',
        ],
    },
)
