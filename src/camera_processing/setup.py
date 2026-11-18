
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
            'occ_mapping = camera_processing.occ_mapping:main',
        ],
    },
)
