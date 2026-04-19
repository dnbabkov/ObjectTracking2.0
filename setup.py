from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'object_tracking_2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*'),
        ),
        (
            os.path.join('share', package_name, 'model_weights'),
            glob('./object_tracking_2/model_weights/*'),
        ),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='clv',
    maintainer_email='dnbabkov@gmail.com',
    description='ROS2 object tracking package with selectable segmentators',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker_node = object_tracking_2.tracker_node:main',
            'episode_test_manager = object_tracking_2.episode_test_manager:main',
            'tracker_node_dynamic = object_tracking_2.tracker_node_dynamic:main',
        ],
    },
)