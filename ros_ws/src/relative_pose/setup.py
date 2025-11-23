from setuptools import setup

setup(
  name='relative_pose',
  version='0.0.1',
  packages=['relative_pose'],
  package_dir={'': 'src'},
  install_requires=['rospy', 'std_msgs', 'sensor_msgs', 'geometry_msgs'],
)
