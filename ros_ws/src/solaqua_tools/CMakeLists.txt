cmake_minimum_required(VERSION 3.0.2)
project(solaqua_tools)

find_package(catkin REQUIRED COMPONENTS)

catkin_python_setup()

catkin_package()

# Install the compute script
catkin_install_python(PROGRAMS
  scripts/compute_net_distance.py
  scripts/sonar_topic_to_npz.py
  scripts/sonar_live_net_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install python modules (utils)
install(DIRECTORY utils
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY config scripts
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
  PATTERN "*.pyc" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
)
