cmake_minimum_required(VERSION 3.0.2)
project(relative_pose)

# Find catkin and other dependencies
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs   # Make sure geometry_msgs is included here
  message_generation
  # tf
  # tf_conversions
)

# Declare ROS messages
add_message_files(
  FILES
  # DepthTemperature.msg
  # DVLVelocity.msg
  RelativePose.msg
  PlaneApproximation.msg
)

# Python setup
catkin_python_setup()

# Generate messages AFTER catkin_python_setup
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs  # Make sure geometry_msgs is included here too
)

# Declare a catkin package
catkin_package(
  CATKIN_DEPENDS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  message_runtime
  # tf
  # tf_conversions
)


# Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
)

