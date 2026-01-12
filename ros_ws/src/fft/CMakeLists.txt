cmake_minimum_required(VERSION 3.0.2)
project(fft)

# add_compile_options(-w)
if(UNIX)
  add_compile_options(-w)
else()
  add_compile_options(/w)
endif(UNIX)

if(UNIX)
  find_package(Eigen3 REQUIRED)
endif(UNIX)

# # Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  sensor_msgs
  std_msgs
  geometry_msgs
  message_generation
  rosbag
)

find_package(OpenCV REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_message_files(
  FILES
  NetApproximationFFT.msg
  NetSquare.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

# # Declare a catkin package
catkin_package(
  #INCLUDE_DIRS includes
  LIBRARIES fft
  CATKIN_DEPENDS roscpp sensor_msgs std_msgs message_runtime
)

# set(MAIN_SOURCE_FILE "src/main_ros_threads.cpp")
# set(MAIN_SOURCE_FILE "src/main_with_subscriber.cpp")
set(MAIN_SOURCE_FILE "src/main.cpp")

# set(MAIN_SOURCE_FILE "src/position_publisher.cpp")

# set(MAIN_SOURCE_FILE "src/main.cpp")
set(OUTPUT_FILE "fft_node")

set(SOURCE_FILES
  ${MAIN_SOURCE_FILE}
  data_handling/data_handling.cpp
  fft/fft.cpp
  fft/localmaxima.cpp
  fft/grid_detection.cpp
  fft/RPP.cpp
  fft/Rpoly.cpp
  shape_fitting/shape_fitting.cpp
  utils/stopwatch.cc
  utils/thread_pool.cpp
)

# # Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

if(UNIX)
  include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    include
    #src/includes
    ${EIGEN3_INCLUDE_DIR} # Add Eigen include directories
  )
endif(UNIX)

add_executable(${OUTPUT_FILE}
  ${MAIN_SOURCE_FILE}
  src/fft/fft.cpp
  src/data_handling/data_handling.cpp
  src/fft/localmaxima.cpp
  src/fft/grid_detection.cpp
  src/fft/RPP.cpp
  src/fft/Rpoly.cpp
  src/shape_fitting/shape_fitting.cpp
  src/utils/stopwatch.cc
  src/utils/thread_pool.cpp
)

add_dependencies(${OUTPUT_FILE} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

if(UNIX)
  target_link_libraries(
    ${OUTPUT_FILE}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    Threads::Threads
    Eigen3::Eigen
  )
endif(UNIX)

target_compile_features(${OUTPUT_FILE} PUBLIC cxx_std_17)
set_target_properties(${OUTPUT_FILE} PROPERTIES CXX_EXTENSIONS OFF)

# # Mark the node for installation
install(TARGETS ${OUTPUT_FILE}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
