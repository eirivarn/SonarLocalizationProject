cmake_minimum_required(VERSION 2.8)
project(fft)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -pg -Wall -Wno-long-long -pedantic")
endif()

find_package(OpenCV REQUIRED)

set(SOURCE_FILES
  main_with_net_plane_approximation.cpp
  data_handling/data_handling.cpp
  fft/fft.cpp
  fft/localmaxima.cpp
  fft/grid_detection.cpp
  fft/RPP.cpp
  fft/Rpoly.cpp
  shape_fitting/shape_fitting.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
#target_link_libraries (${PROJECT_NAME} ${EIGEN_INCLUDE_DIR})
