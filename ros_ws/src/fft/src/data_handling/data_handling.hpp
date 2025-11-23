#ifndef DATA_HANDLING
#define DATA_HANDLING

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp" 
#include "opencv2/core/core.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/highgui/highgui.hpp"

// Function to load the intrinsic parameters of the camera from a .txt file
cv::Mat load_intrinsics(const std::string& filename, const float rescale_factor);

cv::Mat load_distortion_coefficients(const std::string& filename,  const float rescale_factor);

void write_3d_points_to_file(const std::vector<cv::Point3f>& points, const std::string& filename = "data/out/test_points.txt");

#endif /* DATA_HANDLING */