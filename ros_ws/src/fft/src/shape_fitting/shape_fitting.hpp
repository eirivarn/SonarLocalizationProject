#ifndef SHAPE_FITTING
#define SHAPE_FITTING

#include <vector>

#include <Eigen/Dense>

//#include "opencv2/opencv.hpp" 
//#include "opencv2/core/core.hpp" 
//#include "opencv2/imgproc/imgproc.hpp" 
//#include "opencv2/highgui/highgui.hpp"

// Function that fits a plane to a vector of 3D points
Eigen::Vector3d fit_plane(const std::vector<Eigen::Vector3d>& points);

// Function that fits a paraboloid to a vector of 3D points
Eigen::VectorXd fit_paraboloid(const std::vector<Eigen::Vector3d>& points);

#endif /* SHAPE_FITTING */