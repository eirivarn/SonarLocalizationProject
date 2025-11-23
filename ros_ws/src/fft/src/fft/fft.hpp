#ifndef FFT
#define FFT

#include <vector>
#include <string>
#include <iostream>
#include <numeric>

#include "opencv2/opencv.hpp" 
#include "opencv2/core/core.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/highgui/highgui.hpp"

extern int NET_DETECTION_STATUS;
extern double MESHSIZE;
extern cv::Mat INTRINSICS;
extern cv::Mat DISTORTION;

extern std::string out_path;
extern std::string out_name;
extern std::string ROI_number;

enum {WELLDETECTED,MEDIUMDETECTED,NOTDETECTED};

// FFT functions (fft.cpp)
std::vector<cv::Point2d> dft_test(cv::Mat image);

std::vector<std::vector<double>> cs_fft_LocalMaxIdea(const cv::Mat &magnitude_image);

// Local Maxima detection (localmaxima.cpp)
void localMaxima(cv::Mat src,cv::Mat &dst,int squareSize);

void non_maxima_suppression(const cv::Mat& inimage, cv::Mat& mask,int GaussKernel, bool remove_plateaus);

std::vector<cv::Point> GetLocalMaxima(const cv::Mat Src,int MatchingSize, int Threshold, int GaussKernel);

// Grid detection (grid_detection.cpp)
std::vector<int> checkgridstructure(std::vector<cv::Point2f> maximas, std::vector<std::vector<double>> *basevector);

void compute_orientation_and_distance_RPP(std::vector<cv::Point2d> net_square,cv::Point3f &squareposition, cv::Mat &ext_rotation);

void compute_orientation_and_distance(std::vector<cv::Point2d> net_square,cv::Point3f &squareposition, cv::Mat &ext_rotation);

#endif /* FFT */