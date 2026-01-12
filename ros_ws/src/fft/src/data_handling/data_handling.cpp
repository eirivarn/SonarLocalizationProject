#include "data_handling.hpp"

// Function to load the intrinsic parameters of the camera from a .txt file
cv::Mat load_intrinsics(const std::string& filename, const float rescale_factor) {
    //std::cout << "[INFO]\tLoad camera intrinsics matrix..." << std::endl;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return cv::Mat();
    }

    std::string line;
    std::getline(file, line); // Skip the first line (Reprojection Error)
    std::getline(file, line); // Skip the second line (Camera Matrix)

    double values[9];
    for (int i = 0; i < 3; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        iss >> values[i * 3] >> values[i * 3 + 1] >> values[i * 3 + 2];
    }

    cv::Mat K = (cv::Mat_<double>(3, 3) << values[0], values[1], values[2],
                                               values[3], values[4], values[5],
                                               values[6], values[7], values[8]);

    // Rescale the intrinsic matrix
    K.at<double>(0, 0) *= rescale_factor; // fx
    K.at<double>(1, 1) *= rescale_factor; // fy
    K.at<double>(0, 2) *= rescale_factor; // cx
    K.at<double>(1, 2) *= rescale_factor; // cy

    //std::cout << "      \t...done!" << std::endl;
    //std::cout << "Camera Matrix Loaded:\n" << Qtmp << std::endl; // Visual inspection

    return K;
}

// Function to load distortion coefficients from a .txt file
cv::Mat load_distortion_coefficients(const std::string& filename, const float rescale_factor) {
    //std::cout << "[INFO]\tLoad distortion coefficients..." << std::endl;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return cv::Mat();
    }

    std::string line;
    for (int i = 0; i < 7; ++i) {
        std::getline(file, line); // Skip lines until the distortion coefficients line
    }

    double values[5];
    std::istringstream iss(line);
    for (int i = 0; i < 5; ++i) {
        iss >> values[i];
    }

    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << values[0], values[1], values[2], values[3], values[4]);

    // Rescale the distortion coefficients
    distCoeffs *= rescale_factor;

    //std::cout << "      \t...done!" << std::endl;
    //std::cout << "Distortion Coefficients Loaded:\n" << distCoeffs << std::endl; // Visual inspection

    return distCoeffs;
}

// Function that writes a list of vectors as separate lines to a .txt file
// Input: a vector of 3d points that should be saved to a file, the name of the output file
void write_3d_points_to_file(const std::vector<cv::Point3f>& points, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "[ERROR]\tFailed to open the file: " << filename << std::endl;
        return;
    }

    for (const auto& point : points) {
        file << point.x << " " << point.y << " " << point.z << std::endl;
    }

    file.close();

    if (!file) {
        std::cerr << "[ERROR]\tFailed to write to the file: " << filename << std::endl;
    } 
    /* else {
        std::cout << "[INFO]\t3D points successfully written to: " << filename << std::endl;
    } */
}