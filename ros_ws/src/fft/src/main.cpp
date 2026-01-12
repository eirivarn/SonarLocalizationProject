#include <iostream>
#include <fstream>

#include <vector>

#include <chrono>

#include <signal.h>

// OpenCV
#include "opencv2/opencv.hpp" 
#include "opencv2/core/core.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/highgui/highgui.hpp"

// own code
#include "data_handling/data_handling.hpp"
#include "fft/fft.hpp"
#include "shape_fitting/shape_fitting.hpp"
#include "utils/thread_pool.h"
#include "utils/stopwatch.h"

// ROS
#include <ros/ros.h>
#include "std_msgs/Bool.h"
#include <geometry_msgs/Point32.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud.h>
#include <cv_bridge/cv_bridge.h>
#include "fft/NetApproximationFFT.h" //for .msg file
#include "fft/NetSquare.h" //for .msg file

/// Global Variables
int NET_DETECTION_STATUS = 0;
double MESHSIZE = 7.0;  

cv::Mat INTRINSICS = cv::Mat::eye(3, 3, CV_64F);
cv::Mat DISTORTION = (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);

// cv::Mat INTRINSICS;
// cv::Mat DISTORTION;

cv::Mat input_image;
cv::Size input_image_size;

float rescale_factor = 1.0;
int rows_of_ROIs = 10;
int cols_of_ROIs = 10;
int size_of_ROI = 300;
int border_buffer = 50;

std::string base_folder = "/home/shared_folder/ros_ws/src/integrated-localization-and-mapping-for-uuv/packages/";
std::string in_path = base_folder + "fft/src/data/in/";
std::string out_path = base_folder + "fft/src/data/out/";
std::string in_name = "test.jpg";
std::string out_name = "test";
std::string calibration_path = in_path + "calibration_right_2.txt";
// std::string calibration_path = in_path + "synthetic_data.txt";
// std::string calibration_path = in_path + "calibration_alphasense_cam0.txt";


ThreadPool thread_pool;
Stopwatch stopwatch;

ros::Publisher publisher_fft;
ros::Publisher depth_pointcloud_pub;
ros::Publisher net_square_pub;

std::vector<double> fft_callback_runtimes;
std::string fft_runtime_csv_path = out_path + "fft_callback_runtimes.csv";


// Function to save runtimes to CSV
void saveRuntimesToCSV() {
    std::ofstream file(fft_runtime_csv_path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open FFT runtime CSV file!" << std::endl;
        return;
    }

    file << "FFT_Callback_Runtime_s\n";
    double sum = 0.0;
    for (const auto& rt : fft_callback_runtimes) {
        file << rt << "\n";
        sum += rt;
    }
    if (!fft_callback_runtimes.empty()) {
        file << "Average," << sum / fft_callback_runtimes.size() << "\n";
    }
    file.close();
    std::cout << "[INFO] Saved FFT callback runtimes to " << fft_runtime_csv_path << std::endl;
}

// Function handeling shutdown signal
void mySigintHandler(int sig) {
    std::cout << "[INFO] ROS is shutting down, saving FFT runtimes..." << std::endl;
    saveRuntimesToCSV();
    ros::shutdown(); // actually shut down ROS
}


// Function to visualize detected net squares on the image and publish it
void visualizeNetSquares(cv::Mat& image, const std::vector<std::vector<cv::Point2d>>& net_squares) {
    cv::Mat showimg = image.clone();

    
    if (net_squares.empty()) {
        std::cerr << "[ERROR] ❌ No net squares received for drawing!" << std::endl;
        return;
    }

    for (const auto& square : net_squares) {
        line(showimg, square[0], square[1], cv::Scalar(0,0,255),4); 
        line(showimg, square[1], square[2], cv::Scalar(0,255,255),4); 
        line(showimg, square[2], square[3], cv::Scalar(100,0,255),4);
        line(showimg, square[3], square[0], cv::Scalar(100,255,255),4);
    }

    cv_bridge::CvImage cv_image;
    cv_image.image = showimg;
    cv_image.encoding = "bgr8";
    sensor_msgs::Image ros_image;
    cv_image.toImageMsg(ros_image);

    net_square_pub.publish(ros_image);
}

// Function to rescale a point from original image size to new image size
cv::Point2d rescale_point(const cv::Point2d& point, int original_width, int original_height, int new_width, int new_height) {
    return cv::Point2d(point.x * double(new_width)/original_width, point.y * double(new_height)/original_height);
}

// Function to compute heading and pitch angles from the plane coefficients
std::pair<double, double> get_angles_to_net(const Eigen::Vector3d& coefficients) {
    double a = coefficients[0];
    double b = coefficients[1];
    
    // Compute the pitch angle (angle of the projection of the normal vector on the yz-plane)
    double pitch = atan2(-b, 1);  // Angle from the z-axis
    
    // Compute the heading angle (angle of the projection of the normal vector on the xz-plane)
    double heading = atan2(-a, 1);  // Angle from the z-axis

    return std::make_pair(heading, pitch);
}

// Function to convert std::vector<cv::Point3f> to std::vector<Eigen::Vector3d>
std::vector<Eigen::Vector3d> convertToEigen(const std::vector<cv::Point3f>& points) {
    std::vector<Eigen::Vector3d> eigen_points;  // Create a vector of Eigen::Vector3d
    eigen_points.reserve(points.size());        // Reserve space for efficiency

    // Iterate over the input points and fill the Eigen vector
    for (const auto& point : points) {
        eigen_points.emplace_back(point.x, point.y, point.z);  // Convert cv::Point3f to Eigen::Vector3d
    }

    return eigen_points;  // Return the vector of Eigen::Vector3d
}

// Function to convert cv::Point2d to geometry_msgs::Point32
geometry_msgs::Point32 point2dToPoint32(const cv::Point2d& point) {
    geometry_msgs::Point32 ros_point;
    ros_point.x = point.x;
    ros_point.y = point.y;
    ros_point.z = 0.0;  // Since it's a 2D point, set z to 0
    return ros_point;
}


int perform_fft_on_image(cv::Mat& input_image, const ros::Time& stamp, bool save_prior) {
  
    if (input_image.empty()) {
        std::cout << "[ERROR] 🚨 Input image is EMPTY!" << std::endl;
        return 2;
    }

    // Print input image size
    std::cout << "[DEBUG] Input image size: " << input_image.cols << " x " << input_image.rows << std::endl;


    cv::resize(input_image, input_image, cv::Size(), rescale_factor,
                rescale_factor);

    input_image_size = input_image.size();

    // convert image to gray scale for FTT
    cv::Mat image_gray;
    cv::cvtColor(input_image, image_gray, cv::COLOR_BGR2GRAY);

    int available_space_x = input_image.cols - 2 * border_buffer - size_of_ROI;
    int available_space_y = input_image.rows - 2 * border_buffer - size_of_ROI;

    int step_x = available_space_x / (cols_of_ROIs - 1);
    int step_y = available_space_y / (rows_of_ROIs - 1);

    int detection_counter = 0;

    std::vector<cv::Point2d> center_points;
    std::vector<cv::Point3f> detected_positions;
    std::vector<cv::Mat> detected_rotations;
    std::vector<double> detected_distances;
    std::vector<int> successes;
    std::vector<std::vector<cv::Point2d>> detected_net_squares; // new

    const size_t num_ROIs = rows_of_ROIs * cols_of_ROIs;
    successes.resize(num_ROIs);
    center_points.resize(num_ROIs);
    detected_positions.resize(num_ROIs);
    detected_rotations.resize(num_ROIs);
    detected_distances.resize(num_ROIs);
    detected_net_squares.resize(num_ROIs); // new

    int counter = 0;
    for (int row = 0; row < rows_of_ROIs; ++row) {
        for (int col = 0; col < cols_of_ROIs; ++col) {
        auto &success = successes[counter];
        auto &center_point = center_points[counter];
        auto &detected_position = detected_positions[counter];
        auto &detected_rotation = detected_rotations[counter];
        auto &detected_distance = detected_distances[counter];
        auto &detected_net_square = detected_net_squares[counter];
        thread_pool.add_task([col, step_x, row, step_y, &image_gray, &success,
                                &center_point, &detected_position,
                                &detected_rotation, &detected_distance, &detected_net_square]() {

            success = 0;
            int x = border_buffer + col * step_x;
            int y = border_buffer + row * step_y;
            cv::Point2d center_point_tmp =
                cv::Point(x + size_of_ROI / 2, y + size_of_ROI / 2);
            cv::Rect roi_rect(x, y, size_of_ROI, size_of_ROI);
            cv::Mat roi = image_gray(roi_rect);

            std::vector<cv::Point2d> net_square = dft_test(roi);

            if (net_square.empty()) {
                // std::cout << "[WARNING] ⚠️ FFT found NO net squares in this frame!" << std::endl;
                return;
            }

            if (NET_DETECTION_STATUS != NOTDETECTED) {
            for (size_t ii = 0; ii < net_square.size(); ii++) {
                net_square[ii] += center_point_tmp;
            }
            } else {
            //           std::cout << "[INFO]\tNet not detected, trying next ROI!"
            //           << std::endl;
                return;
            }

            cv::Point3f square_position;
            cv::Mat square_rotation;

            compute_orientation_and_distance(net_square, square_position,
                                            square_rotation);

            if (square_position.z < 0.1) {
                return;
            }                                         

            success = 1;
            center_point = center_point_tmp;
            detected_position = square_position;
            detected_rotation = square_rotation;
            detected_distance = square_position.z;
            detected_net_square = net_square;
        });

        ++counter;
        }
    }
    thread_pool.wait_all();

    // Filter out unsuccessful results
    auto filter = [&mask = std::as_const(successes)](auto &vector) {
        assert(vector.size() == mask.size());
        auto it = mask.begin();
        vector.erase(std::remove_if(vector.begin(), vector.end(),
                                    [&](auto) { return !(*it++); }),
                    vector.end());
    };
    filter(center_points);
    filter(detected_positions);
    filter(detected_rotations);
    filter(detected_distances);
    filter(detected_net_squares);
    detection_counter =
        static_cast<int>(std::count(successes.begin(), successes.end(), 1));

    // Draw and save the detected net squares in the image
    visualizeNetSquares(input_image, detected_net_squares);

    // Publish results if any detections were made

    if (!detected_positions.empty()) {



        fft::NetApproximationFFT msg;
        
        msg.detected_positions.clear();

        // Run your functions here
        const std::vector<Eigen::Vector3d>& eigen_points = convertToEigen(detected_positions);  // convert to right format
        Eigen::Vector3d plane_coefficients = fit_plane(eigen_points); 
        std::pair<double, double> plane_angles = get_angles_to_net(plane_coefficients);

        double plane_distance = plane_coefficients.z()*0.01;   // get distance to plane in [cm] -> *0.01 to [m]
        double plane_heading  = plane_angles.first  * 180.0 / M_PI;
        double plane_pitch    = plane_angles.second * 180.0 / M_PI;

        std::cout << "[FFT] " << "Squares found: " << detection_counter << "/" << num_ROIs <<  " | Dist: " << plane_distance << " [m] | " << "Yaw: " << plane_heading << " [deg] | " << "Pitch: "    << plane_pitch    << " [deg]\n";

        /*** Publish the data ***/
    msg.header.stamp = stamp; // Use the timestamp from the image message
        msg.number_of_detections = detection_counter;
        msg.plane_distance  = plane_distance;
        msg.plane_heading   = plane_heading;
        msg.plane_pitch     = plane_pitch;
        msg.parab_distance  = 0.0; // paraboloid_distance;
        msg.parab_heading   = 0.0; // paraboloid_heading;
        msg.parab_pitch     = 0.0; // paraboloid_pitch;
        
        // Iterate over detected net squares
        for (const auto& net_square : detected_net_squares) {
            // Ensure each net square has exactly 4 points
            if (net_square.size() == 4) {
                fft::NetSquare square;  // Create a NetSquare message
                for (int i = 0; i < 4; ++i) {
                    square.points[i].x = net_square[i].x;  // Correctly access the points
                    square.points[i].y = net_square[i].y;
                    square.points[i].z = 0.0; // Set z to 0 for 2D points (or use actual z if available)
                }
                msg.net_squares.push_back(square); // Add the current square to the message
            }
        }
        
        // Convert each cv::Point3f to geometry_msgs/Point32 and add it to the point cloud
        for (const auto& position : detected_positions) {
            geometry_msgs::Point32 point;
            point.x = position.x;
            point.y = position.y;
            point.z = position.z;
            msg.detected_positions.push_back(point);
        }

        // Create a PointCloud message
        sensor_msgs::PointCloud pointcloud_msg;
    pointcloud_msg.header.stamp = stamp;
        pointcloud_msg.points.resize(center_points.size());
        for (size_t i = 0; i < center_points.size(); ++i) {
            cv::Point2d rescaled_point = rescale_point(center_points[i], input_image_size.width, input_image_size.height, 320, 240);
            pointcloud_msg.points[i].x = rescaled_point.y;
            pointcloud_msg.points[i].y = rescaled_point.x;
            pointcloud_msg.points[i].z = detected_distances[i] / 100.0f; // Convert distance to meters
        }
        depth_pointcloud_pub.publish(pointcloud_msg); // Publish the point cloud

        
        publisher_fft.publish(msg);

    }else {
        std::cout << "No detections." << std::endl;

    }
    return 0;
}

// Image callback for raw sensor_msgs/Image (kept for compatibility)
void imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    auto start = std::chrono::high_resolution_clock::now();

    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        // Non undistorted version
        perform_fft_on_image(cv_ptr->image, msg->header.stamp, false);

    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    } catch (const cv::Exception& e) {
        ROS_ERROR("OpenCV exception: %s", e.what());
    }

    // Measure and store runtime
    auto end = std::chrono::high_resolution_clock::now();
    fft_callback_runtimes.push_back(std::chrono::duration<double>(end-start).count());
}

// Callback for sensor_msgs/CompressedImage
void compressedImageCallback(const sensor_msgs::CompressedImage::ConstPtr& msg) {
    auto start = std::chrono::high_resolution_clock::now();

    try {
        // Decode compressed image data into OpenCV Mat
        std::vector<uchar> data(msg->data.begin(), msg->data.end());
        cv::Mat decoded = cv::imdecode(data, cv::IMREAD_COLOR);
        if (decoded.empty()) {
            ROS_ERROR("Failed to decode compressed image");
            return;
        }

        perform_fft_on_image(decoded, msg->header.stamp, false);

    } catch (const cv::Exception& e) {
        ROS_ERROR("OpenCV exception decoding compressed image: %s", e.what());
        return;
    }

    // Measure and store runtime
    auto end = std::chrono::high_resolution_clock::now();
    fft_callback_runtimes.push_back(std::chrono::duration<double>(end-start).count());
}



int main( int argc, char** argv ){

    std::string input_path;
    std::string file_name;
    std::string output_name;
    std::string folder_name;
    bool process_sequence = false;
    bool save_priors = false;

    in_path = "/home/shared_folder/ros_ws/src/integrated-localization-and-mapping-for-uuv/packages/fft/src/data/in/";
    out_path = "/home/shared_folder/ros_ws/src/integrated-localization-and-mapping-for-uuv/packages/fft/src/data/out/";
    in_name = "test.png";

    input_image = cv::imread(in_path + in_name);

    // Data loading:
    // INTRINSICS = load_intrinsics(calibration_path, rescale_factor);
    // DISTORTION = load_distortion_coefficients(calibration_path, rescale_factor);
    

    // Initialize the ROS node
    std::string type = "fft_node";
    ros::init(argc, argv, type);
    ros::NodeHandle nh;

    ros::Time::init();

    publisher_fft = nh.advertise<fft::NetApproximationFFT>("fft_data", 10, true);
    depth_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud>("fft/depth_pointcloud", 10, true);
    net_square_pub = nh.advertise<sensor_msgs::Image>("fft/net_squares_image", 10, true);
    
    nh.getParam("/fft/cols_of_ROIs", cols_of_ROIs);
    nh.getParam("/fft/size_of_ROI", size_of_ROI);
    nh.getParam("/fft/rows_of_ROIs", rows_of_ROIs);
    nh.getParam("/fft/border_buffer", border_buffer);
    nh.getParam("/fft/mesh_grid_size", MESHSIZE);
    nh.getParam("/camera/fx", INTRINSICS.at<double>(0,0));
    nh.getParam("/camera/fy", INTRINSICS.at<double>(1,1));
    nh.getParam("/camera/cx", INTRINSICS.at<double>(0,2));
    nh.getParam("/camera/cy", INTRINSICS.at<double>(1,2));
    nh.getParam("/camera/k1", DISTORTION.at<double>(0,0));
    nh.getParam("/camera/k2", DISTORTION.at<double>(0,1));
    nh.getParam("/camera/p1", DISTORTION.at<double>(0,2));
    nh.getParam("/camera/p2", DISTORTION.at<double>(0,3));
    nh.getParam("/camera/k3", DISTORTION.at<double>(0,4));

    // Subscribe to the image topic
    std::string camera_topic;
    nh.getParam("/camera/image_topic", camera_topic);

    std::cout << "[INFO] Subscribing to camera topic: " << camera_topic << std::endl;
    // Subscribe to compressed image topic (many recordings use CompressedImage). If your camera publishes raw Image, use imageCallback instead.
    ros::Subscriber image_sub = nh.subscribe(camera_topic, 10, compressedImageCallback); // subscribe to CompressedImage
    // If you prefer to subscribe to raw Image, uncomment the next line and comment the line above:
    // ros::Subscriber image_sub = nh.subscribe(camera_topic, 10, imageCallback); //For raw sensor_msgs::Image
    // ros::Subscriber image_sub = nh.subscribe("/alphasense_driver_ros/cam0", 10, imageCallback); //For stereo camera

    signal(SIGINT, mySigintHandler);
    ros::spin();

    return 0;
}