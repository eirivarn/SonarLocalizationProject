#!/usr/bin/env python3


# ROS imports
import rospy
# import tf.transformations
import message_filters
from fft.msg import NetApproximationFFT  # Replace with actual FFT message type
from sensor_msgs.msg import Image
from relative_pose.msg import RelativePose
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
import tf2_ros
from geometry_msgs.msg import TransformStamped
from relative_pose.msg import PlaneApproximation

# Python imports
import numpy as np
import math
import signal
import matplotlib.pyplot as plt
import sys
import cv2
import os
import rospkg
import time
import csv



class RelativePoseNode:
    def __init__(self):
        rospy.init_node("relative_pose_node", anonymous=True)

        # Initialize parameters
        self.fft_topic = rospy.get_param("/relative_pose/fft_topic", "/fft_data_2")
        self.relative_pose_fft_topic = rospy.get_param("/relative_pose/relative_pose_topic", "/relative_pose/fft")
        self.plane_approx_fft_topic = rospy.get_param("/relative_pose/plane_approx_fft_topic", "/relative_pose/plane_approximation_fft")
        self.use_fft = rospy.get_param("/relative_pose/use_fft", True)
        
        self.tru_depth_topic = rospy.get_param("/relative_pose/tru_depth_topic", "/uw_depth/depth_image")
        self.relative_pose_tru_depth_topic = rospy.get_param("/relative_pose/relative_pose_tru_depth_topic", "/relative_pose/tru_depth")
        self.use_tru_depth = rospy.get_param("/relative_pose/use_tru_depth", False)


        self.fx = rospy.get_param("/camera/fx", 882)
        self.fy = rospy.get_param("/camera/fy", 821)
        self.cx = rospy.get_param("/camera/cx", 680)
        self.cy = rospy.get_param("/camera/cy", 360)

        self.camera_width = rospy.get_param("/camera/width", 1280)
        self.camera_height = rospy.get_param("/camera/height", 720)
        
         # Camera intrinsic matrix
        self.camera_parameters = np.array([[self.fx, 0, self.cx],
                                           [0, self.fy, self.cy],
                                           [0, 0, 1]], dtype=np.float32)

        
        package_path = rospkg.RosPack().get_path('relative_pose')

        # Initialize runtime trackers
        self.fft_runtimes = []
        self.tru_depth_runtimes = []
        
        # Subscribers and Publishers
        if self.use_fft:
            fft_sub = message_filters.Subscriber(self.fft_topic, NetApproximationFFT)
            fft_sub.registerCallback(self.fft_callback)
            self.fft_pose_pub = rospy.Publisher(self.relative_pose_fft_topic, RelativePose, queue_size=1)
            self.plane_approx_fft_pub = rospy.Publisher(self.plane_approx_fft_topic, PlaneApproximation, queue_size=1)
            #create a clean csv file
            self.fft_csv_path = os.path.join(package_path, "output", "relative_pose_fft.csv")
            with open(self.fft_csv_path, 'w') as f:
                f.write("time,distance,heading,pitch\n")
            # CSV for runtimes
            self.fft_runtime_csv_path = os.path.join(package_path, "output", "fft_callback_runtime.csv")

        if self.use_tru_depth:
            self.bridge = CvBridge()
            tru_depth_sub = message_filters.Subscriber(self.tru_depth_topic, Image)
            tru_depth_sub.registerCallback(self.tru_depth_callback)
            self.tru_depth_pose_pub = rospy.Publisher(self.relative_pose_tru_depth_topic, RelativePose, queue_size=1)
            #create a clean csv file
            self.tru_depth_csv_path = os.path.join(package_path, "output", "relative_pose_tru_depth.csv")
            with open(self.tru_depth_csv_path, 'w') as f:
                f.write("time,distance,heading,pitch\n")
            # CSV for runtimes
            self.tru_depth_runtime_csv_path = os.path.join(package_path, "output", "tru_depth_callback_runtime.csv")


        rospy.on_shutdown(self.save_runtimes_to_csv)

        rospy.spin()
        
    def fft_callback(self, msg):

        fft_start_time = time.time()  # Start runtime tracking

        rospy.loginfo("FFT message received")
        # Extract points from the FFT message
        points = np.array([[point.x, point.y, point.z] for point in msg.detected_positions])

        # Fit a paraboloid and plane to the points
        parab_coefficients = fit_paraboloid(points)
        plane_coefficients = fit_plane_lsq(points)

        distance = get_dist_to_center(parab_coefficients)
        heading, pitch = get_angles_to_net(plane_coefficients)

        # Create a RelativePose message
        relative_pose_msg = RelativePose()
        relative_pose_msg.header = msg.header
        relative_pose_msg.relative_net_distance = distance
        relative_pose_msg.relative_net_heading = heading
        relative_pose_msg.relative_net_pitch = pitch

        # Publish the message
        self.fft_pose_pub.publish(relative_pose_msg)
        print(f"FFT Distance: {distance}, Heading: {heading}, Pitch: {pitch}")


        # Create and publish PlaneApproximation message
        plane_approx_msg = PlaneApproximation()
        plane_approx_msg.header = msg.header
        plane_approx_msg.NormalDVL = [-1, plane_coefficients[0], plane_coefficients[1]]  # Normal in Camera should be [a, b, -1], but is transformed to NED here
        plane_approx_msg.NetDistance = distance/100.0
        plane_approx_msg.NetHeading = heading
        plane_approx_msg.NetPitch = pitch
        plane_approx_msg.NetLock = 1.0

        self.plane_approx_fft_pub.publish(plane_approx_msg)


        # Save the relative pose to a CSV file
        save_relative_pose_to_csv(self.fft_csv_path, msg.header.stamp.to_sec(), distance, heading, pitch)

        # End runtime tracking
        self.fft_runtimes.append(time.time() - fft_start_time)


    def tru_depth_callback(self, msg):

        tru_depth_start_time = time.time()  # Start runtime tracking

        rospy.loginfo("Depth image received")
        # Process the image to depth map
        depth_map = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        depth_map = depth_map * 100 # Convert to cm

        # distance = extract_center_mean(depth_map)
        # point_cloud = depth_to_point_cloud(depth_map, self.camera_parameters, num_points=200)
        # plane_coefficients = fit_plane_lsq(point_cloud)
        # a, b, c = plane_coefficients
        # distance = abs(c)
        distance = extract_center_mean(depth_map, region_size=100)
        heading, pitch = extract_angle(depth_map, self.camera_parameters, original_size=(self.camera_width, self.camera_height))

        # Create a RelativePose message
        relative_pose_msg = RelativePose()
        relative_pose_msg.header = msg.header
        relative_pose_msg.relative_net_distance = distance
        relative_pose_msg.relative_net_heading = heading
        relative_pose_msg.relative_net_pitch = pitch

        # Publish the message
        self.tru_depth_pose_pub.publish(relative_pose_msg)

        # Save the relative pose to a CSV file
        save_relative_pose_to_csv(self.tru_depth_csv_path, msg.header.stamp.to_sec(), distance, heading, pitch)

        # End runtime tracking
        self.tru_depth_runtimes.append(time.time() - tru_depth_start_time)



    def save_runtimes_to_csv(self):
        # FFT runtimes
        if self.fft_runtimes:
            avg_fft = sum(self.fft_runtimes) / len(self.fft_runtimes)
            std_fft = (sum((x - avg_fft) ** 2 for x in self.fft_runtimes) / len(self.fft_runtimes)) ** 0.5
            with open(self.fft_runtime_csv_path, 'w') as f:
                f.write("FFT Callback Runtime (s)\n")
                for rt in self.fft_runtimes:
                    f.write(f"{rt}\n")
                f.write(f"\nAverage,{avg_fft},Std,{std_fft}\n")
            rospy.loginfo(f"Saved FFT runtimes to {self.fft_runtime_csv_path}")

        # Tru Depth runtimes
        if self.tru_depth_runtimes:
            avg_tru = sum(self.tru_depth_runtimes) / len(self.tru_depth_runtimes)
            std_tru = (sum((x - avg_tru) ** 2 for x in self.tru_depth_runtimes) / len(self.tru_depth_runtimes)) ** 0.5
            with open(self.tru_depth_runtime_csv_path, 'w') as f:
                f.write("Tru Depth Callback Runtime (s)\n")
                for rt in self.tru_depth_runtimes:
                    f.write(f"{rt}\n")
                f.write(f"\nAverage,{avg_tru},Std,{std_tru}\n")
            rospy.loginfo(f"Saved Tru Depth runtimes to {self.tru_depth_runtime_csv_path}")



# Function to fit a 3D plane
def fit_plane_lsq(points):
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    # Create the design matrix for the plane ax + by + c = z
    A = np.c_[X, Y, np.ones_like(X)]
    # Solve the least squares problem
    C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    
    return C



def fit_plane_ransac(points, sigma=7.5, k =3.0, threshold=None, iterations=2000, seed=None):
    """
    Fit a plane z = a x + b y + c to 3D points using RANSAC.

    Args:
        points: (N,3) ndarray of 3D points.
        threshold: inlier distance threshold. If None, estimated from spread.
        iterations: number of RANSAC iterations.
        seed: RNG seed.

    Returns:
        C: array([a, b, c]) defining plane z = a x + b y + c
    """
    rng = np.random.default_rng(seed)
    pts = np.asarray(points)
    N = len(pts)
    if N < 3:
        raise ValueError("Need at least 3 points.")

        # --- Threshold selection ---
    if sigma is not None:
        threshold = k * sigma
    else:
        mad = np.median(np.abs(pts - np.median(pts, axis=0)), axis=0)
        threshold = max(1e-6, 3.0 * np.max(mad))

    # Print out a point too see the scale
    print(f"Example point: {pts[0]}, threshold: {threshold}")

    best_inliers = 0
    best_plane = None

    for _ in range(iterations):
        # sample 3 non-collinear points
        idx = rng.choice(N, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        v1, v2 = p1 - p0, p2 - p0
        n = np.cross(v1, v2)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        d = -np.dot(n, p0)

        # orthogonal distances
        dist = np.abs(pts @ n + d)
        inliers = dist < threshold
        n_inliers = np.count_nonzero(inliers)

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_plane = (n, d, inliers)

    if best_plane is None:
        raise RuntimeError("No plane found.")

    # refine with all inliers using least squares on z = a x + b y + c
    _, _, inliers = best_plane
    inlier_pts = pts[inliers]
    X = inlier_pts[:, 0]
    Y = inlier_pts[:, 1]
    Z = inlier_pts[:, 2]
    A = np.c_[X, Y, np.ones_like(X)]
    C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)

    return C


# Function to fit a 3D paraboloid
def fit_paraboloid(points):
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    # Create the design matrix for the paraboloid ax^2 + by^2 + cxy + dx + ey + f = z
    A = np.c_[X**2, Y**2, X*Y, X, Y, np.ones_like(X)]
    # Solve the least squares problem
    C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    return C

def get_dist_to_center(coefficients):
    # Assuming the coefficients are in the order: [a, b, c, d, e, f]
    _, _, _, _, _, f = coefficients
    return f


def get_angles_to_net(plane_coefficients):
    a, b, c = plane_coefficients # -> normal vector [a,b,-1] -> inverted [-a, -b, 1]
 
    # Compute the pitch angle (angle of the projection of the normal vector on the yz-plane)
    pitch = np.arctan2(-b, 1)        # Angle from the z-axis
    
    # Compute the heading angle (angle of the projection of the normal vector on the xz-plane)
    heading = np.arctan2(-a, 1)      # Angle from the z-axis
    
    return heading, pitch

# Function to extract mean value of the center region
def extract_center_mean(depth_map, region_size=100):

    if depth_map is None:
        return None

    h, w = depth_map.shape
    #print(f"Shape of depth map: {depth_map.shape}")
    center_x, center_y = w // 2, h // 2
    half_size = region_size // 2
    
    # Define the center region
    center_region = depth_map[center_y-half_size:center_y+half_size, center_x-half_size:center_x+half_size]

    # Calculate the mean value
    mean_value = np.mean(center_region)
    
    return mean_value

# Function to extract heading and pitch of a plane fit onto a depth map
def extract_angle(depth_map, camera_parameters, original_size=(1280, 720)):

    if depth_map is None:
        return [None, None]

    plane_coefficients = extract_plane_from_depth(depth_map, camera_parameters, original_size=original_size)

    # Calculate the angles

    heading, pitch = get_angles_to_net(plane_coefficients)

    return heading, pitch
    

def extract_plane_from_depth(depth_map, camera_parameters, original_size=(1280, 720)):
    # Step 1: Convert depth map to 3D point cloud
    point_cloud =depth_to_point_cloud(depth_map, camera_parameters, original_size=original_size)

    # Step 2: Fit a plane to the point cloud
    plane_coefficients = fit_plane_lsq(point_cloud)
    
    return plane_coefficients

def depth_to_point_cloud(depth_map, camera_parameters, num_points=200, original_size=(1280, 720)):
    # Camera intrinsics (corresponding to the original image size)

    f_x = camera_parameters[0, 0]
    f_y = camera_parameters[1, 1]
    c_x = camera_parameters[0, 2]
    c_y = camera_parameters[1, 2]

    # Rescale the depth map to the original size
    depth_map_rescaled = cv2.resize(depth_map, original_size, interpolation=cv2.INTER_LINEAR)

    h, w = depth_map_rescaled.shape
    #print(f"width: {w}, height: {h}.")
    j, i = np.indices((h, w))

    # Flatten the indices and depth map
    indices_flat = np.arange(h * w)
    np.random.shuffle(indices_flat)
    indices_flat = indices_flat[:num_points]
    
    # Retrieve the randomly sampled points
    i_sampled = i.flatten()[indices_flat]
    j_sampled = j.flatten()[indices_flat]
    Z_sampled = depth_map_rescaled[j_sampled, i_sampled] # acces image coordinates using [y, x]!
    
    # Calculate X, Y coordinates
    X = (i_sampled - c_x) * Z_sampled / f_x
    Y = (j_sampled - c_y) * Z_sampled / f_y

    # Stack X, Y, Z coordinates
    points = np.column_stack((X, Y, Z_sampled))

    return points

# Function to save relative pose data to a csv file
def save_relative_pose_to_csv(filename, time, distance, heading, pitch):
    with open(filename, 'a') as f:
        f.write(f"{time}, {distance}, {heading}, {pitch}\n")


def main():
    node = RelativePoseNode()

if __name__ == '__main__':
    main()
