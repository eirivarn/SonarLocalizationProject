# NetApproximationFFT.msg
std_msgs/Header header

# Detected rotations as a list of Float32MultiArray (for matrices)
#std_msgs/Float32MultiArray[] detected_rotations

# Detection status for each ROI
bool[] detection_status

# number of detections
int32 number_of_detections

# Net square oxy-psition (in the image) and z-position (measured distance)
geometry_msgs/Point32[] detected_positions

# Corner positions for all net squares
fft/NetSquare[] net_squares

# net distance, heading and pitch by fitting to plane
float64 plane_distance
float64 plane_heading
float64 plane_pitch

# net distance, heading and pitch by fitting to paraboloid - might be unused
float64 parab_distance
float64 parab_heading
float64 parab_pitch