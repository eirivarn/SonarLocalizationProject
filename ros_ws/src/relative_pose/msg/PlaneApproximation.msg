# Plane approximation message
# Contains plane normal (from DVL/camera), distance and net pose info
std_msgs/Header header

# Normal vector in DVL/camera frame (x, y, z)
float32[3] NormalDVL

# Distance to net in meters
float32 NetDistance

# Net heading and pitch (radians)
float32 NetHeading
float32 NetPitch

# Lock/quality indicator (0.0 - 1.0)
float32 NetLock
