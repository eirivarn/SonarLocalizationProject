FROM ros:noetic

# Install system and ROS dependencies
RUN apt update && apt install -y \
    git build-essential mesa-utils libgl1-mesa-dri libgl1-mesa-glx \
    python3-rosdep python3-catkin-tools python3-pip \
    ros-noetic-rqt-image-view \
    ros-noetic-rviz \
    ros-noetic-tf \
    ros-noetic-eigen-conversions \
    libgoogle-glog-dev \
    qt5-qmake qtbase5-dev qtchooser qtbase5-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip3 install --upgrade pip numpy && \
    pip3 install \
    numpy \
    Pillow pandas matplotlib scipy\
    opencv-python opencv-contrib-python \
    torch torchvision torchaudio tensorboard

# Initialize rosdep
RUN rosdep update

# Set working directory inside the container
WORKDIR /home/shared_folder/ros_ws

# Source ROS setup (optional for CMD/ENTRYPOINT layer)
CMD ["bash"]

