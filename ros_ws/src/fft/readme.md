# Gstreamer pipeline for retrieving BlueROV2 video stream

A ROS package for retrieving video stream from the Bluerov2 onboard camera.

### Prerequisites 

GStreamer is currently necessary for obtaining the BlueROV2 camera images. Guide for installation and testing that the install was correct can be found in the root folder.

### Build instructions

* Inside the ROS command prompt, run: 

<pre><code> catkin_make </code></pre>

* Source the files with:

<pre><code> devel\setup.bat </code></pre>

* <b>Test gstreamer</b>: 
    In the ROS cmd prompt run 
        <pre><code> roslaunch gstproject test_pipeline.launch </code></pre>
	It may take some time for the videostream to load



