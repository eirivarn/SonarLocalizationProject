# Net pose estimation

This code is a modification of the original FFT method proposed by Schellewald et al. in https://www.sciencedirect.com/science/article/pii/S2405896321015305?via%3Dihub.
It utilizes multiple ROIs to compute mutliple ditance estimates from the camera to the net enabeling the generation of 3D sparce points and 2d sparce depth maps.

The program requires monocular images as input and generates a .txt file containing the estimated 3D points (x y z) in camera coordinates in [cm]: 
- x -> right
- y -> down 
- z -> into the image plane
As well as a .csv file containing 2d depth priors (p_y, p_x, depth)
- p_y -> y-pixelcoordinate
- p_x -> x-pixelcoordinate
- depth -> depth value in [m]

## Installation

This code is written and tested on a VM running Ubuntu 20.04. To successfully build the project the installation of the following libraries is required:
- OpenCV (v4.2.0)

## Run Program

After successfully building the program, the executable should be located in the build fulder. To execute the program run ``` ./build/net_pose ``` from the project's root folder. When running the program you can use the following input flags to specify more details:
- ```-i``` to specify the name of a single input image including the datatype(make sure the image is located in the ``` /data/in ``` folder) or set the basename of a image sequence (leave out numbers and .jpg ending) and set -f flag additionally.

        ./build/net_pose -i input.jpg
                
        ./build/net_pose -i input -f path/to/folder
- ```-o``` to specify the name of the output files without datatype (the output is going to be saved in the ``` /data/out ``` folder)

        ./build/net_pose -o test_01
- ```-n``` to specify the number of ROI's that should be used to estimate the net pose (needs to be given in number of rows and columns e.g)

        ./build/net_pose -n 20 15 
- ```-m``` to specify the side length of a single ne square in [cm]

        ./build/net_pose -m 2.0
- ```-s``` Set numbers of images where the intermediate results should be saved (order and amount does not matter).

        ./build/net_pose -s 15 34 46
- ```-f``` to specify the folder where a sequence of images can be found and thus enable processing of a sequence of images

        ./build/net_pose -f path/to/folder  
- ```-h``` to display the help regarding the input flags (execution stops when displaying the help info)

        ./build/net_pose -h  
