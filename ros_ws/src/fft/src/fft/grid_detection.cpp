#include "fft.hpp"

#include "RPP.h"
//https://github.com/ngocdaothanh/PoseEstimationForPlanarTarget

std::vector<int> checkgridstructure(std::vector<cv::Point2f> maximas, std::vector<std::vector<double>> *basevector){
    
    std::vector<int> indicatorreconfirm(maximas.size(),0);

    cv::flann::KDTreeIndexParams indexParams(1);

    cv::Mat allpoints = cv::Mat(maximas).reshape(1);
    cv::Mat qpt;

    cv::flann::Index kdTree(allpoints, indexParams);
    int maxPoints=5;
    cv::Mat indices;
    cv::Mat dists;
    indices.create(cv::Size(1,maxPoints), CV_32S);
    dists.create(cv::Size(1,maxPoints), CV_32F);
    
    // Find the two grid-vectors  
    std::vector<std::vector<cv::Point2f>> maindirections;
    std::vector<cv::Point2f> currentdirection;
    cv::Point2f temp_point; 

    for(int i = 0; i< (int)maximas.size(); i++){
 
        cv::Point ref= maximas[i];
    
        qpt = cv::Mat(ref).reshape(1).t();
        qpt.convertTo(qpt, CV_32F);  
        kdTree.knnSearch(qpt, indices, dists, maxPoints);

        // check direction
        std::vector<std::vector<double>> four_points;
        four_points.clear(); 
        for(int k = 1; k< maxPoints; k++){ // 0 is the point itself
            std::vector<double> mydat;
            int myidx=indices.at<int>((int)k);
            //double mydist=dists.at<float>((int)k);
            cv::Point outer= maximas[myidx];
            mydat.clear();
            mydat.push_back(outer.x); //new 0 // index 0  
            mydat.push_back(outer.y); //new 1 // index 1 
            four_points.push_back(mydat);
        }

        // build an accumulation array for appearing orientations
        // unify vectors to to start in lower left corner
        // fourpts contains points around the reference
        //Point2f centerp=ref;
        cv::Point2f endpoint, startpt;

        for(int m = 0; m< (int)four_points.size(); m++){
            
            temp_point = cv::Point(four_points[m][0], four_points[m][1]); // x,y pos

            double dx=cv::norm( ref.x-temp_point.x);
            double dy=cv::norm( ref.y-temp_point.y);
            if(dx>=dy){                         // x is dominant direction
                if(ref.x< temp_point.x){
                    startpt = ref;
                    endpoint = temp_point;
                }
                else{
                    startpt = temp_point;
                    endpoint = ref;
                }
            }
            else{                               // y is dominant direction
                if(ref.y<temp_point.y){
                    startpt = ref;
                    endpoint = temp_point;
                }
                else{
                    startpt=temp_point;
                    endpoint=ref;
                }
            }

            cv::Point2f gridvector = endpoint-startpt;

            //cout << "ref " <<  ref << "  grid  "  <<  gridvector << " From"  << startpt << " to " << endpoint <<  endl;
            
            bool foundexitstingclass=false;
            double mytoleranz = 10.0;
            
            for(int mycls=0; mycls<(int)maindirections.size(); mycls++){
                cv::Point2f representativedir=maindirections[mycls][0];
                if( !foundexitstingclass &&  cv::norm(representativedir - gridvector) < mytoleranz){
                    foundexitstingclass=true;
                    if(indicatorreconfirm[i]==0)indicatorreconfirm[i]=mycls+1;
                    // add gridvector to list
                    maindirections[mycls].push_back(gridvector);
                    // Note the algo below considers the current mean too
                    // Maybe necessary to start from index 1 instead ...
                    cv::Point2f sum = std::accumulate(
                            maindirections[mycls].begin(), maindirections[mycls].end(), // Run from begin to end
                                        cv::Point2f(0.0f,0.0f),       // Initialize with a zero point
                                        std::plus<cv::Point2f>()      // Use addition for each point (default)
                                        );
                    cv::Point2f mean = sum / (double)maindirections[mycls].size();
                    maindirections[mycls][0]=mean;
                }
            }

            if(!foundexitstingclass){ // Add a new class if it is not assigned to existing cluster
                std::vector<cv::Point2f> tmpvec;
                // First push_back is the representativedir!
                tmpvec.push_back(gridvector);
                tmpvec.push_back(gridvector);
                maindirections.push_back(tmpvec);
            }
        }  
    }

    std::vector<std::vector<int>> gatherdat;
    std::vector<int> gathertmp;
        
    for(int i=0; i < (int)maindirections.size(); i++){
        int csize=maindirections[i].size();
        gathertmp.clear();
        gathertmp.push_back(csize);
        gathertmp.push_back(i);
        gatherdat.push_back(gathertmp);
    }

    std::sort (gatherdat.rbegin(), gatherdat.rend());

    /* Simple net-detection approach */
    double sumfirsttwo=0.0;
    double sumrest=0.0;
    for(int i=0; i < (int)gatherdat.size(); i++){
        if(i<2)
            {
            sumfirsttwo+=gatherdat[i][0];
            }
        else
            {
            sumrest+=gatherdat[i][0];
            }
    }

    double fraction = sumfirsttwo/(sumfirsttwo+sumrest);

    if(fraction>0.50){
        NET_DETECTION_STATUS = WELLDETECTED;
    }
    else{
        if(fraction<0.4){
            NET_DETECTION_STATUS = NOTDETECTED;
        }
        else{
            NET_DETECTION_STATUS = MEDIUMDETECTED;
        }
    }
    
    int kidx=gatherdat[0][1];
    int kidxb=gatherdat[1][1];

    basevector->clear();
    
    std::vector<double> vtmp;
    vtmp.clear();
    vtmp.push_back(maindirections[kidx][0].x);
    vtmp.push_back(maindirections[kidx][0].y); 
    basevector->push_back(vtmp);
    vtmp.clear();
    vtmp.push_back(maindirections[kidxb][0].x);
    vtmp.push_back(maindirections[kidxb][0].y); 
    basevector->push_back(vtmp);
    
    return indicatorreconfirm;
}

void compute_orientation_and_distance_RPP(std::vector<cv::Point2d> net_square,cv::Point3f &squareposition, cv::Mat &ext_rotation){

    int numpts=4;
    
    // create ideal net-square model
    double meshx=MESHSIZE;
    double meshy=MESHSIZE;
    double model_data[8] = {-meshx/2.0, meshx/2.0,  meshx/2.0,  -meshx/2.0,
                            meshy/2.0, meshy/2.0, -meshy/2.0,  -meshy/2.0};

    cv::Mat model = cv::Mat::zeros(3, numpts, CV_64F); // 3D points, z is zero
    cv::Mat iprts = cv::Mat::ones(3, numpts, CV_64F); // 2D points, homogenous points
    cv::Mat rotation;
    cv::Mat translation;
    int iterations;
    double obj_err;
    double img_err;

    cv::Mat invCameraMatrix = INTRINSICS.inv();
                        
    for(int i=0; i < numpts; i++) {
        iprts.at<double>(0,i) = net_square[i].x;
        iprts.at<double>(1,i) = net_square[i].y;
    }

    iprts=invCameraMatrix*iprts;
    
    for(int i=0; i < numpts; i++) {
        model.at<double>(0,i) = model_data[i];
        model.at<double>(1,i) = model_data[i+numpts];
    }

    if(!RPP::Rpp(model, iprts, rotation, translation, iterations, obj_err, img_err)) {
        fprintf(stderr, "[ERROR]\tError with RPP\n");
    }

    /* std::cout << "*************************RPP translation******************************" << std::endl;   
    std::cout << translation << std::endl;
    std::cout << "**************************RPP rotation********************************" << std::endl;
    std::cout << rotation << std::endl;
    std::cout << "**********************************************************************" << std::endl; */
    
    ext_rotation = rotation;

    squareposition.x=translation.at<double>(0,0);
    squareposition.y=translation.at<double>(1,0);
    squareposition.z=translation.at<double>(2,0);

    return;   
}

void compute_orientation_and_distance(std::vector<cv::Point2d> net_square,cv::Point3f &squareposition, cv::Mat &ext_rotation){

    std::vector<cv::Point3f> object_pts;
    std::vector<cv::Point2f> image_pts;

    //Mat rvec;
    //Mat tvec;
    cv::Mat rotation;
    cv::Mat world_position;

    cv::Mat distCoeffs(4,1,cv::DataType<double>::type, 0.0);
    cv::Mat rvec(3,1,cv::DataType<double>::type);
    cv::Mat tvec(3,1,cv::DataType<double>::type);
    
    double meshx=MESHSIZE;
    double meshy=MESHSIZE;

    object_pts.push_back(cv::Point3f(-meshx/2.0, meshy/2.0, 0));
    object_pts.push_back(cv::Point3f( meshx/2.0, meshy/2.0, 0));
    object_pts.push_back(cv::Point3f( meshx/2.0, -meshy/2.0, 0));
    object_pts.push_back(cv::Point3f(-meshx/2.0, -meshy/2.0, 0));

    //Corresponding points detected in the image (should have the same order as object points)
    image_pts.push_back(net_square[0]);
    image_pts.push_back(net_square[1]);
    image_pts.push_back(net_square[2]);
    image_pts.push_back(net_square[3]);

    // For undistorted images
    // cv::Mat zero_dist = (cv::Mat_<double>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);
    // cv::Size imageSize(720, 540);
    // cv::Mat K_new = cv::getOptimalNewCameraMatrix(INTRINSICS, DISTORTION, imageSize, 0, imageSize);
    // solvePnP(object_pts, image_pts, K_new, zero_dist, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);


    // For distorted images
    solvePnP(object_pts, image_pts, INTRINSICS, DISTORTION, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    
    //Get rotation matrix
    //Rodrigues(rvec, rotation);  

    //std::cout << "Rotation (rodriges):" << std::endl;
    //std::cout << rvec << std::endl;
    Rodrigues(rvec, rotation);
    //std::cout << "Rotation 3x3:" << std::endl;
    //std::cout << rotation << std::endl;
    //std::cout << "Translation:" << std::endl;
    //std::cout << tvec << std::endl;
    
    ext_rotation = rotation;

    squareposition = cv::Point3f(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0));
}
