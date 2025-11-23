#include "fft.hpp"

std::vector<cv::Point2d> dft_test(cv::Mat image){

    std::vector<cv::Point2d> netquadri;
    cv::Mat padded;

    int m = cv::getOptimalDFTSize( image.rows );
    int n = cv::getOptimalDFTSize( image.cols ); // on the border add zero values

    cv::copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};

    cv::Mat complex_image;

    cv::merge(planes, 2, complex_image);

    cv::dft(complex_image, complex_image);

    cv::Mat cv_dft = complex_image.clone();

    cv::split(complex_image, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::magnitude(planes[0], planes[1], planes[0]);     // planes[0] = magnitude
    cv::Mat magnitude_image = planes[0];

    magnitude_image += cv::Scalar::all(1);
    log(magnitude_image, magnitude_image);

    // crop the spectrum, if it has an odd number of rows or columns
    magnitude_image = magnitude_image(cv::Rect(0, 0, magnitude_image.cols & -2, magnitude_image.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magnitude_image.cols/2;
    int cy = magnitude_image.rows/2;

    cv::Mat q0(magnitude_image, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magnitude_image, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magnitude_image, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magnitude_image, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;                       // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    tmp.release();

    // return basis-vectors of the grid
    std::vector<std::vector<double>> basevectors;
    basevectors = cs_fft_LocalMaxIdea(magnitude_image);

    if (basevectors.empty()){
        return netquadri;
    }

    cv::Point ref(0,0);

    std::vector<std::vector<double> > fftmaxpos;
    std::vector<double> postmp;

    cv::Mat filtermask = cv::Mat::zeros(magnitude_image.size(), CV_8U);

    std::vector<double> va=basevectors[0];
    std::vector<double> vb=basevectors[1];

    
    double angle_a_rad = atan2(va[1],va[0]) + M_PI/2.0; // angle in fft + pi/2 (+90degrees)
    double angle_a_deg = angle_a_rad * 180.0 / M_PI; 
    
    double dist_a=sqrt(vb[0]*vb[0]+vb[1]*vb[1]);
    double length_a= (double)n/dist_a;

    cv::Mat rot_a = cv::getRotationMatrix2D(cv::Point(0.0,0.0), -angle_a_deg, 1.0); // requires homogeneous coordinates
    cv::Point3d l_a= cv::Point3f(length_a,0.0,1.0);
    
    cv::Mat a_spat= rot_a * cv::Mat(l_a); // Spatial base vector/point from freq-basevector 

    cv::Point2d mya = cv::Point2d(a_spat);


    double angle_b_rad = atan2 (vb[1],vb[0])+M_PI/2.0;
    double angle_b_deg = angle_b_rad * 180.0 / M_PI; 
    //double distb=sqrt(vb[0]*vb[0]+vb[1]*vb[1]);
    double dist_b=sqrt(va[0]*va[0]+va[1]*va[1]);
    double length_b= (double)m/dist_b;

    cv::Mat rot_b= cv::getRotationMatrix2D(cv::Point(0.0,0.0), -angle_b_deg, 1.0); // requires homogeneous coordinates
    cv::Point3d l_b= cv::Point3f(length_b,0.0,1.0);
    
    cv::Mat b_spat= rot_b * cv::Mat(l_b); // Spatial base vector/point from freq-basevector 

    cv::Point2d myb = cv::Point2d(b_spat);

    cv::Point2d cntr = cv::Point2d(image.cols/2, image.rows/2); 
    
    //store data of the resulting rectangle 
    cv::Point2d zero = cv::Point2d(0.0, 0.0);  // TODO use center as origin?
    netquadri.push_back(zero);
    netquadri.push_back(mya);
    netquadri.push_back(mya +myb);
    netquadri.push_back(myb);

    std::vector<cv::Point2d> cpynetquadri=netquadri;
    
    cv::Point2d offs = cntr + cv::Point2d(0,0);
    for(size_t i=0;i<cpynetquadri.size();i++){
	 cpynetquadri[i] += offs;
    }

    return netquadri;
}

std::vector<std::vector<double>> cs_fft_LocalMaxIdea(const cv::Mat &magnitude_image){
    
    cv::Mat mygray;
    cv::Mat fft_image;
    std::vector<cv::Point> maxima_result;
    cv::Mat mtmp; 
    cv::normalize(magnitude_image, mtmp, 0, 255, cv::NORM_MINMAX);
    mtmp.convertTo(mygray, CV_8U);

    cvtColor(mygray, fft_image, cv::COLOR_GRAY2BGR);

    // center of fft-image
    double cx = mygray.cols/2.0;
    double cy= mygray.rows/2.0;
    cv::Point fftcenter(cx,cy);

    //Consider only a sub region to find local maxima ....
    double searchradius= mygray.rows/5;

    cv::Size fsize = mygray.size();
    int w= fsize.width;
    int h= fsize.height;
    int wlength=(int)(searchradius*2.0);
    int hlength=(int)(searchradius*2.0);
    int xs=w/2-(int)(searchradius);
    int ys=h/2-(int)(searchradius);
    cv::Mat maxima_ROI(mygray, cv::Rect(xs,ys,wlength,hlength));

    maxima_result = GetLocalMaxima(maxima_ROI, 13, 60, 19);

    // correct coordinates of the found maxima regaring the shift due to the sub-region
    for(size_t i= 0; i < (int) maxima_result.size(); i++){
        maxima_result[i] += cv::Point(xs,ys);
    }

    std::vector<cv::Point2f> maxima_indicators;
    maxima_indicators.clear();
    
    for( int i = 0; i< (int) maxima_result.size(); i++ ){
        cv::Point c = maxima_result[i];
        if(cv::norm(fftcenter-c) < searchradius){
            maxima_indicators.push_back(cv::Point(c.x,c.y));
        }
    }

    std::vector<int> indicator;
    std::vector<std::vector<double> > basevectors;

    if((int) maxima_indicators.size() < 5) // minimum number of detected maximas to search for a grid
    {
        //std::cout << "[INFO]\tNot enough maxima to analyse!" << std::endl;
        basevectors.clear();
        return basevectors;  
    }

    indicator = checkgridstructure(maxima_indicators, &basevectors);

    return basevectors;
}
