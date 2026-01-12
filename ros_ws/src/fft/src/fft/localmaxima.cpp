#include "fft.hpp"

void localMaxima(cv::Mat src,cv::Mat &dst,int squareSize){
  if (squareSize==0){
    dst = src.clone();
    return;
  }

  cv::Mat m0;
  dst = src.clone();
  cv::Point maxLoc(0,0);

  //1.Be sure to have at least 3x3 for at least looking at 1 pixel close neighbours
  //  Also the window must be <odd>x<odd>
  //SANITYCHECK(squareSize,3,1);
  int sqrCenter = (squareSize-1)/2;

  //2.Create the localWindow mask to get things done faster
  //  When we find a local maxima we will multiply the subwindow with this MASK
  //  So that we will not search for those 0 values again and again
  cv::Mat localWindowMask = cv::Mat::zeros(cv::Size(squareSize,squareSize),CV_8U);//boolean
  localWindowMask.at<unsigned char>(sqrCenter,sqrCenter)=1;

  //3.Find the threshold value to threshold the image
  //  this function here returns the peak of histogram of picture
  //  the picture is a thresholded picture it will have a lot of zero values in it
  //  so that the second boolean variable says :
  //    (boolean) ? "return peak even if it is at 0" : "return peak discarding 0"

  int thrshld = 125;
  cv::threshold(dst,m0,thrshld,1,cv::THRESH_BINARY);

  //4.Now delete all thresholded values from picture
  dst = dst.mul(m0);

  //put the src in the middle of the big array
  for (int row=sqrCenter;row<dst.size().height-sqrCenter;row++){
    for (int col=sqrCenter;col<dst.size().width-sqrCenter;col++){
      //1.if the value is zero it can not be a local maxima
      if (dst.at<unsigned char>(row,col)==0){
        continue;
      }
      //2.the value at (row,col) is not 0 so it can be a local maxima point
      m0 =  dst.colRange(col-sqrCenter,col+sqrCenter+1).rowRange(row-sqrCenter,row+sqrCenter+1);
      cv::minMaxLoc(m0,NULL,NULL,NULL,&maxLoc);
      //if the maximum location of this subWindow is at center
      //it means we found the local maxima
      //so we should delete the surrounding values which lies in the subWindow area
      //hence we will not try to find if a point is at localMaxima when already found a neighbour was
      if ((maxLoc.x==sqrCenter)&&(maxLoc.y==sqrCenter)){
        m0 = m0.mul(localWindowMask);
        //we can skip the values that we already made 0 by the above function
        col+=sqrCenter;
      }
    }
  }
}


void non_maxima_suppression(const cv::Mat& inimage, cv::Mat& mask,int GaussKernel, bool remove_plateaus) {
  cv::Mat image;
   
  cvtColor(inimage, image,cv::COLOR_BGR2GRAY);
   

  int morph_elem=2; //0: Rect - 1: Cross - 2: Ellipse
  int lero=3;
  lero=GaussKernel/2;
  cv::Mat erodeele = cv::getStructuringElement(morph_elem, cv::Size(3, 3), cv::Point( 1, 1 ));
  erode(image, image, erodeele);//
  //cv::imshow("grayvalue image +erode", image);
  morph_elem=2; //0: Rect - 1: Cross - 2: Ellipse
  int morph_size=GaussKernel/2;
  cv::Mat dilele = cv::getStructuringElement(morph_elem, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point(morph_size, morph_size));
  int ldil=2*morph_size + 1;
  std::cout << "Erode with:" << lero <<"x" << lero  << "Dilate with: " << ldil << "x" << ldil << std::endl;

  if(GaussKernel > 1){ // If You need a smoothing
    GaussianBlur(image, image, cv::Size(2*GaussKernel+1,2*GaussKernel+1), 0, 0, 4);
  }
  //cv::imshow("smoothed grayvalue image", image);
  cv::dilate(image, mask, dilele);// cv::Mat());
  //cv::imshow("dilate smoothed grayvalue", mask);
  cv::compare(image, mask, mask, cv::CMP_EQ);

  // optionally filter out pixels that are equal to the local minimum ('plateaus')
  if (remove_plateaus) {
    cv::Mat non_plateau_mask;
    cv::erode(image, non_plateau_mask, cv::Mat());
    cv::compare(image, non_plateau_mask, non_plateau_mask, cv::CMP_GT);
    cv::bitwise_and(mask, non_plateau_mask, mask);
  }
}

std::vector<cv::Point> GetLocalMaxima(const cv::Mat Src,int MatchingSize, int Threshold, int GaussKernel  )
{  
  std::vector<cv::Point> vMaxLoc(0); 

  if ((MatchingSize % 2 == 0) || (GaussKernel % 2 == 0)) // MatchingSize and GaussKernel have to be "odd" and > 0
  {
    return vMaxLoc;
  }

  vMaxLoc.reserve(100); // Reserve place for fast access 
  cv::Mat ProcessImg = Src.clone();
  int W = Src.cols;
  int H = Src.rows;
  int SearchWidth  = W - MatchingSize;
  int SearchHeight = H - MatchingSize;
  int MatchingSquareCenter = MatchingSize/2;

  if(GaussKernel > 1){ // If You need a smoothing
    GaussianBlur(ProcessImg, ProcessImg, cv::Size(GaussKernel,GaussKernel), 0, 0, 4);
  }
  uchar* pProcess = (uchar *) ProcessImg.data; // The pointer to image Data 

  int Shift = MatchingSquareCenter * ( W + 1);
  int k = 0;

  for(int y=0; y < SearchHeight; ++y){ 
    int m = k + Shift;
    for(int x=0;x < SearchWidth ; ++x)
    {
      if (pProcess[m++] >= Threshold){
        cv::Point LocMax;
        cv::Mat mROI(ProcessImg, cv::Rect(x,y,MatchingSize,MatchingSize));
        minMaxLoc(mROI,NULL,NULL,NULL,&LocMax);
        if (LocMax.x == MatchingSquareCenter && LocMax.y == MatchingSquareCenter){ 
          vMaxLoc.push_back(cv::Point( x+LocMax.x,y + LocMax.y )); 
          //imshow("W1",mROI);cvWaitKey(0); //For debug              
        }
      }
    }
    k += W;
  }
  return vMaxLoc; 
}
