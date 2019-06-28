/*
 * Created by denis on 10/06/19.
 *
 * To do:
 * - detect coins
 *   Smoothing with a Gaussian filter
 *   Compute gradient (module and direction)
 *   Quantize the gradient angles
 *   Non-maxima suppression
 *   Thresholding with double threshold
 *
 * - identify them
 */


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <list>

#include "coinRecognize.h"

using namespace cv;
using namespace std;

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold = 1;
int const max_lowThreshold = 300;
int ratio = 2;
int kernel_size = 3;
char* window_name = "Edge Map";

void CannyThreshold(int, void*)
{
    /// Reduce noise with a kernel 3x3
    blur( src_gray, detected_edges, Size(3,3) );

    /// Canny detector
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);

    src_gray.copyTo( dst, detected_edges);
    imshow( window_name, dst );
}


void HougeThreshold(int, void*){
    vector<Vec3f> coin;
    HoughCircles(src_gray,coin,CV_HOUGH_GRADIENT,src_gray.rows/8,20,lowThreshold*ratio,lowThreshold,0,0 );

    int l = coin.size();
    // Get the number of coins.

    cout<<"\n The number of coins is: "<<l<<"\n\n";

    // To draw the detected circles.
    for( size_t i = 0; i < coin.size(); i++ )
    {
        Point center(cvRound(coin[i][0]),cvRound(coin[i][1]));
        // Detect center
        // cvRound: Rounds floating point number to nearest integer.
        int radius=cvRound(coin[i][2]);
        // To get the radius from the second argument of vector coin.
        circle(src_gray,center,3,Scalar(0,255,0),-1,8,0);
        // circle center
        //  To get the circle outline.
        circle(src_gray,center,radius,Scalar(0,0,255),3,8,0);
        // circle outline
        cout<< " Center location for circle "<<i+1<<" :"<<center<<"\n Diameter : "<<2*radius<<"\n";
    }
    cout<<"\n";

    src_gray.copyTo( dst, detected_edges);
    imshow( window_name, dst );

}


int main(int argc, char *argv[])
{
    src = imread("../images_T2/1.jpg" , CV_LOAD_IMAGE_COLOR);

    if(!src.data){
        cout << "Error on loading the image." << endl;
        return -1;
    }

    coinRecognize cr;
    Mat edges = cr.test(src);

    //namedWindow("Undistorted image", CV_WINDOW_NORMAL);
    //imshow("Undistorted image", edges);

    // Create a window
    //namedWindow( window_name, CV_WINDOW_NORMAL );

    // Create a Trackbar for user to enter threshold
    //createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, HougeThreshold );

    // Show the image
    //CannyThreshold(0, 0);
    //HougeThreshold(0,0);

    waitKey(0);
    return 0;
}



