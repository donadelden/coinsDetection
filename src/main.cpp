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
#include <glob.h>

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
    String img_test = "../images_T2/1.jpg";
    String img_train = "../coins-dataset/classified/total/20c/*";  //train_22_32.jpg";
    String filename_20c = "20c_";

    // preprocess images

    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));
    // do the glob operation
    int return_value = glob((img_train + "*").c_str(), GLOB_TILDE, NULL, &glob_result);
    if(return_value != 0) {
        globfree(&glob_result);
        stringstream ss;
        ss << "glob() failed with return_value " << return_value << endl;
        throw runtime_error(ss.str());
    }

    // collect all the filenames into a std::list<std::string>
    vector<string> filenames;
    for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.emplace_back(string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);
    int i = 0;
    for(auto & fileName : filenames){
        cout << "i: " << i << endl;
        Mat src = imread(fileName , CV_LOAD_IMAGE_COLOR);
        cout << "Filename: " + fileName << endl;

        if(!src.data){
            cout << "Error on loading the image." << endl;

            return -1;
        }

        coinRecognize cr;
        vector<vector<Point>> coins = cr.findCoin(src);
        vector<Point> biggest_coin;
        biggest_coin = coins.front();
        coins.pop_back();
        double biggest_area = UINTMAX_MAX;
        for(auto & coin : coins) {
            double area = contourArea(coin);
            if (area > biggest_area)
                biggest_coin = coin;
        }
        Mat res = cr.blackOutside(src, biggest_coin);
        String res_name = "../coins-dataset/classified/total/preprocessed/20c/20c_"+filename_20c+ to_string(i) +".jpg";
        i++;

        cout << res_name << endl;

        imwrite(res_name, res);

    }

    /*
    String img_test = "../images_T2/1.jpg";
    String img_train = "../coins-dataset/classified/total/20c/train_22_32.jpg";
    src = imread(img_train , CV_LOAD_IMAGE_COLOR);

    if(!src.data){
        cout << "Error on loading the image." << endl;
        return -1;
    }

    coinRecognize cr;
    vector<vector<Point>> coins = cr.findCoin(src);
    vector<Point> biggest_coin;
    biggest_coin = coins.front();
    coins.pop_back();
    double biggest_area = UINTMAX_MAX;
    for(auto & coin : coins) {
        double area = contourArea(coin);
        if (area > biggest_area)
            biggest_coin = coin;
    }
    cr.blackOutside(src, biggest_coin);

    //namedWindow("Undistorted image", CV_WINDOW_NORMAL);
    //imshow("Undistorted image", edges);

    // Create a window
    //namedWindow( window_name, CV_WINDOW_NORMAL );

    // Create a Trackbar for user to enter threshold
    //createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, HougeThreshold );

    // Show the image
    //CannyThreshold(0, 0);
    //HougeThreshold(0,0);
    */
    waitKey(0);
    return 0;
}



