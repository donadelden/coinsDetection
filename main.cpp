//
// Created by denis on 10/06/19.
//


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <list>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    Mat img = imread("../lena.png" , CV_LOAD_IMAGE_COLOR);

    if(!img.data){
        cout << "Error on loading the image." << endl;
        return -1;
    }

    namedWindow("Undistorted image", CV_WINDOW_NORMAL);
    imshow("Undistorted image", img);

    waitKey(0);
    return 0;
}



