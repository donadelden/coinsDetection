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

int main(int argc, char *argv[])
{
    Mat img = imread("../images_T2/1.jpg" , CV_LOAD_IMAGE_COLOR);

    if(!img.data){
        cout << "Error on loading the image." << endl;
        return -1;
    }

    coinRecognize cr;
    Mat edges = cr.test(img);

    namedWindow("Undistorted image", CV_WINDOW_NORMAL);
    imshow("Undistorted image", edges);

    waitKey(0);
    return 0;
}



