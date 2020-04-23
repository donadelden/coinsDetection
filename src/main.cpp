/**
 * Coin Recognize Software for Computer Vision course
 * @file Main file for execution
 *
 * @author Denis Donadel <denis.donadel@studenti.unipd.it>
 * @date 6 July 2019
 * @version 1.0
 * @since 1.0
 *
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


/**
 * Detect coin inside an image
 * @param path of the image (otherwise, it's used the default one)
 * @return 0 for ok, -1 for error
 */
int main(int argc, char *argv[])
{

    cout << "Welcome!\nLoad of the image..." << endl;
    string path = "../images/pic/4.jpg"; // default images
    if(argc == 2){ // if user pass his path to an images
        path = argv[1];
    }

    Mat src = imread(path);
    if(!src.data){
        cout << "Error on loading the image." << endl;
        return -1;
    }
    cout << "Image correctly loaded." << endl;

    // create the object
    coinRecognize cr(src);

    // findCoins
    cout << "Begin finding coins..." << endl;
    cr.findCoins(1);
    cout << "done!" << endl;

    // predict
    cout << "Begin predictions generation..." << endl;
    cr.result();
    cout << "done!" << endl;

    // print prediction
    cout << "Begin printing predictions..." << endl;
    cr.printResults();
    cout << "done!" << endl;

    waitKey(0);
    return 0;
}



