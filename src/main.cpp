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

int main(int argc, char *argv[])
{
    string img_test = "../images_T2/1.jpg";

    string type = "20c"; //20c - 50c - 1e - 2e

    string img_train = "../coins-dataset/classified/total/"+type+"/*";  //train_22_32.jpg"
    string res_name_base = "../coins-dataset/classified/total/preprocessed/"+type+"/"+type+"_";
    string filename_20c = type+"_";


    /// preprocessing images
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


     /// find countours version
        vector<vector<Point>> coins = cr.findCoin(src, false);

        if(!coins.empty()) {
            vector<Point> biggest_coin;
            biggest_coin = coins.back();
            coins.pop_back();
            double biggest_area = contourArea(biggest_coin);
            for (auto &coin : coins) {
                double area = contourArea(coin);
                if (area > biggest_area)
                    biggest_coin = coin;
            }
            Mat res = cr.blackOutside(src, biggest_coin);
            String res_name =
                    res_name_base + filename_20c + to_string(i) + ".jpg";

            cout << res_name << endl;

            imwrite(res_name, res);
        }
        i++;
    }

/*

    ///testing on one image
    Mat src = imread("../coins-dataset/classified/total/20c/IMG_4210_1.jpg" , CV_LOAD_IMAGE_COLOR);

    //Mat src = imread("../images_T2/2.jpg");

    if(!src.data){
        cout << "Error on loading the image." << endl;
        return -1;
    }

    coinRecognize cr;
    vector<vector<Point>> coins = cr.findCoin(src, true);/*
    vector<Point> biggest_coin;
    biggest_coin = coins.back();
    coins.pop_back();
    double biggest_area = contourArea(biggest_coin);
    for(auto & coin : coins) {
        double area = contourArea(coin);
        if (area > biggest_area)
            biggest_coin = coin;
    }
    cr.blackOutside(src, biggest_coin);*/



    waitKey(0);
    return 0;
}



