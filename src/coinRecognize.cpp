//
// Created by denis on 10/06/19.
//

#include <iostream>
#include <opencv2/imgproc.hpp>

#include "coinRecognize.h"

using namespace cv;

Mat coinRecognize::test(Mat img) {

        // need to use a trackbar for now: https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html

        Mat edges;

        double tl = 5; // low threshold
        double th = 30; // high threshold
        int apertureSize = 3; // default in 3

        Canny(img, edges, tl, th, apertureSize);

        return edges;

}