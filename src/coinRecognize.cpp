//
// Created by denis on 10/06/19.
//

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <list>

#include "coinRecognize.h"

using namespace cv;
using namespace std;

Mat coinRecognize::test(Mat img) {

        /// max value of the area of an ellipse so that we can consider it a coin
        //TODO: it might be set in relation with the maxArea
        int minArea = 19000;

        Mat img_gray, img_blur, thresh, kernel, closing, cont_img;

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;


        /// Convert the image in a grayscale
        cvtColor(img, img_gray, CV_BGR2GRAY);

        /// Apply a gaussian blur in order to remove noise
        GaussianBlur(img_gray, img_blur, Size(15,15), 3);

        /// Create an adaptive threshold
        adaptiveThreshold(img_blur, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 1);

        /// Create a rect kernel and use it in a morphological closure
        kernel = getStructuringElement( MORPH_RECT, Size(3,3));
        morphologyEx(thresh, closing, MORPH_CLOSE, kernel);

        /// find all the contours of the image
        cont_img = closing.clone();
        findContours(cont_img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        /// for each contour print out only the ones that are coin (almost, with high probability)
        for(auto & contour : contours){
            double area = contourArea(contour);
            if (area < minArea)
                continue;

            if (contour.size() < 5)
                continue;

            RotatedRect el = fitEllipse(contour);
            ellipse(img, el, (0,255,255), 2);
        }

        imshow("Morphological Closing", closing);
        imshow("Adaptive Thresholding", thresh);
        namedWindow("Contours", CV_WINDOW_NORMAL);
        imshow("Contours", img);

    return img;




}

