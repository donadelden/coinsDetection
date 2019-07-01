//
// Created by denis on 10/06/19.
//

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <list>
#include <fstream>
#include <string>

#include "coinRecognize.h"

using namespace cv;
using namespace std;




/**
 * Function for individuate the coin in an image
 * @param img the colored image
 * @param print_contours decide if print, or not, green contourns on the output image (default is false)
 * @return vector of vectors of points that are the contours of every coin
 */
vector<vector<Point>> coinRecognize::findCoin(Mat img, bool print_contours = false) {

    /// max value of the area of an ellipse so that we can consider it a coin
    //TODO: it might be set in relation with the maxArea
    int minArea = 600;

    Mat img_gray, img_blur, thresh, kernel, closing, cont_img;
    vector<vector<Point>> contours, coins;
    vector<Vec4i> hierarchy;

    /// Convert the image in a grayscale
    cvtColor(img, img_gray, CV_BGR2GRAY);

    /// Apply a gaussian blur in order to remove noise
    //img_blur = img_gray; // bo blur
    GaussianBlur(img_gray, img_blur, Size(15,15), 3);

    /// Create an adaptive threshold
    //adaptiveThreshold(img_blur, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 1); //... or you can use Otsu
    threshold(img_blur, thresh, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    /// Create a rect kernel and use it in a morphological closure
    kernel = getStructuringElement( MORPH_RECT, Size(3,3));
    morphologyEx(thresh, closing, MORPH_CLOSE, kernel);

    /// find all the contours of the image
    cont_img = closing.clone();
    //dilate(cont_img, cont_img, noArray());
    findContours(cont_img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); //... or you can use hough

    /// for each contour print out only the ones that are coin (almost, with high probability)
    for(auto & contour : contours){
        double area = contourArea(contour);
        if (area >= minArea && contour.size() >= 5) {
            RotatedRect el = fitEllipse(contour);
            if(print_contours)
                ellipse(img, el, {0,255,0}, 2);
            coins.push_back(contour);
        }
    }

/*
    /// Apply the Hough Transform to find the circles
    vector<Vec3f> circles;
    int minDist = 100;
    int minRadius = 50;
    int maxRadius = 120;
    HoughCircles( cont_img, circles, CV_HOUGH_GRADIENT, 1, minDist, 1000, 20, minRadius, maxRadius);

    /// Detect biggest circle
    for(auto & i : circles) {
        /// get the center
        Point center(cvRound(i[0]), cvRound(i[1]));
        /// get the radius
        int radius = cvRound(i[2]);

        /// print the circles
        circle( img, center, 3, Scalar(0,255,255), -1);
        circle( img, center, radius, Scalar(0,0,255), 1 );
    }
*/

    imshow("Morphological Closing", closing);
    imshow("Adaptive Thresholding", thresh);
    namedWindow("Contours", CV_WINDOW_NORMAL);
    imshow("Contours", img);

    return coins;
}

Mat coinRecognize::blackOutside(Mat img, vector<Point> coin){

    /// check that there is (for real) a coin
    if(coin.empty())
        exit(EXIT_FAILURE);

    Mat result;

    /// create a black mask
    Mat mask(img.size(), CV_8UC1, {0,0,0});

    /// create a white filled ellipse
    RotatedRect el = fitEllipse(coin);
    ellipse(mask, el, {255,255,255}, -1);

    ///Bitwise AND operation to black out regions outside the mask
    bitwise_and(img, img, result , mask);

    imshow("result.jpg", result);
    return result;

}

vector<string> coinRecognize::result() {

    /// run the testing application
    system("python3 ./src/tester.py");
    cout << "Running tester.py done succesfully." << endl;

    /// output vector of string
    vector<string> pred;

    /// read predictions from CSV created by tester.py
    ifstream f("images/detect/pred.csv");
    if (!f.is_open()) {
        cout << "Error on opening CSV file." << endl;
        exit(EXIT_FAILURE);
    }

    else{
        string single_pred;
        while(getline(f, single_pred, ',')) {
            cout << "Found: " << single_pred << endl;
            pred.emplace_back(single_pred);
        }
    }

    return pred;

}

