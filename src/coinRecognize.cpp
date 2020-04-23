/**
 * Coin Recognize Software for Computer Vision course
 * @file Support file for coin recognize
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
#include <iostream>
#include <list>
#include <fstream>
#include <string>
#include <chrono> //for timestamp

#include "coinRecognize.h"

using namespace cv;
using namespace std;


coinRecognize::coinRecognize(Mat img, bool res){
    if(res && img.size().width > EXPXDIM) {
        // resize image to speed up the computation
        double factor = EXPXDIM / img.size().width;
        resize(img, img, Size(cvRound(EXPXDIM), cvRound(img.size().height * factor)));
    }
    //initialize the image
    this->img = img;
}



vector<Mat> coinRecognize::findCoins(int save) {

    // clear the two vector with the results and the folder with the images
    coinResults.clear();
    circlesResults.clear();
    prediction.clear();
    if(save==1) {
        system("exec rm -r ../images/predict/*");
        system("exec mkdir ../images/predict/coins");
    }

    Mat imgGray, imgBlur, thresh, edges, kernel, closing, imgPrint;
    vector<Vec3f> circles;

    // generate a copy of the image where draw circles
    imgPrint = img.clone();

    // Convert the image in a grayscale
    cvtColor(img, imgGray, CV_BGR2GRAY);

    // Apply a gaussian blur in order to remove noise
    //imgBlur = imgGray; // DEBUG (no blur)
    GaussianBlur(imgGray, imgBlur, Size(15,15), 1);

    // Create an adaptive threshold
    //adaptiveThreshold(img_blur, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 1); //deprecated
    double otsuTresh = threshold(imgBlur, thresh, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    double highThreshVal = otsuTresh,
            lowerThreshVal = otsuTresh / 2;

    // Create a rect kernel and use it whit some iteration of morphological operations
    cout << "\tBegin use morphological closure..." << endl;
    kernel = getStructuringElement( MORPH_RECT, Size(3,3));
    //morphologyEx(edges, closing, MORPH_CLOSE, kernel); // deprecated
    //morphologyEx(img_blur, closing, MORPH_CLOSE, kernel, Point(-1,-1), 3); // deprecated
    morphologyEx(imgBlur, closing, MORPH_ERODE, kernel, Point(-1,-1));
    morphologyEx(closing, closing, MORPH_DILATE, kernel, Point(-1,-1), 5);
    cout << "\tdone!" << endl;

    // Use canny to find edges (do also inside HoughCircles, so it's only for a visual feedback)
    cout << "\tBegin finding edges with Canny algorithm..." << endl;
    Canny(closing, edges, lowerThreshVal, highThreshVal, 3);
    cout << "\tdone!" << endl;

    // Apply the Hough Transform to find the circles
    int minDist = min(img.size().width, img.size().height) / 8;
    int maxRadius = min(img.size().width, img.size().height) / 4;
    int minRadius = cvRound(minDist * 0.5);
    cout << "\tBegin applying Hough Circles..." << endl;
    HoughCircles(closing, circles, CV_HOUGH_GRADIENT, 1.5, minDist, highThreshVal * 2, 20, minRadius, maxRadius);
    cout << "\tdone!\n\tFound " << to_string(circles.size()) << " circle(s)." << endl;
    if(circles.empty()){
        cout << "\tNo circles found!" << endl;
    }

    // find circles
    // counter of output files
    int j = 0;
    // padding for filenemes
    string zeros = "0";
    for(int i = 0; i < circles.size() && j < limiter; i++) {
        // center
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        // radius
        int radius = cvRound(circles[i][2]);

        // create the rect to generate the image of the single coin
        int xL = max(center.x - radius, 0);
        int xR = min(center.x + radius, closing.size().width);
        int yT = max(center.y - radius, 0);
        int yB = min(center.y + radius, closing.size().height);

        Mat coin = img(Rect(xL, yT, xR - xL, yB - yT));


        // I consider a circle as a possible coins only if a big part of it is inside the image, otherwise I discard it
        double epsilon = 0.6; //[0,1] where 0 = accept all, 1 = accept only circle completely inside the image // 0.8
        if( (xR - xL) * (yB - yT) > pow(radius*2,2)*epsilon ){
            // save the coin frame in the disk (if the option is selected)
            if (save == 1) {
                // get timestamp to make file unique
                unsigned now = std::chrono::duration_cast<std::chrono::milliseconds>
                        (std::chrono::system_clock::now().time_since_epoch()).count();
                imwrite("../images/predict/coins/" + zeros + to_string(j++) + "-" + to_string(now) + ".jpg", coin);
            }
            // print the circles
            circle(imgPrint, center, 3, Scalar(0, 255, 0), -1, 8, 0);
            circle(imgPrint, center, radius, Scalar(0, 0, 255), 3, 8, 0);

            // add to the list both the coin images both the circle
            coinResults.emplace_back(coin);
            circlesResults.emplace_back(circles[i]);

            // check j and in case remove a 0 from the padding string
            if(j==10)
                zeros = "";
            else if(j==100){
                cout << "Ehy man, you have more that 100 circles, maybe there is something wrong.." << endl;
                exit(EXIT_FAILURE);
            }
            //cout << "	Printed circle number " << to_string(i) << endl; // DEBUG

        }
    }

    namedWindow("Morphological Operations", CV_WINDOW_AUTOSIZE);
    imshow("Morphological Operations", closing);
    namedWindow("Otsu Thresholding", CV_WINDOW_AUTOSIZE);
    imshow("Otsu Thresholding", thresh);
    namedWindow("Circles", CV_WINDOW_AUTOSIZE);
    imshow("Circles", imgPrint);
    cout << "----- Press a key to continue ------" << endl;
    waitKey(0);


    return coinResults;
}


Mat coinRecognize::blackOutside(Mat img, vector<Point> coin){

    // check that there is (for real) a coin
    if(coin.empty())
        exit(EXIT_FAILURE);

    Mat result;

    // create a black mask
    Mat mask(img.size(), CV_8UC1, {0,0,0});

    // create a white filled ellipse
    RotatedRect el = fitEllipse(coin);
    ellipse(mask, el, {255,255,255}, -1);

    //Bitwise AND operation to black out regions outside the mask
    bitwise_and(img, img, result , mask);

    namedWindow("result");
    imshow("result", result);
    return result;

}


vector<string> coinRecognize::result() {

    // run the testing application
    cout << "Running tester.py to classify the coins..." << endl;
    system("python3 ../src/tester.py");
    cout << "done!" << endl;

    // delete eventually prediction already generated
    prediction.clear();

    // read predictions from CSV created by tester.py
    ifstream f("../images/predict/predictions.csv");
    if (!f.is_open()) {
        cout << "Error on opening CSV file." << endl;
        exit(EXIT_FAILURE);
    }
    else{
        string singlePred;
        while(getline(f, singlePred, ',')) {
            cout << "Found: " << singlePred << endl;
            // generate prediction
            prediction.emplace_back(singlePred);
        }
        // delete the end of line character from the last element
        int lastIndex = prediction.size()-1;
        prediction[lastIndex] =
                prediction[lastIndex].substr(0, prediction[lastIndex].size()-2);
    }

    return prediction;

}


Mat coinRecognize::printResults(){
    // if there are not already prediction, generate them
    if(prediction.empty())
        result();

    //cout << "pred len: " << prediction.size() << endl;
    //cout << "circles len: " << circlesResults.size() << endl;

    // copy the image
    Mat res = img.clone();

    if(prediction.size() != circlesResults.size())
        cout << "ERROR! Different size between prediction and circlesResults...how can it be?!?" << endl;

    for(int i = 0; i < circlesResults.size(); i++) {
        // if the coin is recognized print them
        if(prediction[i] != "unknown" && prediction[i] != "unknown\r\n") {
            // center
            Point center(cvRound(circlesResults[i][0]), cvRound(circlesResults[i][1]));
            // radius
            int radius = cvRound(circlesResults[i][2]);
            // print circle
            circle(res, center, 3, Scalar(0, 255, 0), -1, 8, 0);
            circle(res, center, radius, Scalar(0, 0, 255), 3, 8, 0);
            // print name
            putText(res, prediction[i], center, FONT_HERSHEY_SIMPLEX, 2, Scalar(0,255,255), 3, LINE_AA);
        }
    }

    namedWindow("Result", CV_WINDOW_AUTOSIZE);
    imshow("Result", res);

    return res;
}


void coinRecognize::setLimiter(int limiter){
    if(limiter > 0)
        this->limiter = limiter;
}
