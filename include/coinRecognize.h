//
// Created by denis on 10/06/19.
//

#ifndef COINSDETECTION_COINRECOGNIZE_H
#define COINSDETECTION_COINRECOGNIZE_H


class coinRecognize {

public:
    std::vector<std::vector<cv::Point>> findCoin(cv::Mat img);
    cv::Mat blackOutside(cv::Mat img, std::vector<cv::Point> coin);

};


#endif //COINSDETECTION_COINRECOGNIZE_H
