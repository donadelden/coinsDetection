//
// Created by denis on 10/06/19.
//

#ifndef COINSDETECTION_COINRECOGNIZE_H
#define COINSDETECTION_COINRECOGNIZE_H


class coinRecognize {

public:
    std::vector<std::vector<cv::Point>> findCoin(cv::Mat img, bool print_contours);
    cv::Mat blackOutside(cv::Mat img, std::vector<cv::Point> coin);
    std::vector<std::string> result();

};


#endif //COINSDETECTION_COINRECOGNIZE_H
