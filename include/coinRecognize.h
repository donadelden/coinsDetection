/**
 * Coin Recognize Software for Computer Vision course
 * @headerfile header file for coinRecognize
 *
 * @author Denis Donadel <denis.donadel@studenti.unipd.it>
 * @date 6 July 2019
 * @version 1.0
 * @since 1.0
 *
 */

#ifndef COINSDETECTION_COINRECOGNIZE_H
#define COINSDETECTION_COINRECOGNIZE_H


class coinRecognize {

public:
    /**
     * @brief Constructor
     * @param img the image to be used
     * @param res decide if resize the image (in order to make the computation faster)
     */
    explicit coinRecognize(cv::Mat img, bool res = false);

    /**
     * @brief Function for individuate the coin in an image
     * @param save default = 0 for not saving anything on disk, 1 is for saving on disk
     * @return vector of vectors of points that are the contours of every coin
     */
    std::vector<cv::Mat> findCoins(int save = 0);

    /**
     * @brief Create a black mask outside the coin
     * @deprecated not used in this version
     * @param img the coin to "black outside"
     * @param coin countorns of the coin
     * @return img with black outside the coin
     */
    cv::Mat blackOutside(cv::Mat img, std::vector<cv::Point> coin);

    /**
     * @brief Classify images
     * @return vector with classifications of images
     */
    std::vector<std::string> result();

    /**
     * @brief Print and return the result in a new images
     * @return mat whit coins
     */
    cv::Mat printResults();

    /**
     * @brief Setters for limiter
     * @param limiter new limiter
     */
    void setLimiter(int limiter);

    /**
     * @brief Getter for limiter
     * @return limiter of output images
     */
    double getLimiter() {return limiter;}

private:
    cv::Mat img;
    /// vector containing the Mat with the single coins
    std::vector<cv::Mat> coinResults;
    /// vector containing the center and the radius of the circles
    std::vector<cv::Vec3f> circlesResults;
    /// output vector of prediction
    std::vector<std::string> prediction;
    /// dimension for the width of (optional) resizing
    const double EXPXDIM = 800;
    /// limit of coins in output
    double limiter = 19;
};


#endif //COINSDETECTION_COINRECOGNIZE_H
