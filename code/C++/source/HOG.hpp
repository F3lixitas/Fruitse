#ifndef FRUITSE_HOG_HPP
#define FRUITSE_HOG_HPP

#include <opencv2/opencv.hpp>
typedef struct HOGFeatures {
    // matrice de CV_32F2
    cv::Mat2f grads;
    // matrice de CV_8UC2
    cv::Mat2b angles;
} HOGFeatures;

HOGFeatures HOG(const cv::Mat& image);

void drawHOGPoints(cv::Mat image, HOGFeatures hog);

#endif //FRUITSE_LBP_HPP
