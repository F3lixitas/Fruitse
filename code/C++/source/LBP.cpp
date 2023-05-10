#include "LBP.hpp"

template <typename T>
cv::Mat LBPImage(const cv::Mat& image) {
    cv::Mat outputImage = cv::Mat::zeros(image.rows-2, image.cols-2, CV_8UC1);
    for(int i = 1; i < image.rows-1; i++) {
        for(int j = 1; j < image.cols-1; j++) {
            T center = image.at<T>(i,j);
            uint8_t code = 0;
            code |= (image.at<T>(i-1,j-1) > center) << 7;
            code |= (image.at<T>(i-1,j) > center) << 6;
            code |= (image.at<T>(i-1,j+1) > center) << 5;
            code |= (image.at<T>(i,j+1) > center) << 4;
            code |= (image.at<T>(i+1,j+1) > center) << 3;
            code |= (image.at<T>(i+1,j) > center) << 2;
            code |= (image.at<T>(i+1,j-1) > center) << 1;
            code |= (image.at<T>(i,j-1) > center);
            outputImage.at<uint8_t>(i-1,j-1) = code;
        }
    }
    return outputImage;
}