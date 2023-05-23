#include "HOG.hpp"
#include <opencv2/objdetect.hpp>

struct::HOGFeatures HOG(const cv::Mat& image) {
    cv::HOGDescriptor hog;
    std::vector<float> desc;
    std::vector<cv::Point> locs;

    cv::Mat grads = cv::Mat();
    cv::Mat angles = cv::Mat();

    hog.computeGradient(image, grads, angles);
    HOGFeatures features;
    features.grads = grads;// CV_32F2
    features.angles = angles;// CV_8UC2

    return features;
}

void drawHOGPoints(cv::Mat image, HOGFeatures hog) {

    int linSpace = 16;
    hog.grads.at<float*>();
    for (int i = 0; i < hog.grads.size().width; i=i+linSpace) {
        for (int j = 0; j < hog.grads.size().height; j=j+linSpace) {
            // grads made of 2 32-bits floats
            cv::Vec2f grad = hog.grads.at<cv::Vec2f>(j,i);
            float startArrowX = image.size().width * (.5*linSpace + float(i)) / (hog.grads.size().width);
            float startArrowY = image.size().height * (.5*linSpace + float(j)) / (hog.grads.size().height);
            float endArrowX = startArrowX + grad[0]*linSpace/2;
            float endArrowY = startArrowY + grad[1]*linSpace/2;
            cv::line(image,cv::Point(startArrowX,startArrowY),cv::Point(endArrowX,endArrowY),{255,0,0});
            // red
            cv::drawMarker(image,cv::Point(endArrowX,endArrowY),{255,0,0},cv::MARKER_STAR,linSpace/4);
        }
    }
}
