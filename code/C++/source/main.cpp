#include "LBP.hpp"
#include "pretraitement.hpp"
#include "histo16.hpp"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>

const int NB_CLASSES = 9;
const char* classes[] = {
        "apple", "apricot", "banana", "blueberry", "grape", "kiwi", "orange", "pear", "strawberry"
};

typedef struct {
    std::string path;
    int imageClass;
} DataPoint;

#define PROJECT_ROOT_PATH std::filesystem::current_path().parent_path().parent_path().parent_path()

cv::Mat histogram(const cv::Mat& image) {
    float range[] = { 0, 256 };
    int histSize = 256;
    const float* histRange[] = { range };
    cv::Mat hist;
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, true, false);
    hist.at<float>(0) = 0.0f;
    return hist;
}

cv::Mat displayHistogram(const cv::Mat& hist) {
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/256 );
    cv::Mat histImage( hist_h, hist_w, CV_8UC1, cv::Scalar( 0) );

    normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    for( int i = 1; i < 256; i++ )
    {
        line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
              cv::Scalar( 255), 2, 8, 0  );
    }
    return histImage;
}

/*
 * Charge une image ou une série d'images et écrit ses caractéristiques dans un fichier csv.
 */
int main(int argc, char* argv[]) {
    if(argc >= 2) {
        cv::HOGDescriptor hog;
        for (int i = 1; i < argc; i++) {
            std::filesystem::path path = std::filesystem::absolute(PROJECT_ROOT_PATH.concat(argv[i]));

            cv::Mat image = cv::imread(path.string());
            cv::Mat mask = pretraitement(image);

            cv::Mat hueHist = histo16(image);

            cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
            bitwise_and(image, image, image, mask);


            cv::Mat lbp_pe = LBPImage(image);
            cv::Mat hist_pe = histogram(lbp_pe);

            hog.winSize = image.size();
            std::vector<float> hog_pe;
            hog.compute( image, hog_pe, cv::Size( 4, 4 ), cv::Size( 0, 0 ) );

            cv::resize(image, image, cv::Size(), 0.5, 0.5);
            cv::Mat lbp_de = LBPImage(image);
            cv::Mat hist_de = histogram(lbp_de);

            hog.winSize = image.size();
            std::vector<float> hog_de;
            hog.compute( image, hog_de, cv::Size( 4, 4 ), cv::Size( 0, 0 ) );

            cv::resize(image, image, cv::Size(), 0.5, 0.5);
            cv::Mat lbp_qe = LBPImage(image);
            cv::Mat hist_qe = histogram(lbp_qe);

            std::filesystem::path p = std::filesystem::absolute(PROJECT_ROOT_PATH.concat(std::string("/datasets/test")+std::to_string(i)+".csv"));
            if(std::filesystem::exists(p)) {
                std::filesystem::remove(p);
            }
            std::ofstream outputFile(p, std::fstream::app);
            outputFile << 9 << ", " << format(hueHist.t(), cv::Formatter::FMT_CSV) << ", " << format(hist_pe.t(), cv::Formatter::FMT_CSV) << ", " <<
                       format(hist_de.t(), cv::Formatter::FMT_CSV) << ", " << format(hist_qe.t(), cv::Formatter::FMT_CSV) << ", " <<
                       format(hog_pe, cv::Formatter::FMT_CSV) << ", " << format(hog_de, cv::Formatter::FMT_CSV) << std::endl;
            outputFile.close();
        }
        return 0;
    }
    std::vector<DataPoint> trainDataFiles;
    for(int i = 0; i < NB_CLASSES; i++) {
        std::string s = "/images/base_apprentissage/";
        s += classes[i];
        std::filesystem::path path = std::filesystem::absolute(PROJECT_ROOT_PATH.concat(s));
        if(std::filesystem::exists(path)) {
            for (auto const& imagePath : std::filesystem::directory_iterator{path})
            {
                trainDataFiles.push_back({imagePath.path().string(), i});
            }
        } else {
            std::cerr << "Learning folder for " << classes[i] << " does not exist!\n";
        }
    }
    std::filesystem::path path = std::filesystem::absolute(PROJECT_ROOT_PATH.concat("/datasets/sortie.csv"));
    if(std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }

    std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
    std::chrono::system_clock::duration dtn = tp.time_since_epoch();
    auto rng = std::default_random_engine {};
    rng.seed(dtn.count());
    std::shuffle(trainDataFiles.begin(), trainDataFiles.end(), rng);
    cv::HOGDescriptor hog;

    for(DataPoint& dp : trainDataFiles) {
        // extraction des caractéristiques
        cv::Mat image = cv::imread(dp.path);
        cv::Mat mask = pretraitement(image);
        cv::Mat hueHist = histo16(image);
        cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
        bitwise_and(image, image, image, mask);
        cv::Mat lbp_pe = LBPImage(image);
        cv::Mat hist_pe = histogram(lbp_pe);

        hog.winSize = image.size();
        std::vector<float> hog_pe;
        hog.compute( image, hog_pe, cv::Size( 4, 4 ), cv::Size( 0, 0 ) );

        cv::resize(image, image, cv::Size(), 0.5, 0.5);
        cv::Mat lbp_de = LBPImage(image);
        cv::Mat hist_de = histogram(lbp_de);

        hog.winSize = image.size();
        std::vector<float> hog_de;
        hog.compute( image, hog_de, cv::Size( 4, 4 ), cv::Size( 0, 0 ) );

        cv::resize(image, image, cv::Size(), 0.5, 0.5);
        cv::Mat lbp_qe = LBPImage(image);
        cv::Mat hist_qe = histogram(lbp_qe);

        std::ofstream outputFile(path, std::fstream::app);
        outputFile << dp.imageClass << ", " << format(hueHist.t(), cv::Formatter::FMT_CSV) << ", " << format(hist_pe.t(), cv::Formatter::FMT_CSV) << ", " <<
            format(hist_de.t(), cv::Formatter::FMT_CSV) << ", " << format(hist_qe.t(), cv::Formatter::FMT_CSV) << ", " <<
            format(hog_pe, cv::Formatter::FMT_CSV) << ", " << format(hog_de, cv::Formatter::FMT_CSV) << std::endl;
        outputFile.close();
    }
/*
    if(argc >= 2){
        for(int i = 1; i < argc; i++) {
            std::string s = argv[i];
            std::filesystem::path path = std::filesystem::absolute(PROJECT_ROOT_PATH.concat(s));
            if(std::filesystem::exists(path)) {
                cv::Mat image = pretraitement(cv::imread(std::filesystem::absolute(PROJECT_ROOT_PATH.concat(s)).string()));
                cv::namedWindow("Image", 1);
                cv::imshow("Image", image);
                cv::Mat dst;
                cv::cvtColor(image, dst, cv::COLOR_BGR2GRAY);

                cv::Mat lbp_pe = LBPImage(dst);
                cv::resize(dst, dst, cv::Size(), 0.5, 0.5);
                cv::Mat lbp_de = LBPImage(dst);
                cv::resize(dst, dst, cv::Size(), 0.5, 0.5);
                cv::Mat lbp_qe = LBPImage(dst);
                cv::namedWindow("Pleine echelle", 1);
                cv::imshow("Pleine echelle", lbp_pe);
                cv::namedWindow("Demi echelle", 1);
                cv::imshow("Demi echelle", lbp_de);
                cv::namedWindow("Quart d'echelle", 1);
                cv::imshow("Quart d'echelle", lbp_qe);

                // histogramme
                cv::Mat hist_pe = histogram(lbp_pe);
                cv::Mat histImage_pe = displayHistogram(hist_pe);
                cv::namedWindow("Histogramme pleine echelle", 1);
                imshow("Histogramme pleine echelle", histImage_pe );

                cv::Mat hist_de = histogram(lbp_de);
                cv::Mat histImage_de = displayHistogram(hist_de);
                cv::namedWindow("Histogramme demi echelle", 1);
                imshow("Histogramme demi echelle", histImage_de );

                cv::Mat hist_qe = histogram(lbp_qe);
                cv::Mat histImage_qe = displayHistogram(hist_qe);
                cv::namedWindow("Histogramme quart d'echelle", 1);
                imshow("Histogramme quart d'echelle", histImage_qe );



                cv::waitKey(0);
            } else {
                std::cerr << "The path <" << path.string() << "> does not exist !\n";
            }
        }
    }*/
    return 0;
}
