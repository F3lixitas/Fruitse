#include "LBP.hpp"
#include <filesystem>
#include <iostream>

#define PROJECT_ROOT_PATH std::filesystem::current_path().parent_path().parent_path().parent_path()

/*
 * Charge une image ou une série d'images et écrit ses caractéristiques dans un fichier csv.
 */
int main(int argc, char* argv[]) {
    if(argc >=2){
        for(int i = 1; i < argc; i++) {
            std::string s = argv[i];
            std::filesystem::path path = std::filesystem::absolute(PROJECT_ROOT_PATH.concat(s));
            if(std::filesystem::exists(path)) {
                cv::Mat image = cv::imread(std::filesystem::absolute(PROJECT_ROOT_PATH.concat(s)).string());
                cv::Mat dst;
                cv::cvtColor(image, dst, cv::COLOR_BGR2GRAY);
                cv::Mat lbp = LBPImage(dst);
                cv::namedWindow("Image", 1);
                cv::imshow("Image", lbp);
                cv::waitKey(0);
            } else {
                std::cerr << "The path <" << path.string() << "> does not exist !\n";
            }
        }
    }
    return 0;
}