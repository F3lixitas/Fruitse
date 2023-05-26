//
// Created by Melissa on 26/05/2023.
//
#include "histo16.hpp"

cv::Mat histo16(cv::Mat img) {
    if (img.empty())
    {
        std::cout << "Impossible de charger l'image." << std::endl;
    }

    // Convertir l'image en espace de couleur HSV
    cv::Mat hsvImage;
    cv::cvtColor(img, hsvImage, cv::COLOR_BGR2HSV);

    // Séparer les canaux de l'image HSV
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);

    // Définir le nombre de classes de l'histogramme
    int numClasses = 16;

    // Définir les plages de valeurs pour la teinte
    float range[] = { 0, 180 };  // La teinte varie de 0 à 180 dans l'espace de couleur HSV
    const float* histRange = { range };

    // Calculer l'histogramme de la teinte
    cv::Mat histogram;
    cv::calcHist(&hsvChannels[0], 1, 0, cv::Mat(), histogram, 1, &numClasses, &histRange);

    // Créer l'affichage de l'histogramme
    int histWidth = 512;
    int histHeight = 400;
    int binWidth = cvRound((double)histWidth / numClasses);
    cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    // Normaliser les valeurs de l'histogramme pour qu'elles tiennent dans l'image
    cv::normalize(histogram, histogram, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    // Dessiner les barres de l'histogramme
    for (int i = 0; i < numClasses; i++)
    {
        rectangle(histImage, cv::Point(binWidth * i, histHeight), cv::Point(binWidth * i + binWidth - 1, histHeight - cvRound(histogram.at<float>(i))), cv::Scalar(255, 255, 255), -1);
    }

    // Afficher l'histogramme
    cv::namedWindow("Histogramme", cv::WINDOW_AUTOSIZE);
    cv::imshow("Histogramme", histImage);
    cv::waitKey(0);

    return histImage;
}