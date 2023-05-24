#include "opencv2/opencv.hpp"
#include <vector>

using namespace std;

/**
 * Charge une image, en enlève les bordures et le fond
 */

cv::Mat pretraitement(cv::Mat img) {
    // Charger l'image de fruits
    // cv::Mat img = cv::imread("C:\\Users\\Melissa\\Documents\\FISE2\\S8\\chef_d_oeuvre\\couleur_CO_git\\images\\grape\\156.jpg");

    // Convertir l'image en niveau de gris
    cv::Mat gray;
    cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Détecter les bandes blanches qui entourent l'image
    int thresh = 200;
    // Vérifier si l'image a des bandes blanches horizontales
    bool horizontal_bands = true;
    int top_horizontal_band = 0;
    int bottom_horizontal_band = 255;
    // On cherche une bordure par en haut
    for (int i = 0; (i < gray.rows) & horizontal_bands; i++) {
        for (int j = 0; (j < gray.cols) & horizontal_bands; j++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
            if (pixel[0] < thresh || pixel[1] < thresh || pixel[2] < thresh) {
                horizontal_bands = false;
            }
        }
        top_horizontal_band = i;
    }
    // S'il y a une bordure en haut, on cherche la bordure en bas
    if (top_horizontal_band != 0){
        horizontal_bands = true;
        for (int i = gray.rows-1; (i >= 0) & horizontal_bands; i--) {
            for (int j = 0; (j < gray.cols) & horizontal_bands; j++) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
                if (pixel[0] < thresh || pixel[1] < thresh || pixel[2] < thresh) {
                    horizontal_bands = false;
                }
            }
            bottom_horizontal_band = i;
        }
    }
    // Vérifier si l'image a des bandes blanches verticales
    bool vertical_bands = true;
    int left_vertical_band = 0;
    int right_vertical_band = 207;
    // On cherche une bordure par la gauche
    for (int i = 0; (i < gray.cols) & vertical_bands; i++) {
        for (int j = 0; (j < gray.rows) & vertical_bands; j++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(j, i);
            if (pixel[0] < thresh || pixel[1] < thresh || pixel[2] < thresh) {
                vertical_bands = false;
            }
        }
        left_vertical_band = i;
    }
    // S'il y a une bordure à gauche, on cherche la bordure à droite
    if (left_vertical_band != 0){
        vertical_bands = true;
        for (int i = gray.cols-1; (i >= 0) & vertical_bands; i--) {
            for (int j = 0; (j < gray.rows) & vertical_bands; j++) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(j, i);
                if (pixel[0] < thresh || pixel[1] < thresh || pixel[2] < thresh) {
                    vertical_bands = false;
                }
            }
            right_vertical_band = i;
        }
    }

    //Création de "lignes" grâce aux deux étapes précédentes
    std::vector<cv::Vec2f> lignes(8);
    if ((top_horizontal_band != 0) & (left_vertical_band != 0)) {
        lignes[0][0] = 0;//208*256
        lignes[0][1] = top_horizontal_band;
        lignes[1][0] = 207;
        lignes[1][1] = top_horizontal_band;
        lignes[2][0] = 0;//208*256
        lignes[2][1] = bottom_horizontal_band;
        lignes[3][0] = 207;
        lignes[3][1] = bottom_horizontal_band;

        lignes[4][0] = left_vertical_band;//208*256
        lignes[4][1] = 0;
        lignes[5][0] = left_vertical_band;
        lignes[5][1] = 256;
        lignes[6][0] = right_vertical_band;//208*256
        lignes[6][1] = 0;
        lignes[7][0] = right_vertical_band;
        lignes[7][1] = 256;
    }
    else if (top_horizontal_band != 0) {
        lignes[0][0] = 0;//208*256
        lignes[0][1] = top_horizontal_band;
        lignes[1][0] = 207;
        lignes[1][1] = top_horizontal_band;
        lignes[2][0] = 0;//208*256
        lignes[2][1] = bottom_horizontal_band;
        lignes[3][0] = 207;
        lignes[3][1] = bottom_horizontal_band;
    }
    else if (left_vertical_band != 0) {
        lignes[0][0] = left_vertical_band;//208*256
        lignes[0][1] = 0;
        lignes[1][0] = left_vertical_band;
        lignes[1][1] = 256;
        lignes[2][0] = right_vertical_band;//208*256
        lignes[2][1] = 0;
        lignes[3][0] = right_vertical_band;
        lignes[3][1] = 256;
    }

    // Dessiner les lignes détectées sur l'image d'origine
    cv::Mat imageLignes;
    cv::cvtColor(gray, imageLignes, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < lignes.size(); i+=2) {
        double x1 = lignes[i][0];
        double y1 = lignes[i][1];
        double x2 = lignes[i+1][0];
        double y2 = lignes[i+1][1];
        cv::Point pt1(x1, y1);
        cv::Point pt2(x2, y2);
        cv::line(imageLignes, pt1, pt2, cv::Scalar(0, 0, 255), 2);
    }

    // Afficher l'image avec les lignes détectées
    cv::namedWindow("Lignes détectées",1);
    cv::imshow("Lignes détectées", imageLignes);
    cv::waitKey(0);

    // Appliquer un filtre de gradient pour détecter les contours des fruits
    cv::Mat gradient;
    gray = gray.colRange(left_vertical_band,right_vertical_band);
    gray = gray.rowRange(top_horizontal_band,bottom_horizontal_band);
    cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    morphologyEx(gray, gradient, cv::MORPH_GRADIENT, kernel);

    // Appliquer un seuil pour séparer les pixels de fond des pixels de premier plan
    cv::Mat thresh_grad;
    cv::threshold(gradient, thresh_grad, 0, 255, cv::THRESH_BINARY | cv::THRESH_TRIANGLE);

    // Appliquer une fermeture pour combler les trous dans les contours des fruits
    cv::morphologyEx(thresh_grad, thresh_grad, cv::MORPH_CLOSE, kernel);

    // Extraire les contours des fruits en utilisant l'algorithme RETR_TREE
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    cv::findContours(thresh_grad, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // Créer un masque pour extraire le fond de l'image de fruits
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
    for (int i = 0; i < contours.size(); i++) {
        for (int j = 0; j < contours[i].size(); j++) {
            contours[i][j].x = contours[i][j].x + left_vertical_band;
            contours[i][j].y = contours[i][j].y + top_horizontal_band;
        }
    }
    drawContours(mask, contours, -1, cv::Scalar(255), cv::FILLED);

    // Appliquer le masque pour extraire le fond de l'image de fruits
    cv::Mat background;
    bitwise_and(img, img, background, mask);

    // Enregistrer le fond de l'image de fruits
    imwrite("C:\\Users\\Melissa\\Documents\\FISE2\\S8\\chef_d_oeuvre\\couleur_CO_git\\code\\C++\\source\\test.jpg", background);

    return background;
}