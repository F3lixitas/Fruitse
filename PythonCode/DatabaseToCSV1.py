import cv2
import os
import numpy as np
import pandas as pd
from skimage import feature
from pathlib import Path

def dominant_hue(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    result = cv2.bitwise_and(image, image, mask=mask)

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    exclude_mask = cv2.bitwise_or(black_mask, white_mask)
    hue = hsv[:, :, 0]
    hue_filtered = np.ma.masked_array(hue, mask=exclude_mask)

    median_hue = np.ma.median(hue_filtered)
    return median_hue

def lbp_features(image, scales, P=8, R=1):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_histograms = []
    for scale in scales:
        resized_image = cv2.resize(gray_image, (0, 0), fx=scale, fy=scale)
        lbp = feature.local_binary_pattern(resized_image, P, R, method='uniform')
        lbp_histogram = np.histogram(lbp, bins=range(0, P + 3), density=True)[0]
        lbp_histograms.extend(lbp_histogram)
    return np.array(lbp_histograms)


def hu_moments(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    moments = cv2.moments(threshold_image)
    hu_moments = cv2.HuMoments(moments)
    return np.array(hu_moments).flatten()


def process_images(input_folder, output_folder):
    data = []
    scales = [1, 0.5, 0.25]
    class_number = 0
    min_images = float('inf')

    folder_list = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]
    for folder in folder_list:
        folder_path = os.path.join(input_folder, folder)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        min_images = min(min_images, len(image_files))

    for folder in folder_list:
        folder_path = os.path.join(input_folder, folder)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in image_files[:min_images]:
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            hue = dominant_hue(image)
            lbp = lbp_features(image, scales, P=8, R=1)
            hu = hu_moments(image)

            features = [class_number, hue, lbp, hu]
            data.append(features)
        class_number += 1

    columns = ['Classe', 'Caractéristique Couleur', 'Caractéristique LPB', 'Caractéristique Fréquentielle']
    df = pd.DataFrame(data, columns=columns)
    output_csv_path = os.path.join(output_folder, 'features.csv')
    df.to_csv(output_csv_path, index=False)

input_folder = os.path.join("C:\\", "Users", "Alex", "Desktop", "FISE 2", "Semestre 8", "FruitClassification")
output_folder = os.path.join("C:\\", "Users", "Alex", "Desktop", "FISE 2", "Semestre 8", "FruitClassification")
process_images(input_folder, output_folder)


import csv

n = 1  # Remplacez 3 par le numéro de la colonne souhaitée
filename_in = "features.csv"  # Remplacez input.csv par le nom de votre fichier d'entrée
filename_out = "output.csv"  # Remplacez output.csv par le nom de votre fichier de sortie

with open(filename_in, 'r') as file_in, open(filename_out, 'w', newline='') as file_out:
    reader = csv.reader(file_in)
    writer = csv.writer(file_out)
    for row in reader:
        writer.writerow([row[n-1]])