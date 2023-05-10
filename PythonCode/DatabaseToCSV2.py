import cv2
import os
import numpy as np
import pandas as pd
from skimage import feature
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path

def color_histogram(image):
    channels = ('b', 'g', 'r')
    hist_features = []
    for channel in channels:
        hist = cv2.calcHist([image], [channels.index(channel)], None, [256], [0, 256])
        hist_features.extend(hist.flatten())
    return np.array(hist_features)

def glcm_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm_matrix = graycomatrix(gray_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    contrast = graycoprops(glcm_matrix, prop='contrast')
    correlation = graycoprops(glcm_matrix, prop='correlation')
    energy = graycoprops(glcm_matrix, prop='energy')
    homogeneity = graycoprops(glcm_matrix, prop='homogeneity')
    glcm_props = np.hstack((contrast.flatten(), correlation.flatten(), energy.flatten(), homogeneity.flatten()))
    return glcm_props

def contour_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_length = sum([cv2.arcLength(contour, True) for contour in contours])
    num_contours = len(contours)
    avg_contour_length = total_length / num_contours if num_contours > 0 else 0
    contour_stats = np.array([total_length, num_contours, avg_contour_length])
    return contour_stats

def process_images(input_folder, output_folder):
    data = []
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
            color_hist = color_histogram(image)
            glcm = glcm_features(image)
            contour = contour_features(image)

            features = np.concatenate(([class_number], color_hist, glcm, contour))
            data.append(features)
        class_number += 1

    columns = ['Classe'] + [f'Histogramme Couleur_{i}' for i in range(color_hist.size)] + [f'Caractéristique GLCM_{i}' for i in range(glcm.size)] + [f'Caractéristique Contours_{i}' for i in range(contour.size)]
    df = pd.DataFrame(data, columns=columns)
    output_csv_path = os.path.join(output_folder, 'features_new.csv')
    df.to_csv(output_csv_path, index=False)

input_folder = os.path.join("C:\\", "Users", "Alex", "Desktop", "FISE 2", "Semestre 8", "FruitClassification")
output_folder = os.path.join("C:\\", "Users", "Alex", "Desktop", "FISE 2", "Semestre 8", "FruitClassification")
process_images(input_folder, output_folder)


import csv

n = 1  # Remplacez 3 par le numéro de la colonne souhaitée
filename_in = "features_new.csv"  # Remplacez input.csv par le nom de votre fichier d'entrée
filename_out = "output_new.csv"  # Remplacez output.csv par le nom de votre fichier de sortie

with open(filename_in, 'r') as file_in, open(filename_out, 'w', newline='') as file_out:
    reader = csv.reader(file_in)
    writer = csv.writer(file_out)
    for row in reader:
        writer.writerow([row[n-1]])

