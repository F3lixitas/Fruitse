import cv2
import os
import numpy as np
import pandas as pd
from skimage import feature
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy, moments_hu
from pathlib import Path

def color_mean_std(image):
    # Calculer la moyenne et l'écart-type pour chaque canal de couleur
    means, stds = cv2.meanStdDev(image)
    color_features = np.concatenate([means.flatten(), stds.flatten()])
    return color_features

def glcm_energy_homogeneity(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm_matrix = graycomatrix(gray_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    energy = graycoprops(glcm_matrix, prop='energy').flatten()
    homogeneity = graycoprops(glcm_matrix, prop='homogeneity').flatten()
    glcm_props = np.hstack((energy, homogeneity))
    return glcm_props

def shape_entropy(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = feature.canny(gray_image, sigma=3)
    entropy = shannon_entropy(edges)
    return np.array([entropy])

def hu_moments(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray_image)
    hu = cv2.HuMoments(moments).flatten()
    return hu

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
            color_features = color_mean_std(image)
            glcm = glcm_energy_homogeneity(image)
            shape = shape_entropy(image)
            hu = hu_moments(image)

            features = np.concatenate(([class_number], color_features, glcm, shape, hu))
            data.append(features)
        class_number += 1

    columns = ['Classe'] + [f'Couleur_{i}' for i in range(color_features.size)] + [f'Caractéristique GLCM_{i}' for i in range(glcm.size)] + [f'Entropie Forme'] + [f'Moments de Hu_{i}' for i in range(hu.size)]
    df = pd.DataFrame(data, columns=columns)
    output_csv_path = os.path.join(output_folder, 'features_reduced.csv')
    df.to_csv(output_csv_path, index=False)
input_folder = os.path.join("C:\\", "Users", "Alex", "Desktop", "FISE 2", "Semestre 8", "FruitClassification")
output_folder = os.path.join("C:\\", "Users", "Alex", "Desktop", "FISE 2", "Semestre 8", "FruitClassification")
process_images(input_folder, output_folder)
