import cv2
import numpy as np
import pickle
from skimage import feature
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path
import datetime

# Charger le modèle
with open('trained_model.pkl', 'rb') as f:
    clf = pickle.load(f)

# Charger le scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Les modifications commencent ici
FRUITS = {
    0: ('Abricot', (datetime.date(2023, 5, 15), datetime.date(2023, 6, 20))),
    1: ('Pomme', None),
    2: ('Banane', None),
    3: ('Myrtille', (datetime.date(2023, 7, 1), datetime.date(2023, 11, 1))),
    4: ('Raisin', (datetime.date(2023, 8, 1), datetime.date(2023, 10, 31))),
    5: ('Kiwi', (datetime.date(2023, 11, 6), datetime.date(2023, 6, 15))),
    6: ('Orange', (datetime.date(2023, 12, 1), datetime.date(2023, 4, 30))),
    7: ('Poire', (datetime.date(2023, 7, 1), datetime.date(2023, 4, 30))),
}

def is_in_season(fruit, today=None):
    if today is None:
        today = datetime.date.today()

    season = FRUITS[fruit][1]
    if season is None:
        return "Oui"  # Toute l'année
    start, end = season
    if start < end:
        return "Oui" if start <= today <= end else "Non"
    else:
        return "Oui" if not (end < today < start) else "Non"

# Les modifications se terminent ici
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

def extract_features(image):
    color_hist = color_histogram(image)
    glcm = glcm_features(image)
    contour = contour_features(image)
    features = np.concatenate((color_hist, glcm, contour))
    return features
# Le reste du code reste inchangé, sauf pour la fonction `predict_class` et la boucle principale

def predict_class(image):
    resized_image = cv2.resize(image, (256, 256))
    features = extract_features(resized_image)
    features_scaled = scaler.transform([features])
    predicted_class = clf.predict(features_scaled)[0]
    fruit_name = FRUITS[predicted_class][0]
    in_season = is_in_season(predicted_class)
    return fruit_name, in_season

# Accéder à la webcam
cap = cv2.VideoCapture(0)

text = ""
text_color = (0, 255, 0)  # Vert par défaut

while True:
    # Capturer l'image de la webcam
    ret, frame = cap.read()

    # Détecter un appui sur une touche
    key = cv2.waitKey(1)

    # Si la touche 'c' est enfoncée, capturer l'image et prédire la classe
    if key == ord('c'):
        print("Image capturée")
        fruit_name, in_season = predict_class(frame)
        print(f"Classe prédite: {fruit_name}, de saison: {in_season}")

        text = f"{fruit_name}, de saison: {in_season}"
        text_color = (0, 255, 0) if in_season == "Oui" else (0, 0, 255)  # Vert si de saison, sinon rouge

    # Afficher le nom du fruit et la saisonnalité sur l'image
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Afficher l'image
    cv2.imshow('Webcam', frame)

    # Si la touche 'q' est enfoncée, quitter la boucle
    if key == ord('q'):
        break

# Libérer la webcam et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
