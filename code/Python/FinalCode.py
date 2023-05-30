import cv2
import numpy as np
import pickle
from skimage import feature
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path
import datetime
import subprocess
import pandas as pd

# Charger le modèle
with open('trained_model.pkl', 'rb') as f:
    clf = pickle.load(f)

# Charger le scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

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


def predict_class(features):
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
        resized_image = cv2.resize(frame, (208, 256))
        cv2.imwrite("../../images/application/capture.jpg", resized_image)
        subprocess.call("../C++/executables/Fruitse.exe /images/application/capture.jpg")
        data = pd.read_csv("../../datasets/test1.csv")
        fruit_name, in_season = predict_class(data)
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
