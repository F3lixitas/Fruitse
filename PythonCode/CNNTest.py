import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Charger le modèle
model = load_model('fruit_classifier.h5')

# Dictionnaire pour convertir les indices de classe en noms de fruits
class_labels = {
    0: 'Abricot',
    1: 'Pomme',
    2: 'Banane',
    3: 'Myrtille',
    4: 'Raisin',
    5: 'Kiwi',
    6: 'Orange',
    7: 'Poire'
}

def preprocess_image(image):
    # Redimensionner l'image à la taille d'entrée du modèle
    image = cv2.resize(image, (100, 100))
    # Convertir l'image en tableau numpy et normaliser
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255
    return image

def predict_fruit(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_idx = np.argmax(prediction)
    return class_labels[class_idx]

# Initialiser la capture vidéo (utilisez 1 pour une deuxième caméra)
cap = cv2.VideoCapture(0)

while True:
    # Capturer l'image de la webcam
    ret, frame = cap.read()

    # Afficher l'image capturée
    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF

    # Si la touche 'c' est appuyée, capturer l'image et prédire la classe
    if key == ord('c'):
        fruit_name = predict_fruit(frame)
        print('Fruit détecté :', fruit_name)
        # Afficher le nom du fruit sur l'image capturée
        cv2.putText(frame, fruit_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Prediction', frame)

    # Si la touche 'q' est appuyée, quitter la boucle
    if key == ord('q'):
        break

# Fermer toutes les fenêtres et relâcher la capture vidéo
cap.release()
cv2.destroyAllWindows()
