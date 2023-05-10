import cv2
import numpy as np

# Charger l'image
img = cv2.imread('C:\\Users\\Alex\\Desktop\\FISE 2\\Semestre 8\\FruitClassification\\apple\\0.jpg')

# Convertir en niveau de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer le seuillage adaptatif
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Appliquer l'érosion et la dilatation pour améliorer la segmentation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Appliquer le masque pour extraire le fruit
result = cv2.bitwise_and(img, img, mask=mask)

# Enregistrer l'image résultante
cv2.imwrite("fruit_segmented.jpg", result)

def dominant_hue(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Filtrer les pixels noirs et blancs
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

print(dominant_hue(result))
