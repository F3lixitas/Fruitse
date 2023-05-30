# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# Lecture du fichier CSV
df = pd.read_csv('../../datasets/sortie.csv')

# Séparation des caractéristiques et des étiquettes
X = df.drop('Classe', axis=1)  # Assurez-vous que 'classe' est le nom de votre colonne d'étiquette
y = df['Classe']

# Normalisation des caractéristiques
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Division des données en ensemble d'apprentissage et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle K-NN
knn = KNeighborsClassifier(n_neighbors=4)  # Vous pouvez choisir un autre nombre de voisins si vous le souhaitez

# Formation du modèle
knn.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = knn.predict(X_test)

# Calcul de l'accuracy
accuracy = accuracy_score(y_test, y_pred)

# Affichage de l'accuracy
print('Accuracy du modèle :', accuracy)

print('Nombre d\'images par classe dans l\'ensemble de test :\n', y_test.value_counts())

# Calcul de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Affichage de la matrice de confusion
print('Matrice de confusion :\n', cm)

# Application de la validation croisée k-fold
scores = cross_val_score(knn, X, y, cv=5)  # Vous pouvez choisir un autre nombre de 'folds' si vous le souhaitez

# Affichage des scores de la validation croisée
print('Scores de la validation croisée :', scores)
print('Score moyen de la validation croisée :', scores.mean())


import pickle

# Enregistrer le modèle dans un fichier pickle
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(knn, file)

# Enregistrer le scaler dans un fichier pickle
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
