import streamlit as st
from PIL import Image
import requests

# Titre de l'application
st.title("Classification d'images avec Fashion MNIST")

# Classes Fashion MNIST
fmnist_classes = [
    "T-shirt/top",  # Classe 0
    "Trouser",      # Classe 1
    "Pullover",     # Classe 2
    "Dress",        # Classe 3
    "Coat",         # Classe 4
    "Sandal",       # Classe 5
    "Shirt",        # Classe 6
    "Sneaker",      # Classe 7
    "Bag",          # Classe 8
    "Ankle boot"    # Classe 9
]

# URL de l'API Flask (remplacez par l'URL de votre API si nécessaire)
API_URL = "http://127.0.0.1:5000/predict"

# Téléchargement de l'image
uploaded_file = st.file_uploader("Téléchargez une image (28x28 pixels)", type=["png", "jpg", "jpeg"])

# Afficher l'image téléchargée une seule fois
if uploaded_file is not None:
    # Réinitialiser le pointeur du fichier pour éviter l'erreur PIL
    uploaded_file.seek(0)
    
    # Ouvrir l'image
    image = Image.open(uploaded_file).convert('L')  # Convertir en niveaux de gris
    
    # Afficher l'image avec une taille réduite
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:  # Utilise la colonne du milieu pour centrer l'image
        st.image(image, caption="Image téléchargée", width=150)
# Sélection du modèle
model_name = st.selectbox(
    "Sélectionnez un modèle",
    ["Logistic Regression", "Linear SVC", "KNN"]
)

# Bouton pour lancer la prédiction
if st.button("Prédire"):
    if uploaded_file is not None:
        # Réinitialiser le pointeur du fichier
        uploaded_file.seek(0)
        
        # Envoyer la requête à l'API Flask
        files = {"file": uploaded_file}
        data = {"model": model_name}
        response = requests.post(API_URL, files=files, data=data)

        # Afficher le résultat
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction")  # Correction : "prediction" -> "prediction"
            
            # Afficher la prédiction
            class_name = fmnist_classes[prediction]  # Convertir le numéro en nom de classe
            st.success(f"**Prédiction :** {class_name} (Classe {prediction})")
        else:
            st.error(f"Erreur lors de la prédiction : {response.text}")
    else:
        st.warning("Veuillez télécharger une image avant de lancer la prédiction.")