import numpy as np
import matplotlib.pyplot as plt
import openml as oml
import os

# Charger le dataset Fashion MNIST
fmnist = oml.datasets.get_dataset(40996)
X, y, _, _ = fmnist.get_data(target=fmnist.default_target_attribute)

# Convertir X en tableau NumPy
X = X.to_numpy()
print(y)
# Créer un dossier pour sauvegarder les images
output_dir = "test_images"
os.makedirs(output_dir, exist_ok=True)

# Sauvegarder 10 images de test
for i in range(10,20):  # Vous pouvez changer le nombre d'images à sauvegarder
    image = X[i].reshape(28, 28)  # Redimensionner en 28x28 pixels
    image_path = os.path.join(output_dir, f"test_image_{i}.png")
    plt.imsave(image_path, image, cmap='gray')
    print(f"Image {i} sauvegardée sous {image_path}")