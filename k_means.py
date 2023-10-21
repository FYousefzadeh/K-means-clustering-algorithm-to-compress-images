import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def initialize_centroids(X, K):
    centroids_indices = np.random.choice(X.shape[0], size=K, replace=False)
    centroids = X[centroids_indices]
    return centroids


def assign_clusters(X, centroids):
    distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
    clusters = np.argmin(distances, axis=1)
    return clusters


def update_centroids(X, clusters, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        centroids[k] = np.mean(X[clusters == k], axis=0)
    return centroids


def kmeans(X, K, num_iterations=100):
    centroids = initialize_centroids(X, K)
    for _ in range(num_iterations):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, K)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters


def compress_image(image_path, num_colors):
    image = Image.open(image_path)
    image_array = np.array(image)
    pixels = image_array.reshape(-1, 3)
    normalized_pixels = pixels / 255.0
    centroids, clusters = kmeans(normalized_pixels, num_colors)
    compressed_pixels = centroids[clusters]
    compressed_pixels = np.round(compressed_pixels * 255).astype(int)
    compressed_image_array = compressed_pixels.reshape(image_array.shape)
    compressed_image = Image.fromarray(np.uint8(compressed_image_array))

    return compressed_image


image_path = r"\image_path"
num_colors = 16
compressed_image = compress_image(image_path, num_colors)

# displaying the result
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(Image.open(image_path))
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(compressed_image)
ax[1].set_title(f"Compressed Image ({num_colors} Colors)")
ax[1].axis("off")
plt.show()
