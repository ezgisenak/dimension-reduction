import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

# Load the full dataset (not just train/test splits)
data = np.loadtxt('data/fashion_mnist/fashion_mnist_data.txt')
labels = np.loadtxt('data/fashion_mnist/fashion_mnist_labels.txt')

# Optional: Standardize the features (helps t-SNE)
data_std = StandardScaler().fit_transform(data)

# t-SNE parameters
tsne = TSNE(
    n_components=2,
    perplexity=30,        # A good starting point
    max_iter=1000,        # Number of iterations
    init='pca',           # Initialization method
    random_state=42       # Set seed
)

print("Running t-SNE... This may take a minute.")
data_tsne = tsne.fit_transform(data_std)

# Plotting the 2D scatter with labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)

# Custom legend using class names
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(i)), label=class_names[i]) for i in range(10)]
plt.legend(handles=handles, title="Classes", loc="best", fontsize=8)

plt.title("t-SNE Visualization of Fashion-MNIST (2D)")
plt.axis('off')
plt.tight_layout()
plt.savefig('img/tsne_visualization.png')
