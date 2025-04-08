import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Load training data (from previously saved .npz file)
data = np.load('data/fashion_mnist_train.npz')
train = np.load('data/fashion_mnist_train.npz')
test = np.load('data/fashion_mnist_test.npz')  # shape: (5000, 784)
train_data = data['data'][:5000]

# Show the sample mean image
mean_image = np.mean(train_data, axis=0)
plt.figure(figsize=(4, 4))
plt.imshow(mean_image.reshape(28, 28).T, cmap='gray', origin='upper')
plt.title("Sample Mean Image (Before Centering)")
plt.axis('off')
plt.savefig('img/sample_mean.png')

# Center the data (subtract the mean from each sample)
mean_vector = np.mean(train_data, axis=0)
X_centered = train_data - mean_vector

# Compute the covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort the eigenvalues (and corresponding eigenvectors) in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# Plot eigenvalues
plt.figure(figsize=(8, 4))
plt.plot(eigenvalues_sorted, marker='o')
plt.title('PCA Eigenvalues (Descending Order)')
plt.xlabel('Principal Component Index')
plt.ylabel('Eigenvalue (Variance)')
plt.grid(True)
plt.tight_layout()
plt.savefig('img/eigenvalues.png')

# Show top PCA bases (eigenvectors) as images
num_components = 20
plt.figure(figsize=(12, 9))
for i in range(num_components):
    plt.subplot(4, 5, i + 1)
    plt.imshow(eigenvectors_sorted[:, i].reshape(28, 28).T, cmap='gray', origin='upper')
    plt.title(f"PC {i+1}")
    plt.axis('off')

plt.suptitle("Top PCA Bases (Eigenvectors as Images)")
plt.tight_layout()
plt.savefig('img/eigenvectors')

X_train = train['data']
y_train = train['labels']
X_test = test['data']
y_test = test['labels']

# Show top PCA bases (eigenvectors) as images
mean_vector = np.mean(X_train, axis=0)
X_train_centered = X_train - mean_vector
X_test_centered = X_test - mean_vector  # use *same mean* to center test data

# PCA
cov_matrix = np.cov(X_train_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# Choose dimensions 
dims = np.unique(np.logspace(np.log10(1), np.log10(350), 25, dtype=int))
train_errors = []
test_errors = []

for d in dims:
    # Project
    W = eigenvectors_sorted[:, :d]
    X_train_pca = X_train_centered @ W
    X_test_pca = X_test_centered @ W

    classes = np.unique(y_train)
    class_means = []
    class_covs = []

    for c in classes:
        X_c = X_train_pca[y_train == c]
        mu = np.mean(X_c, axis=0)
        cov = np.cov(X_c, rowvar=False) + 1e-5 * np.eye(d)
        class_means.append(mu)
        class_covs.append(cov)

    # Predict training set
    train_preds = []
    for x in X_train_pca:
        probs = [multivariate_normal.logpdf(x, mean=class_means[c], cov=class_covs[c])
                 for c in range(len(classes))]
        train_preds.append(np.argmax(probs))
    train_error = 1 - np.mean(train_preds == y_train)
    train_errors.append(train_error)

    # Predict test set
    test_preds = []
    for x in X_test_pca:
        probs = [multivariate_normal.logpdf(x, mean=class_means[c], cov=class_covs[c])
                 for c in range(len(classes))]
        test_preds.append(np.argmax(probs))
    test_error = 1 - np.mean(test_preds == y_test)
    test_errors.append(test_error)

    print(f"Dim {d}: Train error = {train_error:.4f}, Test error = {test_error:.4f}")

# Plotting
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(dims, train_errors, marker='o')
plt.title("Training Error vs Subspace Dimension")
plt.xlabel("PCA Dimension")
plt.ylabel("Classification Error")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(dims, test_errors, marker='o')
plt.title("Test Error vs Subspace Dimension")
plt.xlabel("PCA Dimension")
plt.ylabel("Classification Error")
plt.grid(True)

plt.tight_layout()
plt.savefig('img/pca_results')