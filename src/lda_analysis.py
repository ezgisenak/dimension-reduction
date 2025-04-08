import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the split dataset
train = np.load('data/fashion_mnist_train.npz')
test = np.load('data/fashion_mnist_test.npz')
X_train, y_train = train['data'], train['labels']
X_test, y_test = test['data'], test['labels']

# Apply LDA (maximum 9 components for 10 classes)
lda = LinearDiscriminantAnalysis(n_components=9)
lda.fit(X_train, y_train)

# Get LDA basis vectors (scalings_.T gives components as rows)
lda_components = lda.scalings_.T  # Shape: (9, 784)

# Plot and save the basis images
fig, axs = plt.subplots(1, 9, figsize=(18, 2))
for i in range(9):
    axs[i].imshow(lda_components[i].reshape(28, 28).T, cmap='gray', origin='upper')
    axs[i].axis('off')
    axs[i].set_title(f"LDA {i+1}")
plt.suptitle("LDA Basis Vectors (reshaped to 28x28 images)")
plt.tight_layout()
plt.savefig('img/lda_basis_vectors.png')
plt.show()

max_dim = 9
train_errors = []
test_errors = []

for d in range(1, max_dim + 1):
    # Fit LDA with d components
    lda = LinearDiscriminantAnalysis(n_components=d)
    X_train_proj = lda.fit_transform(X_train, y_train)
    X_test_proj = lda.transform(X_test)

    # Train Gaussian classifier
    clf = GaussianNB()
    clf.fit(X_train_proj, y_train)

    # Predict and calculate error
    y_train_pred = clf.predict(X_train_proj)
    y_test_pred = clf.predict(X_test_proj)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_errors.append(1 - train_acc)
    test_errors.append(1 - test_acc)

    print(f"Dimension {d}: Train Error = {1 - train_acc:.4f}, Test Error = {1 - test_acc:.4f}")

# Plot classification error vs. dimension
plt.figure(figsize=(10, 5))
plt.plot(range(1, max_dim + 1), train_errors, label='Train Error', marker='o')
plt.plot(range(1, max_dim + 1), test_errors, label='Test Error', marker='s')
plt.title("Classification Error vs LDA Subspace Dimension")
plt.xlabel("LDA Subspace Dimension")
plt.ylabel("Classification Error")
plt.grid(True)
plt.legend()
plt.xticks(range(1, max_dim + 1))
plt.savefig('img/lda_error_vs_dim.png')
plt.show()