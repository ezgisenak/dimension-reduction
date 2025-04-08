import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt('data/fashion_mnist/fashion_mnist_data.txt')
labels = np.loadtxt('data/fashion_mnist/fashion_mnist_labels.txt')


# Choose an item to display
i = 100 

# Extract and reshape the image
image = data[i].reshape(28, 28)

# Display the image
plt.imshow(image.reshape(28, 28).T, cmap='gray', origin='upper')
plt.axis('off')  # Hide axes
plt.title(f"Sample #{i}")
plt.savefig('img/sample_image.png')

# Set the random seed 
np.random.seed(42)

# Preallocate lists for training and testing data
train_data = []
train_labels = []
test_data = []
test_labels = []

# Number of classes
num_classes = 10

# Loop through each class
for class_label in range(num_classes):
    # Get all indices of current class
    class_indices = np.where(labels == class_label)[0]
    
    # Shuffle the indices
    np.random.shuffle(class_indices)
    
    # Split indices into 50% train, 50% test
    split = len(class_indices) // 2
    train_idx = class_indices[:split]
    test_idx = class_indices[split:]
    
    # Append to corresponding lists
    train_data.append(data[train_idx])
    train_labels.append(labels[train_idx])
    
    test_data.append(data[test_idx])
    test_labels.append(labels[test_idx])

# Concatenate all class data
train_data = np.vstack(train_data)
train_labels = np.hstack(train_labels)

test_data = np.vstack(test_data)
test_labels = np.hstack(test_labels)

# Save the data
np.savez('data/fashion_mnist_train.npz', data=train_data, labels=train_labels)
np.savez('data/fashion_mnist_test.npz', data=test_data, labels=test_labels)

print("Dataset split complete.")
