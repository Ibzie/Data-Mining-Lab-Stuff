import numpy as np
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import os
from sklearn.model_selection import train_test_split


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            print(img)
        labels.append(0 if 'cat' in filename else 1)
        return images, labels

def preprocess_images(images, target_size=(64, 64)):

    processed_images = []
    for img in images:
        img_resized = resize(img, target_size, anti_aliasing=True, mode='reflect')
    processed_images.append(img_resized.flatten())
    return np.array(processed_images)

folder = 'train/test_train'
images, labels = load_images_from_folder(folder)
processed_images = preprocess_images(images)


features_train, features_test, labels_train, labels_test = train_test_split(
    processed_images, labels, test_size=0.2, random_state=42)


def train_svm(features, labels, epochs=500, learning_rate=0.01, lambda_param=0.01):
    num_samples, num_features = features.shape
    weights = np.zeros(num_features)  # Initialize weight vector

    for epoch in range(epochs):
        for i in range(num_samples):
            if labels[i] == 0:
                labels[i] = -1  # Convert labels to -1 and 1 for SVM

            # Compute the margin and update weights using gradient descent
            margin = labels[i] * np.dot(features[i], weights)
            if margin < 1:
                weights = weights - learning_rate * (2 * lambda_param * weights - labels[i] * features[i])

    return weights

def evaluate_svm(weights, features, labels):
    num_correct = 0
    num_samples = features.shape[0]

    for i in range(num_samples):
        prediction = np.sign(np.dot(features[i], weights))
        if prediction == labels[i]:
            num_correct += 1

    accuracy = num_correct / num_samples
    return accuracy

# Train the SVM and evaluate on the test set
weights = train_svm(features_train, labels_train)
accuracy = evaluate_svm(weights, features_test, labels_test)

print("Accuracy of the SVM classifier: {:.2f}%".format(accuracy * 100))