# ---------------------------
# MNIST Digit Recognition - Full Professional Model (scikit-learn)
# ---------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ---------------------------
# 1. Load MNIST Dataset
# ---------------------------
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32')  # Features (flattened 28x28 images)
y = mnist.target.astype('int')    # Labels (0-9)

# ---------------------------
# 2. Preprocess Data
# ---------------------------
# Normalize pixel values to [0,1]
X = X / 255.0

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1666, random_state=42
)  # ~60k train, 10k test

# ---------------------------
# 3. Initialize Model
# ---------------------------
# Random Forest classifier: simple, fast, accurate
model = RandomForestClassifier(
    n_estimators=200,  # Number of trees
    max_depth=25,      # Max depth of each tree
    random_state=42,
    n_jobs=-1          # Use all CPU cores
)

# ---------------------------
# 4. Train the Model
# ---------------------------
print("Training model...")
model.fit(X_train, y_train)
print("Training completed.")

# ---------------------------
# 5. Evaluate the Model
# ---------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy*100:.2f}%\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# 6. Visualize Predictions (Optional)
# ---------------------------
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X_test.iloc[i].values.reshape(28,28), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# ---------------------------
# 7. Save Model for Reuse
# ---------------------------
joblib.dump(model, "mnist_rf_model.joblib")
print("Model saved as 'mnist_rf_model.joblib'")

# ---------------------------
# 8. Prediction Function
# ---------------------------
def predict_digit(image):
    """
    Predict a single 28x28 grayscale image.
    Input: image -> numpy array of shape (28,28)
    Output: predicted digit (0-9)
    """
    image_flat = image.reshape(1, -1) / 255.0
    return model.predict(image_flat)[0]

# Example usage:
# predicted = predict_digit(X_test.iloc[0].values.reshape(28,28))
# print("Predicted digit:", predicted)