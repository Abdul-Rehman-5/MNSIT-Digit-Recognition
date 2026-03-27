# MNIST Digit Recognition using Random Forest

This project is a **machine learning model** that recognizes handwritten digits (0-9) using the **MNIST dataset**. The model uses a **Random Forest classifier** from **scikit-learn** and is fully offline once the dataset is downloaded.

---

## 🚀 Features

- Simple, **fast, and accurate** Random Forest model (~97% test accuracy).
- Fully **offline after first download**.
- **Preprocessing included**: normalization and flattening of 28x28 images.
- **Evaluation**: accuracy score, classification report, and confusion matrix.
- **Visualization**: optional display of predictions.
- **Reusable model**: saved using `joblib`.
- Ready-to-use **prediction function** for single images.

---

## 📦 Requirements

- Python 3.8+
- Libraries:
  - pandas
  - scikit-learn
  - matplotlib
  - joblib

Install dependencies using:

```bash
pip install pandas scikit-learn matplotlib joblib

Note: The trained model file is not included due to GitHub size limits. Run the script to generate it locally.