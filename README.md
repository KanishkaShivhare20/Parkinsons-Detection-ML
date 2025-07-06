# 🧠 Parkinson's Disease Detection using Machine Learning

This project uses a Support Vector Machine (SVM) to detect Parkinson's Disease based on biomedical voice measurements. It is built in Python and includes data preprocessing, model training, and evaluation.

---

## 📌 Problem Statement

Early detection of Parkinson's Disease is crucial for treatment. This project uses an ML model trained on voice-based features to predict if a person has Parkinson's Disease.

---

## 📂 Dataset

- **Source**: [UCI Parkinson’s Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Rows**: 195 samples  
- **Features**: 22 voice-based biomedical features  
- **Target**: `status` (0 = Healthy, 1 = Parkinson’s)

---

## 🚀 Technologies Used

- Python (Jupyter Notebook)
- Pandas, NumPy
- Scikit-learn (SVM, accuracy)
- Matplotlib, Seaborn

---

## 📊 Results

- **Training Accuracy**: 95.00%
- **Testing Accuracy**: 94.87%

---

## 🧪 How to Run

1. Clone the repository  
2. Navigate to the project folder  
3. Run the Jupyter notebook inside `notebooks/`

```bash
jupyter notebook notebooks/Parkinsons_Prediction.ipynb
