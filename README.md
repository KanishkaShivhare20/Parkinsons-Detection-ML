# 🧠 Parkinson's Disease Detection using Machine Learning

This project uses a Support Vector Machine (SVM) to detect Parkinson's Disease based on biomedical voice measurements. It is built in Python and includes data preprocessing, model training, evaluation, and model saving for later use.

---

## 📌 Problem Statement

Early detection of Parkinson's Disease is crucial for treatment. This project uses a machine learning model trained on voice-based features to predict whether a person has Parkinson's Disease.

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
- Scikit-learn (SVM, accuracy, model saving)
- Matplotlib, Seaborn
- `joblib` (to save model and scaler)

---

## 📊 Results

- **Training Accuracy**: 91.03%  
- **Testing Accuracy**: 84.62%

---

## 💾 Model Saving

After training, the model and scaler are saved using `joblib` for future use:

```python
joblib.dump(model, 'model/svm_model.joblib')
joblib.dump(scaler, 'model/scaler.joblib')
