Bank Churn Prediction using Artificial Neural Networks (ANN)

## Project Overview

This project is an End-to-End Deep Learning application that predicts whether a bank customer is likely to leave the bank (churn) or stay.

The model is built using an Artificial Neural Network (ANN) implemented with TensorFlow and Keras. The application includes:

* Data preprocessing and feature engineering
* Categorical feature encoding
* Feature scaling
* ANN model training and evaluation
* Model serialization using Pickle and H5
* Interactive Streamlit web application
* Deployment-ready project structure

---

## Problem Statement

Banks lose significant revenue when customers leave. Predicting customer churn allows businesses to take proactive actions and improve customer retention.

### Objective

Predict whether a customer will leave the bank based on their demographic and account information.

### Target Variable

| Variable | Description                            |
| -------- | -------------------------------------- |
| Exited   | 1 = Customer Left, 0 = Customer Stayed |

---

##  Dataset

**Dataset:** `Churn_Modelling.csv`

### Features Used

* CreditScore
* Geography
* Gender
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary

### Features Removed

The following columns were removed because they do not contribute to churn prediction:

* RowNumber
* CustomerId
* Surname

---

## Tech Stack

### Machine Learning & Deep Learning

* TensorFlow
* Keras
* Scikit-Learn
* NumPy
* Pandas

### Visualization & Monitoring

* TensorBoard
* Matplotlib

### Deployment

* Streamlit
* Streamlit Cloud




---

## ⚙️ Data Preprocessing

### 1. Label Encoding

The `Gender` column contains only two categories.

```python
Male   -> 1
Female -> 0
```

### 2. One-Hot Encoding

The `Geography` column is transformed into:

```text
Geography_France
Geography_Germany
Geography_Spain
```

### 3. Feature Scaling

All numerical features are standardized using:

```python
StandardScaler()
```

This ensures that features with large values do not dominate training.

---

## 🏗️ ANN Architecture

The model uses a Sequential Neural Network.

### Architecture

```text
Input Layer
      ↓
Dense(64, ReLU)
      ↓
Dense(32, ReLU)
      ↓
Dense(1, Sigmoid)
```

### Hidden Layers

```python
Dense(64, activation='relu')
Dense(32, activation='relu')
```

### Output Layer

```python
Dense(1, activation='sigmoid')
```

Since this is a binary classification problem, Sigmoid outputs a probability between 0 and 1.

---

## 📈 Model Compilation

```python
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Components

| Component     | Value               |
| ------------- | ------------------- |
| Optimizer     | Adam                |
| Loss Function | Binary Crossentropy |
| Metric        | Accuracy            |

---

## 🚀 Training Strategy

### TensorBoard

Used to visualize:

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy

### Early Stopping

Stops training when validation loss stops improving.

```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

Benefits:

* Prevents overfitting
* Saves training time
* Restores best model weights

---

## Model Saving

The trained model and preprocessing artifacts are saved for future inference.

### Saved Files

```text
model.h5
scaler.pkl
label_encoder_gender.pkl
onehot_encoder_geo.pkl
```

---

## Making Predictions

For every new customer:

### Step 1

Load:

* ANN Model
* Scaler
* Encoders

### Step 2

Apply the same preprocessing:

* Label Encoding
* One-Hot Encoding
* Feature Scaling

### Step 3

Generate prediction:

```python
prediction = model.predict(input_scaled)
```

### Interpretation

```python
if prediction > 0.5:
    print("Customer likely to churn")
else:
    print("Customer likely to stay")
```

---

##  Streamlit Web Application

The project includes a Streamlit frontend where users can:

* Select Geography
* Select Gender
* Enter Credit Score
* Enter Balance
* Enter Salary
* Adjust Age and Tenure
* Predict Churn Probability

### Run Locally

```bash
streamlit run app.py
```

Default URL:

```text
http://localhost:8501
```

---

##  Installation

### Clone Repository

```bash
git clone https://github.com/your-username/bank-churn-prediction-ann.git

cd bank-churn-prediction-ann
```

### Create Virtual Environment

```bash
conda create -p venv python=3.11 -y
```

### Activate Environment

```bash
conda activate venv
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Requirements

```text
tensorflow
pandas
numpy
scikit-learn
matplotlib
tensorboard
streamlit
```

---

##  Deployment

The application can be deployed using Streamlit Cloud.

### Deployment Steps

1. Push project to GitHub
2. Login to Streamlit Cloud
3. Create New App
4. Connect GitHub Repository
5. Select:

```text
Repository → Your Repository
Branch     → main
File       → app.py
```

6. Click Deploy

---

## Future Improvements

* Hyperparameter Tuning
* Cross Validation
* Model Explainability using SHAP
* Docker Containerization
* CI/CD Pipeline
* Cloud Deployment on AWS/GCP/Azure
* Advanced Deep Learning Architectures

---

## Learning Outcomes

Through this project, you will understand:

* Data Preprocessing
* Feature Engineering
* ANN Architecture Design
* Forward & Backward Propagation
* Loss Functions
* Optimizers
* Model Evaluation
* TensorBoard Monitoring
* Model Serialization
* Streamlit Deployment
* End-to-End Machine Learning Workflow

---

##  Author

Ayush Pandey

