# 🧬 Breast Cancer Prediction: Neural Network Classification Model (TensorFlow/Keras)

**🚀 Project Overview**

This project uses a **Neural Network (NN)** to predict whether a breast tumor is *benign* or *malignant* based on clinical and diagnostic features from the **Breast Cancer dataset**.  
The notebook demonstrates data preprocessing, model training, evaluation using **TensorFlow/Keras**.


## 📊 Workflow

1. **Data Loading & Cleaning**  
   - Imported and inspected the breast cancer dataset for missing values and inconsistencies.  
   - Checked data balance and feature distributions.

2. **Feature Engineering & Preprocessing**  
   - Scaled numerical features for optimal neural network performance.  
   - Encoded categorical labels (Benign/Malignant) into binary format.

3. **Model Architecture**  
   - Built a feedforward **Neural Network** using TensorFlow/Keras.  
   - Experimented with activation functions (ReLU, Sigmoid) and dropout layers to prevent overfitting.  

4. **Model Training & Evaluation**  
   - Trained on the processed dataset with validation tracking.  
   - Evaluated the model using metrics such as accuracy, precision, recall, and F1-score.  
   - Visualized learning curves and confusion matrix.

---

## 🧠 Model Overview

The neural network was designed for binary classification with a simple yet powerful architecture:

| Layer Type | Units | Activation |
|-------------|--------|------------|
| Dense (Input) | 32 | ReLU |
| Dense (Hidden) | 16 | ReLU |
| Dense (Hidden) | 8 | ReLU |
| Dense (Output) | 1 | Sigmoid |

**Optimizer:** Adam  
**Loss Function:** Binary Crossentropy  
**Evaluation Metrics:** Accuracy

**Input:**  
- Mean Radius  
- Mean Texture  
- Mean Smoothness  
- Mean Compactness  
- Mean Symmetry  

**Output:**  
 **Malignant (1) Benign (0)**

---


**Sample Results (update with yours):**
- Accuracy: 0.97
- Test loss: 0.1129

---

## 🧩 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **Matplotlib, Seaborn**
- **Scikit-learn**

---

## 📁 Repository Contents

- `Breast_Cancer_With_NN.ipynb` – Jupyter Notebook containing the entire workflow  
- `breast_cancer_data.csv` – Dataset used for model training and testing  
- `nn_breast_cancer_model.h5` – Saved trained Neural Network model (if exported)  




