# Heart Disease Prediction

This project aims to predict whether a person has heart disease based on medical attributes. The dataset used for this analysis was sourced from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) to train and test our models. Multiple machine learning models are employed to compare their performance and identify the most accurate one.

### Features:
The dataset contains several medical features, including:
- Age
- Gender
- Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Resting Electrocardiographic Results (restecg)
- Maximum Heart Rate Achieved (thalach)
- Exercise-Induced Angina (exang)
- Oldpeak (ST depression induced by exercise relative to rest)
- Slope of the peak exercise ST segment

### Steps Involved:
1. **Data Collection**: 
   - Used 13 features and 1 target variable
   - Dataset includes both categorical and continuous features

2. **Data Preprocessing**:
   - Removed duplicates and null values
   - Renamed columns for better understanding
   - Performed feature scaling using StandardScaler
   - Created dummy variables for categorical features

3. **Data Visualization**:
   We used **Matplotlib** and **Seaborn** for data visualization, generating graphs to explore various scenarios:
    - Distribution of heart disease among males and females.
    - Relationship between heart disease, age, and blood pressure.
    - Correlation between heart disease, age, and heart rate.
    We confirmed that the dataset's two label classes were well-balanced, allowing us to continue without dropping or increasing data points.

4. **Feature Scaling**:
   Feature scaling was performed to standardize the range of independent variables:
    - Used the `get_dummies` method to create dummy variables for categorical features.
    - Applied `StandardScaler` to scale continuous features.
    These steps were taken to enhance the performance of our machine learning models.

5. **Dataset Splitting**:
   The dataset was split into:
    - **Training Set**: 80% of the data
    - **Test Set**: 20% of the data

6. **Model Implementation**:
We implemented several supervised learning algorithms to predict heart disease. The models used in this project are:
- **Logistic Regression**
- **K-Nearest Neighbors Classifier**
- **Support Vector Classifier (SVC)**
- **Decision Tree Classifier**

### i. K-Nearest Neighbors Classifier (KNN)
We varied the number of neighbors from 1 to 17 and calculated the test score for each. The optimal `k` value was selected based on the best score.

### ii. Support Vector Classifier (SVC)
We tested three different kernels (poly, sigmoid, and rbf) to find the best model. The sigmoid kernel yielded the highest accuracy.

### iii. Decision Tree Classifier
We varied the maximum number of features (from 1 to 20) and found that using 7 features produced the best results.

7. **Model Evaluation**:
   - Used confusion matrix to measure accuracy
   - Calculated precision and recall
    We used a **confusion matrix** to evaluate model performance, with the highest accuracy achieved by Logistic Regression at **93%**.
    
    ### Confusion Matrix Terms:
    - **True Positive (TP)**: The model correctly predicted heart disease.
    - **True Negative (TN)**: The model correctly predicted no heart disease.
    - **False Positive (FP)**: The model predicted heart disease, but the patient does not have it.
    - **False Negative (FN)**: The model predicted no heart disease, but the patient has it.
    
    ### Precision & Recall
    - **Precision**: TP / (TP + FP)
    - **Recall**: TP / (TP + FN)

In medical diagnosis, **high recall** is crucial to reduce the number of false negatives. Our model achieved a **recall score of 1.0** using logistic regression, indicating perfect performance in identifying heart disease cases.

## 8. Conclusion
In this project, we used Machine Learning to predict heart disease. After analyzing the dataset, we preprocessed it by handling categorical features and scaling the others. We implemented and fine-tuned four machine learning models:

- **Logistic Regression**: Best performance with 93% accuracy and 1.0 recall.
- **K Neighbors Classifier**
- **Support Vector Classifier (SVC)**
- **Decision Tree Classifier**

Among these models, **Logistic Regression** achieved the highest accuracy and recall, making it the most reliable for this task.

### Dependencies:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
