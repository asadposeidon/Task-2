# Credit Card Fraud Detection using Machine Learning

## Project Overview
This project builds a machine learning model to detect fraudulent credit card transactions. 
The model analyzes transaction data and classifies each transaction as fraudulent (1) or legitimate (0). 
Fraud detection helps financial institutions prevent financial loss and protect customers.

## Dataset
The dataset contains information about credit card transactions.

Files used:
fraudTrain.csv – training dataset
fraudTest.csv – testing dataset

Each record contains transaction details such as:
- transaction amount
- merchant
- category
- customer information
- transaction time
- fraud label (is_fraud)

Target variable:
is_fraud
0 → Legitimate transaction
1 → Fraudulent transaction

## Technologies Used
Python
Pandas
Scikit-learn
Matplotlib

## Machine Learning Algorithms Used

1. Logistic Regression  
A statistical model used for binary classification problems.

2. Decision Tree  
A tree-based model that splits data into branches based on feature values.

3. Random Forest  
An ensemble model that combines multiple decision trees to improve prediction accuracy.

## Project Workflow

1. Load transaction dataset
2. Remove unnecessary columns
3. Convert categorical data into numerical format
4. Split data into training and validation sets
5. Train machine learning models
6. Evaluate model performance
7. Visualize results using confusion matrix
8. Predict fraud transactions on test data

## Model Evaluation

Model performance is measured using:
- Accuracy
- Confusion Matrix

Example output:

Logistic Regression Accuracy: 0.95
Decision Tree Accuracy: 0.98
Random Forest Accuracy: 0.99

Random Forest generally gives the best performance.

## Visualization

A confusion matrix is plotted using Matplotlib to show:
- True Positives
- True Negatives
- False Positives
- False Negatives

This helps evaluate how well the model detects fraud.

## How to Run the Project

1. Install required libraries

pip install pandas scikit-learn matplotlib

2. Place dataset files in project folder

fraudTrain.csv
fraudTest.csv

3. Run the program

python model.py

4. Output

The program will display model accuracy and generate fraud predictions.

## Applications

This system can be used by:
- Banks
- Credit card companies
- Financial institutions

to automatically detect suspicious transactions.

## Future Improvements

Possible improvements include:
- Deep learning models
- Handling imbalanced datasets with SMOTE
- Real-time fraud detection systems
- Feature importance analysis
