Customer Churn Prediction using Machine Learning
This project aims to predict customer churn for a telecommunications company. By analyzing historical customer data, we build a machine learning model to identify customers who are at high risk of leaving. This allows the business to proactively take retention-focused actions, such as offering special discounts or improved services to at-risk customers, thereby reducing revenue loss.

📋 Table of Contents
-Project Overview

-Dataset

-Project Workflow

-Key Findings

-Model Performance

-Technologies Used

-How to Run This Project

📝 Project Overview
The project follows a standard data science lifecycle:

Data Cleaning & Preprocessing: Loaded the dataset, handled missing values, and converted data types for analysis.

Exploratory Data Analysis (EDA): Visualized data to uncover patterns and relationships between customer attributes and churn.

Feature Engineering: Transformed categorical features into a numerical format using one-hot encoding.

Handling Class Imbalance: Addressed the imbalanced nature of the churn dataset using the SMOTE (Synthetic Minority Over-sampling Technique) on the training data.

Model Building: Trained a RandomForestClassifier model, which is well-suited for this type of classification task.

Model Evaluation: Assessed the model's performance on a held-out test set using metrics like Precision, Recall, F1-Score, and ROC AUC Score.

Interpretation: Analyzed feature importances to understand the key drivers of customer churn.

📊 Dataset
The dataset used is the Telco Customer Churn dataset, sourced from Kaggle. It contains information about customer demographics, subscribed services, account details, and their churn status.

🚀 Project Workflow
The dataset is loaded and cleaned. TotalCharges is converted to a numeric type, and rows with missing values are removed.

EDA is performed to visualize the churn rate across different categories like Contract type, InternetService, etc.

Categorical features are one-hot encoded, and the data is split into training (80%) and testing (20%) sets.

SMOTE is applied to the training set to create a balanced distribution of churned vs. non-churned customers.

Data is scaled using StandardScaler.

A RandomForestClassifier is trained on the preprocessed training data.

The model is evaluated on the unseen test set, and its performance is visualized using a confusion matrix and a feature importance plot.

💡 Key Findings
The analysis and model identified several key factors driving customer churn:

Contract Type: Customers on a Month-to-month contract are significantly more likely to churn compared to those on one or two-year contracts.

Tenure: New customers (low tenure) have a much higher churn rate.

Internet Service: Customers with Fiber optic internet service showed a higher churn rate, possibly indicating issues with price or service quality for that specific offering.

📈 Model Performance
The final RandomForestClassifier achieved the following performance on the test set:

ROC AUC Score: 0.84

F1-Score (for Churn class): 0.61

Recall (for Churn class): 0.77

Precision (for Churn class): 0.51

The model demonstrated a strong ability to identify customers who are genuinely at risk of churning (high Recall), which is crucial for the business to minimize missed opportunities for retention.

💻 Technologies Used
Python 3.x

Pandas for data manipulation

NumPy for numerical operations

Matplotlib & Seaborn for data visualization

Scikit-learn for machine learning models and evaluation

Imbalanced-learn for handling class imbalance (SMOTE)

⚙️ How to Run This Project
Clone the repository:

Bash

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
Create a virtual environment (optional but recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required libraries:

Bash

pip install -r requirements.txt
(Note: Create a requirements.txt file in your project by running pip freeze > requirements.txt)

Run the analysis:
Open and run the Jupyter Notebook churn_prediction.ipynb or execute the Python script.

Bash

jupyter notebook churn_prediction.ipynb
