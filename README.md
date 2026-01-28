# E-commerce Churn Prevention System

This project builds an end-to-end e-commerce analytics and AI-driven decision support system. It converts raw transaction data into customer-level insights using feature engineering and churn modeling, evaluates retention strategies through expected value analysis, and enables natural-language access to insights via a Retrieval-Augmented Generation (RAG) interface. The system helps business teams identify at-risk customers, prioritize retention actions, and make data-backed decisions using both traditional analytics and generative AI.

## Basic Information
**Names:** N M Emran Hussain  
**Email:** nmemranhussain2023@gmail.com  
**Date:** October 2025  
**Model Version:** 1.0.0  
**License:** [Apache License Version 2.0,](LICENSE)

## Intended Use
**Purpose:** This project transforms raw e-commerce data into a decision-ready system by engineering customer behavior features, predicting churn risk, and evaluating retention strategies. It combines descriptive analytics, predictive modeling, and a Retrieval-Augmented Generation (RAG) layer, enabling users to query insights in natural language, all grounded in real customer data.  
**Intended Users:** Marketing and CRM teams, Product and Growth Managers, Data Analysts and Data Scientists, and Business leaders and decision-makers.  
**Out-of-scope Uses:** Operational users, Real-time transaction processing systems, End consumers / shoppers, Users expecting real-time personalization at scale, and Teams seeking compliance, fraud detection, or financial auditing solutions.

## Dataset
**Dataset Name:** Online Retail Dataset  
**Number of Samples:** The original dataset contains 541,909 rows. After the cleaning process—which includes removing duplicates and missing values—the notebook works with a processed set of 401,604 transactions.
**Features Used:** InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country
**Engineered Features:** Year, Month, Day, Hour, DayOfWeek, IsWeekend, TotalAmount, BasketUniqueItems, CustProductDiversity, CustTotalSales, and Churn_Label
**Data Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)

### Data Dictionary

|Column Name	      |Modeling Role	|Measurement Level	|Description|
|------------------|--------------|------------------|-----------|
|CustomerID	       | ID	          |Nominal	          |Unique identifier assigned to each customer.|
|Recency	          |Input	        |Interval	         |Number of days since the customer's last purchase relative to the dataset's reference date.|
|Frequency	        |Input	        |Ratio	            |Total number of unique purchase transactions (invoices) made by the customer.|
|Monetary	         |Input	        |Ratio	            |Total financial value of all successful purchases made by the customer.|
|Churn_Label	 |Target	|Binary	|Classification where 1 indicates a "Churned" customer (inactive for >30 days) and 0 indicates "Not Churned".|
|Country	|Input	|Nominal	|The country where the customer resides; used to analyze regional sales distribution.|
|TotalAmount	|Input	|Ratio	|The calculated value of an individual transaction (Quantity × UnitPrice).|
|BasketUniqueItems	|Input	|Ratio	|The number of distinct products contained within a single invoice.|
|CustProductDiversity	|Input	|Ratio	|Total number of unique StockCodes purchased by a customer over their lifetime.|
|IsWeekend	| Input	|Binary	|Flag indicating if a transaction occurred on a Saturday or Sunday.|
|Estimated_Reward	|Output	|Ratio	|The projected financial Return on Investment (ROI) for a specific retention action.|
|Chosen_Action|	Input	|Nominal	|The specific retention strategy (e.g., 'sms', 'email', 'call+coupon') recommended for the customer.|

## Training & Test Data

**Training Data Percentage:** 70% of the customer-level dataset (the RFM data) was used as training data.

**Testing Data Percentage:** The remaining 30% was reserved as a holdout test set to evaluate model performance.

### Splitting & Model Training Methodology
- The split was implemented using the train_test_split function from the sklearn library with a test_size parameter of 0.3. To address the severe class imbalance (where 62.3% of customers had churned), the split was performed with stratification on the churn label to ensure both the training and testing sets maintained the same proportion of churned vs. active customers.
- This training data was used to fit a Logistic Regression model (with feature scaling) and a Random Forest classifier. Both models utilized balanced class weights to further account for the imbalance in the training labels.

## Model Details
### Architecture  
- This model card utilizes linear model such as **Logistic Regression**. As an alternative model **Random Forest** is used.  

### Evaluation Metrics  
- AUC (Area Under the ROC Curve): Measures the model's ability to distinguish between positive and negative classes.

### Final Values of Metrics for All Data using 'logistic regression' model:

| Dataset     | AUC   | 
|-------------|-------|
| Training    | 0.78  | 
| Validation  | 0.80  |
| Test        | 0.76  | 

### Columns Used as Inputs in the Final Model
The following columns were used as inputs (features) in the final model:
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

### Column(s) Used as Target(s) in the Final Model
- **Target Column:** Survived

### Type of Models
* **[Logistic Regression Classifier](https://github.com/nmemranhussain/titanic-ml-models/blob/main/Titanic_logistic%20(1).ipynb)**
* **[Random Forest Classifier](https://github.com/nmemranhussain/titanic-ml-models/blob/main/Titanic_RF.ipynb)**

### Software Used to Implement the Model
- **Software:** Python (with libraries such as Pandas, Scikit-learn, seaborn & matplotlib)

### Version of the Modeling Software: 
- **'pandas'**: '2.2.2',
- **'scikit-learn'**: '1.4.2',
- **'seaborn'**: '0.13.2',
- **'matplotlib'**: '3.8.4**

### Hyperparameters or Other Settings of the Model
The following hyperparameters were used for the 'logistic regression' model:
- **Solver:** lbfgs
- **Maximum Iterations:** 100
- **Regularization (C):** 1.0
- **Features used in the model**: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
- **Target column**: Survived
- **Model type**: Logistic Regression
- **Hyperparameters**: Solver = lbfgs, Max iterations = 500, C = 1.0
- **Software used**: scikit-learn sklearn.linear_model._logistic

The following hyperparameters were used for the 'random forest' as an alternative model:
- **Columns used as inputs**: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 
- **Target column**: 'Survived',
- **Type of model**: 'Random Forest Classifier',
- **Software used**: 'scikit-learn',

## Quantitative Analysis

### Plots Related to Data or Final Model
 
![Plot of Survival Rate Vs. Passenger Class](SR_by_Class.png) 

**Description**: Passengers in 1st class had the highest survival rate, followed by those in 2nd class. 3rd class passengers had the lowest survival rate.

![Plot of Survival Rate Vs. Passenger Gender](SR_by_Gender.png) 

**Description**: Females had a significantly higher survival rate than males, aligning with the negative coefficient for the "Sex" feature in the logistic regression model.

![Plot of Survival Rate Vs. Passenger Age](SR_by_Age.png) 

**Description**: Children (ages 0-12) had the highest survival rate, while seniors (ages 50-80) had the lowest. Young adults and adults had relatively similar survival rates, though slightly lower than children.

## Potential Impacts, Risks, and Uncertainties using Logistic Regression & Random Forest Model ##
Logistic regression offers a powerful tool for classification tasks. However, it is crucial to acknowledge its limitations. The model assumes a linear relationship between features and the outcome, which could overlook complex patterns in the data. This can lead to biased predictions, particularly when dealing with sensitive attributes like gender or class. Additionally, the probabilistic nature of the output can be misinterpreted as deterministic, potentially leading to misinformed decisions. To mitigate these risks and promote responsible AI practices, this model development employed several strategies. First, the training data was thoroughly examined for potential disparities related to gender and class. Second, interpretability tools from libraries like PiML were used to analyze the model's decision-making process and its impact on different groups. By incorporating these responsible AI practices, we aimed to ensure fairer and more transparent outcomes from the logistic regression model.

While random forests boast strong performance in classification tasks, they also present challenges. Their complex structure can be difficult to interpret, hindering explainability. Despite resilience to noise, random forests can still be susceptible to overfitting if not carefully tuned. Furthermore, biased training data can lead to unfair predictions. Additionally, their reliance on multiple decision trees can obscure the true influence of individual features, and their performance is sensitive to data quality and hyperparameter tuning. This can lead to unexpected patterns with potentially positive or negative consequences. Similar to the logistic regression model, responsible AI practices were prioritized during development. The training data was rigorously scrutinized for biases, particularly regarding gender and class. Tools from InterpretML were utilized to understand the model's behavior and its potential impact on protected groups. By fostering responsible AI throughout the development process, we aimed to ensure fairer and more interpretable predictions from the random forest model.
