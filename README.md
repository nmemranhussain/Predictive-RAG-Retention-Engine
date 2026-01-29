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

## Business Problem & Solution
**Business Problem:** Reducing Chunging rate, Improving ROI
**Business Solution:** Understading Dataset, Predictive Modeling, Optimization (Customer Segmentation & EV), Dynamic Policy Optimization & RAG Chatbot, ROI Improvement, Future opportunity.

## Understanding Dataset
**Dataset Name & Source:** [Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)  
**Number of Samples:** The original dataset contains 541,909 rows. After the cleaning process—which includes removing duplicates and missing values—the notebook works with a processed set of 401,604 transactions.  
**Original Features Used:** InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country.
**Engineered Features:** Year, Month, Day, Hour, DayOfWeek, IsWeekend, TotalAmount, Is_Return / Is_Cancelled, BasketUniqueItems, CustProductDiversity, CustTotalSales, Recency, Frequency, Monetary,
**Target Feature:** Churn_Label
**Unique Entries:** 4289 unique customers.

### Data Dictionary

|Column Name	      |Modeling Role	|Measurement Level	|Description|
|------------------|--------------|------------------|-----------|
|InvoiceNo	|Metadata	|Nominal	|A unique 6-digit identifier for each transaction; used to calculate purchase frequency.|
|StockCode	|Metadata	|Nominal	|A unique product identifier used to track specific items and product diversity.|
|Description	|Metadata	|Nominal	|A text description of the product; primarily used for data inspection and RAG context.|
|Quantity	|Input	|Ratio	|The number of items purchased in a transaction; used to calculate total sales and monetary value.|
|InvoiceDate	|Metadata	|Interval	|The timestamp of the transaction, used to derive all time-based engineered features.|
|UnitPrice	|Input	|Ratio	|The price per unit of the product; crucial for calculating total transaction value.|
|CustomerID	|ID	|Nominal	|A unique identifier for each customer; used as the primary key for aggregating behavioral data.|
|Country	|Input	|Nominal	|The geographic location of the customer; used to analyze market distribution and regional churn patterns.|
|Year	|Metadata/Input	|Ordinal	|The year extracted from the InvoiceDate to identify long-term trends.|
|Month	|Input	|Ordinal	|The month of the transaction; used to capture seasonal shopping behaviors.|
|Day	|Input	|Ordinal	|The day of the month when the purchase occurred.|
|Hour	|Input	|Ordinal	|The hour of the day, used to identify peak shopping times (e.g., morning vs. evening).|
|DayOfWeek	|Input	|Ordinal	|The specific day (Monday-Sunday) used to analyze weekly purchase cycles.|
|IsWeekend	|Input	|Binary	|A flag (1/0) indicating if the purchase happened on a Saturday or Sunday.|
|TotalAmount	|Input	|Ratio	|The calculated total value of a transaction (Quantity × UnitPrice).|
|Is_Return / Is_Cancelled	|Input	|Binary	|Flags identifying if a transaction was a return or a cancellation; used to compute return rates.|
|BasketUniqueItems	|Input	|Ratio	|The number of distinct products (unique StockCodes) contained within a single invoice.|
|CustProductDiversity	|Input	|Ratio	|The cumulative number of different products a customer has bought over time.|
|CustTotalSales	|Input	|Ratio	|The total lifetime monetary value of all successful purchases made by the customer.|
|Recency	|Input	|Ratio	|The number of days since the customer’s last purchase; a core feature for churn modeling.|
|Frequency	|Input	|Ratio	|The total count of unique purchase transactions (orders) made by the customer.|
|Monetary	|Input	|Ratio	|The total expenditure of the customer; used alongside Recency and Frequency to profile value.|
|Churn_Label	|Target	|Binary	|The predicted variable; 1 indicates the customer has churned (inactive for >30 days), and 0 indicates they are active.|

### Training & Test Data

**Training Data Percentage:** 70% of the customer-level dataset (the RFM data) was used as training data.  
**Testing Data Percentage:** The remaining 30% was reserved as a holdout test set to evaluate model performance.

## Predictive Modeling Details

### Model Type
- **Churn Classifier:** Logistic Regression and Random Forest.
- **Recommendation Engine:** Contextual Bandit (LinUCB) for personalized retention actions.
- **Conversational Layer:** Retrieval-Augmented Generation (RAG) using Gemini-2.5-flash and Gemini-embedding-001.
  
### Model Training Methodology

This training data was used to fit a Logistic Regression model (with feature scaling) and a Random Forest classifier. Both models utilized balanced class weights to further account for the imbalance in the training labels.

### Evaluation Metrics  
- **Churn Prediction:** AUC, Precision, Recall, and F1-score (calculated using stratified test sets).
- **Retention Policy:** Projected ROI and Average Reward per action.
- **RAG System:** Groundedness and factual accuracy based strictly on provided context.

### Final Values of Metrics for All Data using **Logistic Regression** and **Random Forest** model:

| Model       | AUC   | Precision | Recall |F1-score|
|-------------|-------|-----------|--------|--------|
| Logistic Regression | 0.723  | 0.529 | 0.763 | 0.625 |
| Random Forest| 0.704 | 0.559 | 0.442 | 0.494 |

### Retention Policy (ROI Analysis):

- **Highest ROI Action:** 'call+coupon' (Average Reward: $3.74).
- **Lowest ROI Action:** 'email' (Average Reward: $0.00).

### Model & Rag Architecture & Programming
- **Feature Engineering:** Conversion of raw transactional data into aggregated customer profiles including diversity of products purchased and weekend shopping flags.
- **Vector Database:** ChromaDB for persistent storage and retrieval of semantic embeddings.
- **Text Splitting Technology:** LangChain
- **RAG Workflow:** Embedding Model: models/gemini-embedding-001.
- **Generative Model:** models/gemini-2.5-flash.

- Knowledge Base Construction:
- **Source 1:** Structured customer profiles (RFM + Predicted Policy) converted to natural language strings.
- **Source 2:** Technical documentation and code snippets extracted directly from the processing notebook (E-commerce_1_1.ipynb).
- **Vector Indexing:** Recursive character splitting into 1,000-character chunks with 200-character overlap for context preservation.
- **Retrieval Mechanism:** Persistent ChromaDB store using cosine similarity of embeddings.


### Version of the Modeling Software: 
- **'pandas'**: '2.2.2',
- **'scikit-learn'**: '1.4.2',
- **'seaborn'**: '0.13.2',
- **'matplotlib'**: '3.8.4**

### Hyperparameters or Other Settings of the Model


The following hyperparameters were used for the 'random forest' as an alternative model:


## Quantitative Analysis

### Plots Related to Data or Final Model
 
![Plot of Survival Rate Vs. Passenger Class](SR_by_Class.png) 

**Description**: Passengers in 1st class had the highest survival rate, followed by those in 2nd class. 3rd class passengers had the lowest survival rate.


## Potential Impacts, Risks, and Uncertainties using Logistic Regression & Random Forest Model ##
