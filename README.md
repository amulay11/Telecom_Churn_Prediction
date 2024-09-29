# Telecom_Churn_Prediction
# Problem Statement
**Telecom Landscape and Strategy Evolution**

Telecom companies across the world are facing a need to constantly evolve due to the changing customer needs and competition in the market. Telecom companies not just need to keep their plans and services competitive but also need to customize their offerings based on the patterns and needs of customer segments. Adopting a correct strategy is key not only towards increasing the customer base but also to retain the current customers and preventing churn. Based on surveys, post pandemic the customer loyalty to Telecom providers has gone down by 22% - taking the average churn rate for telecom companies in the range of 22-35% with customer stickiness being impacted more by the customer experience than ever. Moreover customers are becoming increasingly price sensitive and a good majority perceive Telco offerings as expensive. As per studies, it costs companies 5x more to acquire a new customer as compared to retaining current customer. Moreover, it is critical to retain high value customers like corporate customers, small and medium enterprises, family plan customers and such other categories.
Therefore its is a key part of the Telco strategy to identify the underlying patterns that impact churn and then take necessary actions for customer retention - the policies applied for retention may be different as per the customer category - such as network service improvement, better internet speed, bundled plans - such as mobile + broadband + content bundling or family bundles etc. Thus churn prediction plays an important role in driving marketing strategy decisions for a Telecom company.
As part of this analysis a the data set used is from a major Telecom company in South East Asia and the objective is to analyse the data and prepare a model that can predit the churn probability with maximum accuracy.

**Customer behaviour during churn:**

Customers usually do not decide to switch to another competitor instantly, but rather over a period of time (this is especially applicable to high-value customers). In churn prediction, we assume that there are three phases of customer lifecycle :
The ‘good’ phase: In this phase, the customer is happy with the service and behaves as usual.
The ‘action’ phase: The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than the ‘good’ months. It is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)
The ‘churn’ phase: In this phase, the customer is said to have churned. In this case, since you are working over a four-month window, the first month is the ‘good’ phase, the second & third month is the ‘action’ phase.

# About the Data Set
The training dataset is provided in a csv format with 172 fields indicating various features and behavioral characteristics of a telecom customer. The end goal is to predict the churn_probability of the customer through an analysis and statistical processing of the features provided by means of machine learning techniques. Multiple machine learning modelling techniques are implemented in the project and they are evaluated on the test dataset to assess the performance based on accuracy and other metrics. Finally the aim is to narrow down to the best model that provides the most accurate results on the test data.

# Data Processing & EDA
For the initial data processing the training dataset has been loaded into dataframe followed by split into train and validation set and then handling of missing data. For missing data handling simple imputation with constant value has been used for few numeric fields and date fields have been imputed with a past date followed by derivation of new fields indicating "Days since".
As observed from the box plots below, for the customers with a churn probability of 1, there is a reduction in the total monhly outgoing usage from month 6 (Good phase ) thru month 8 (churn phase)

![image](https://github.com/user-attachments/assets/01473fd7-1c37-45d6-ab50-88668c597120)

From the scatter plot below it can be observed that there is a higher concentration of churned customers for lower values of "age on network" and higher values of "days since last recharge".
![image](https://github.com/user-attachments/assets/c9599eb0-5ced-4ccc-9dad-ee25c121f04e)

The below plots show the trend of the average outgoing monthly usage for churned customers (blue) and non churned customers (red). We see a stark decline in the usage for the churned customers from the good phase to the action phase and then further steep decline towards the churn phase.

![image](https://github.com/user-attachments/assets/ed1efdae-772c-4c5c-82e5-e563876f63d2)

**Data Capping, Class imbalance and Scaling**

Further as part of the data processing, the outliers have been handled using k-sigma-capping method that caps the upper and lower bound in a range of k time std deviation from the mean.
As observed there is a class imbalance in the train data as the minority class population is 10% of the majority class. SMOTE technique has been applied to oversample the minority class to increase it to 20% of the overall data.
Features have been scaled using the standard scaler in order to eliminate issues in statistical processing due to different scale of data. 

**Data Transformation**

Based on the box plots for the data features and the calculation of the skewness it is observed that some of features have a positive skew as indicated by the upper tail of the box plot that has more length. Yeo-Johnson transformation has been applied on the data features with positive skewness. 

**Feature Engineering**

As seen from the below plotting of feature importance using random forest, it is clear that the features for the action phase (month 7) have a significant contribution towards the churn probability.

![image](https://github.com/user-attachments/assets/8f571ea5-8243-4297-9b38-975bc4490e91)

Further, PCA has been performed on the data to derive principal components that contribute towards the target variable.

Below scatter plots give an indication of the contribution of PC1 and PC2 towards the churn probability

![image](https://github.com/user-attachments/assets/4d42ad16-0fab-42a4-a418-4c9f29d7db72)

![image](https://github.com/user-attachments/assets/6f507c1c-9ee3-4d1f-bb35-a3ec93a065ae)


# Model Building & Evaluation

**Logistic Regression - Model**

**LR Model 1**: The base model built as a pipeline with imputation, scaling, transformation, PCA and logistic regression provides an accuracy of **90.91 on the train data** and **90.97 on the test data**.

**LR Model 2**: With the use of train data oversampled through SMOTE the same model shows a reduction in accuracy with **train accuracy of 87.62%** and **test accuracy of 90.78%**.

**Random Forest - Model**

A model built with pipeline of imputation, scaling, transformation, PCA and RandomForestClassifier (with 150 estimators, min_samples_leaf = 3 and min_samples_split=6) provides following results:

**RF Model 1**: **With SMOTE on Train Data** --> **Training Accuracy: 97.19%** ... **Test Accuracy : 91.7%** ... Indicates overfitting on train data

**RF Model 2**: **Without SMOTE on Train Data** --> **Training Accuracy: 96.97%** ... **Test Accuracy : 91.8%** ... Indicates overfitting on train data

**RF Pipeline with GridSerchCV**

The parameters for RandomForestClassifier derived through GridSearchCV turn out to be - min_samples_leaf=2, min_samples_split=6, n_estimators=200.
With these parameters the accuracy scores are as below

**RF Model 3**: **With SMOTE on Train Data** --> **Training Accuracy: 98.04%** ... **Test Accuracy : 91.76%** ... Indicates overfitting on train data

**RF Model 4**: **Without SMOTE on Train Data** --> **Training Accuracy: 97.62%** ... **Test Accuracy : 91.91%** ... Indicates overfitting on train data

**XGBoost Model**

**XG Model 1**: Model built with XGBoost using default parameters gives a **test accuracy score of 93%** (with/without SMOTE)

**XG Model 2**: After performing a hyperparameter tuning with random search on XGBoost there is a slight improvement in **test accuracy to 93.45%**.

## Results on Unseen data

**LR Model 2** : Accuracy of **90.81%**

**RF Model 2** : Accuracy of **91.95%**

**RF Model 3** : Accuracy of **92.11%**

**RF Model 4** : Accuracy of **92.11%**

**RF Model 4 with capping of unseen data** : Accuracy of **92.10%**

**XG Model 2 with SMOTE on train data** : Accuracy of **93.33%**

**XG Model 2 without SMOTE with capping on unseen data** : Accuracy of **93.57%**

#### Best Accuracy obtained is 93.570% for the XGBoost tuned model without SMOTE as XGBoost has a hyperparameter for handling class imbalance

## Overall Assessment

The dataset provided primarily has features providing information about the subscriber's age on network, recharge behavior and usage patterns for calls and internet. After performing initial processing on the dataset, deriving some features with EDA and then using multiple ML algorithms for prediction of churn probability a fair accuracy of 93.57% is obtained using XGBoost on the unseen data. The final model may be used on more unseen data with similar feature characteristics to predict the churn probability and take appropriate strategic measures to take the right steps in the action phase so that the churn can be prevented and the subscriber can be retained. 

However overall there can be more features that can help to assess the churn probability better such as - age/gender/address region/profession/income demographics of the customer that can help in segmentation of the data and then use classification on different segments as the segments may exhibit varying characteristics. 

