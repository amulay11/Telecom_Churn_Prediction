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

