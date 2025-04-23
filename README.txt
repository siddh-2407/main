
# Airline Ancillary Revenue Prediction: Capstone Project

## Objective

This project aims to develop a machine learning model to predict the likelihood of customers purchasing various ancillary services—such as travel insurance, extra baggage, meals, priority boarding, and premium lounge access during their flight bookings. The goal is to identify audience definitions that can be used for targeted marketing and personalized experience delivery via industry leading mar-tech platforms for optimizing the airline's ancillary revenue .

The model is trained using consolidated airline customer data derived from website and mobile app interactions, previous booking behavior, and reservation system records. This data is transformed and structured for analysis through a standard ETL (Extract, Transform, Load) pipeline prior to model training.

Note: For demonstration purposes, this dataset is taken from Kaggle which has a good resemblance with what a real airline's digital analytics and booking system data might contain. No real customer data has been used to preserve privacy and comply with data governance regulations. It is designed for the only purpose of this project study to showcase how a model could be applied using real-world data from any airlines digital analytics and booking management systems.


The model uses a variety of features such as customer demographics, booking behaviors, device interactions, and past purchases. Based on the model's output, airlines can personalize their offers and experiences, targeting the most likely buyers for services such as Travel Insurance, Extra Baggage, Meals, Priority Boarding, and Premium Lounges.

This approach is aimed at improving customer experience by offering relevant services while simultaneously maximizing revenue from ancillary products.  

This project aims to enhance ancillary revenue for an airline by predicting customer affinity toward additional services such as Travel Insurance, Preferred Seating, Extra Baggage, Meals, Priority Boarding, and Premium Lounge access. The solution uses a two-model pipeline:

- **Model 1:** Identifies customers likely to purchase any ancillary product.
- **Model 2:** For the 'Likely' customers from Model 1, predicts their probability of purchasing each individual ancillary product and segments them accordingly.

It has mainly three objectives as follows : 

- **Customer Segmentation:** Classify customers by their likelihood to purchase ancillary services.
- **Personalized Marketing:** Enable targeted marketing through industry-leading martech tools for personalization, retargeting, and campaign management.
- **Revenue Optimization:** Leverage predictive analytics to boost ancillary sales.

## Data Overview
The data used for this model consists of aggregated and pre-processed information sourced from the airline’s website and app interactions, as well as from previous booking history. Data preparation was done through an ETL process to consolidate user interactions, flight bookings, and previous ancillary purchases. 

The following features in the dataset considered from the input data:

- **Customer_ID**: Unique identifier for each customer.
- **Age**: Customer’s age. (Demographic information)
- **Gender**: Customer’s gender. (Demographic information)
- **Family_Size**: Size of the customer’s family. (Demographic information)
- **Num_Adults**: Number of adults in the family. (Demographic information)
- **Num_Children**: Number of children in the family. (Demographic information)
- **Has_Return_Flight**: Whether the booking is for a return flight. (Booking characteristics)
- **Is_Multi_City**: Whether the booking is for multiple cities. (Booking characteristics)
- **Origin**: Starting location of the flight.(Travel route related info)
- **Destination**: Destination of the flight. (Travel route related info)
- **Trip_Duration**: Duration of the trip. (Travel route related info)
- **Device_Type**: Type of device used for booking (e.g., Mobile, Desktop). (Channel of interaction )
- **Session_Duration**: Duration of the session on the airline's booking website/app. (Engagement metrics )
- **Page_Views**: Number of pages viewed during the booking session. (Engagement metrics)
- **Previous_Ancillary_Purchases**: Total number of ancillary products purchased in the past.
- **Recency_Days**: Number of days since the customer’s last booking.
- **Booking_Frequency**: Frequency of the customer’s bookings in the past.
- **Booking_Lead_Time: Days between booking and travel

Target Variables (Multi-label Binary Classification)

- **Travel_Insurance**: Whether the customer purchased travel insurance (binary).
- **Extra_Baggage**: Whether the customer purchased extra baggage (binary).
- **Meal**: Whether the customer purchased a meal (binary).
- **Priority_Boarding**: Whether the customer purchased priority boarding (binary).
- **Premium_Lounge**: Whether the customer purchased premium lounge access (binary).
- **wants_extra_seat**: Whether the customer extra seat (binary).
The dataset includes:

- **Demographics:** Age, Family Size, Number of Adults/Children.
- **Booking Details:** Booking Lead Time, Trip Duration, Length of Stay, Flight Duration.
- **Customer Behavior:** Session Duration, Page Views, Recency, Booking Frequency.
- **Ancillary Purchase History:** Previous purchases and flags for current booking.
- **Digital Touchpoints:** Device Type, Booking Completion, Travel Pattern (Return, Multi-City).

---

## Model Details
### ⚖️ Methodology

#### Model 1: Predict Any Ancillary Buyer
- **Target:** Binary `Ancillary_Buyer` flag.
- **Features:** All non-target customer features.
- **Model Selection:** Compared Random Forest, Logistic Regression, XGBoost.
- **Evaluation:** Used train/validation/test split, confusion matrix, ROC AUC, and accuracy.
- **Output Files:**
  - `model1_full_predictions.csv`: Prediction and probability for all customers.
  - `model1_likely_customers.csv`: Filtered list of 'Likely' customers for Model 2.

#### Model 2: Predict Product Affinity
- **Input:** Customers labeled as 'Likely' from Model 1.
- **Approach:** For each product:
  - Train a Random Forest classifier.
  - Predict probability of purchase.
  - Assign segments based on thresholds:
    - > 0.8: Enthusiast
    - 0.6–0.8: Upsellable
    - 0.3–0.6: Persuadable
    - < 0.3: Unlikely
- **Output File:** `model2_product_affinity_scores_and_segments.csv`

---- **Training Method**: The model was trained on a dataset of customer interactions and booking behavior to predict the likelihood of purchasing each of the ancillary services.
- **Evaluation Metrics**: Accuracy, confusion matrix, classification report, and feature importance report.
- **Preprocessing:
		-One-hot encoding for categorical features
		-Standardization for numerical features
		-Dimensionality reduction via PCA
- ** Data Split:
		-70% Training, 
		-15% Validation, 
		-15% Test
- Libraries: pandas, scikit-learn, 
## Approach
This project a binary classification approach to predict the likelihood of a customer purchasing each of the ancillary products. The model is trained using features related to customer demographics, device type, booking behaviors, and past purchase history related to ancilliary products and services mentioned above in detail. The goal is to assign a probability to each ancillary product, indicating how likely a customer is to purchase each one.

### Key Steps in the Approach:
1. **Data Preprocessing**:
   - Consolidated customer data from booking systems and user interaction data from websites/apps
   - Data was cleaned and pre-processed using standard ETL process to handle missing values, categorical variables for smooth scaling of numerical features and model training eventually.
   - Data includes demographics, booking history, and web analytics behavior
  

2. **Feature Engineering**:
 - Categorical variables like `Device_Type`, `Is_Multi_City`, `Origin`, and `Destination` were converted into numerical format using one-hot encoding.
 - Numerical scaling
 - Added fields like Booking_Lead_Time, Recency_Days, Booking_Frequency

3. **Dimensionality Reduction (PCA)**:
   - Principal Component Analysis (PCA) was applied to reduce the dimensionality of the dataset and ensure that the model focuses on the most important components.

4. **Model Building**:
   - A **Random Forest Classifier** was chosen for the multi-label classification task due to its ability to handle high-dimensional data and provide feature importance insights.
   - The model was trained for each ancillary product separately, and probabilities for purchasing each product were predicted.

5. **Model Evaluation**:
   - The model’s performance was evaluated using metrics such as accuracy, confusion matrix, and classification report.
   - Overfitting and underfitting were checked by evaluating model performance on both the training and validation datasets.

6. **Feature Importance**:
   - feature importance reports were used to explain the model’s predictions, providing transparency into which features most strongly influence each decision.

7. **Segmentation**:
   - Customers were segmented based on their predicted likelihood of purchasing each ancillary product. These segments include:
     - **Enthusiasts**: High likelihood of purchase.
     - **Upsell Persuadables**: Moderate likelihood.
     - **Neutral Buyers**: Neutral or undecided likelihood.
     - **Unlikely Buyers**: Low likelihood of purchase.


## Results
The model provided predicted probabilities for each customer and each ancillary product. These probabilities were used to create customer segments tailored to each product. The performance of the model was evaluated based on both accuracy and its ability to generalize well to unseen data.

Probability Range	Segment Name
> 0.8			[Product] Enthusiast
0.6 - 0.8		[Product] Upsell Persuadable
0.3 - 0.6		[Product] Persuadable Buyer
< 0.3			[Product] Unlikely Buyer

Where [Product] corresponds to Travel Insurance, Extra Baggage, etc.

## Inference
Based on the predicted probabilities and segments, airlines can target personalized offers to specific customer segments. For example:
- **Enthusiasts** could be offered tailored promotions or discounts.
- **Upsell Persuadables** could receive persuasive messaging to encourage purchasing additional services.
- **Neutral Buyers** might be offered standard packages or discounts to incentivize them.
- **Unlikely Buyers** could be excluded from specific promotional campaigns.
- ** Customers with higher session durations and page views are more likely to purchase ancillary services.
- ** Booking lead time, family size, and return/multi-city indicators influence ancillary purchase behavior.

## Explainability: 
Feature Importance method was used to interpret the impact of each feature on predictions for the Travel Insurance model.

## Output Usage: The predicted segments (e.g., "Meal Enthusiast", "Priority Boarding Upsell Persuadable") are exported as CSV
This output can be used by industry-leading martech platforms to:
 **Personalize Campaigns:** Tailor messages and offers.
- **Retarget Wisely:** Focus on Persuadable and Upsellable segments.
- **Optimize Resources:** Drive higher ROI through data-driven prioritization.

---
## References
1. Random Forest Classifier Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html2. 
4. Airline marketing literature



## Further Enhancements
- **Model Enhancements**: More advanced models like Gradient Boosting Machines (GBM) or XGBoost can be tested for improved performance.
- **Model Fine-tuning**: Hyperparameter optimization using techniques like Grid Search or Randomized Search could be applied to enhance model accuracy.
- **Additional Data**: Incorporating more customer interaction data, such as time of booking or seasonal trends, Incorporate route-level seasonal demand trends etc. could further improve model accuracy.
- ** Extend personalization to offer bundling strategies.


