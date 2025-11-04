# Food Delivery Prediction Model

## Project Overview

This project develops a machine learning classification model to predict food delivery outcomes (acceptance/rejection) based on customer and delivery characteristics. The model analyzes 388 customer records with 55 features covering demographic information, delivery preferences, satisfaction factors, and past experience.

## Dataset Description

### Size and Structure
- **Total Records**: 388 customers
- **Features**: 55 columns (after preprocessing)
- **Target Variable**: Output (Binary: Yes/No)
- **Data Type**: Categorical and numerical mixed dataset

### Key Features

#### Demographic Information
- **Age**: Customer age (range: 18-33 years, mean: 24.6)
- **Gender**: Customer gender classification
- **Marital Status**: Marital status categories
- **Occupation**: Employment type
- **Educational Qualifications**: Education level
- **Family Size**: Number of family members (range: 1-6)

#### Geographic Data
- **Latitude/Longitude**: Customer location coordinates
- **Pin Code**: Postal code (Bangalore region: 560001-560109)

#### Delivery Preferences (Yes/No/Agree-Disagree)
- **Ease and Convenient**: Preference for convenience
- **More Restaurant Choices**: Preference for variety
- **Self Cooking**: Preference vs. delivery
- **Good Tracking System**: Requirement for order tracking
- **Good Food Quality**: Quality expectations
- **Less Delivery Time**: Speed expectations

#### Past Experience Factors (Yes/No/Agree-Disagree)
- **Late Delivery**: History of late arrivals
- **Poor Hygiene**: Past hygiene concerns
- **Bad Past Experience**: Previous negative experiences
- **Wrong Order Delivered**: Order accuracy issues
- **Missing Items**: Incomplete order history
- **Good Taste**: Quality satisfaction

#### Delivery Factors
- **Monthly Income**: Income brackets (5 categories)
  - No Income: 48.2%
  - 25001-50000: 17.8%
  - More than 50000: 16.0%
  - 10001-25000: 11.6%
  - Below Rs.10000: 6.4%

### Target Distribution
- **Class 0 (No)**: 22.4% (87 samples)
- **Class 1 (Yes)**: 77.6% (301 samples)

**Note**: Class imbalance present - model trained with this distribution in mind.

## Exploratory Data Analysis (EDA)

The EDA notebook provides comprehensive insights:

### Statistical Summary
- Numerical features show reasonable variance and distribution
- No significant missing values (< 1%)
- Reviews column has 387 non-null entries (1 missing)

### Distribution Analysis
- Income distribution heavily skewed toward "No Income" category
- Output distribution shows significant class imbalance (77.6% positive class)
- Categorical variables show diverse distributions across all categories

### Data Quality
- Encoded categorical features for machine learning compatibility
- Removed non-predictive columns (latitude, longitude, Reviews)
- Standardized numerical features for model compatibility

## Model Architecture

### Models Implemented

#### 1. Random Forest Classifier
- **Accuracy**: 94.87%
- **Precision (Class 1)**: 0.96
- **Recall (Class 1)**: 0.98
- **F1-Score (Class 1)**: 0.97
- **Parameters**: 200 estimators, random_state=22

#### 2. Gradient Boosting Classifier
- **Accuracy**: 93.59%
- **Precision (Class 1)**: 0.96
- **Recall (Class 1)**: 0.97
- **F1-Score (Class 1)**: 0.96
- **Parameters**: random_state=22

### Feature Importance
Both models identified critical features for predicting delivery outcomes:
- Top features include delivery time factors, past experience variables, and customer preferences
- Random Forest and Gradient Boosting showed similar importance rankings

### Model Evaluation
- **ROC-AUC Analysis**: Both models achieved excellent discrimination between classes
- **Class Balance Handling**: Models naturally handle class imbalance well with high recall
- **Cross-Validation**: Train-test split (80-20) maintained class distribution

## Data Preprocessing

### Steps Applied
1. **Encoding**: Label encoding for categorical features (50 columns)
2. **Scaling**: StandardScaler for numerical features
3. **Feature Selection**: Removed low-relevance columns
4. **Train-Test Split**: 80-20 split, random_state=42

### Handling Missing Data
- Minimal missing values addressed through removal
- No imputation needed (< 1% missing)

## Key Findings

1. **Model Performance**: Random Forest slightly outperforms Gradient Boosting with 94.87% accuracy
2. **High Recall**: Both models achieve >97% recall on positive class (delivery accepted)
3. **Feature Importance**: Past experience and delivery preferences are strong predictors
4. **Class Imbalance**: Successfully handled through algorithm design rather than resampling

## File Structure

```
├── notebooks
      ├──EDA.ipynb                    # Exploratory Data Analysis
      └──train_and_eval.ipynb         # Model Training and Evaluation
├── README.md                    # This file
└── data/
    └── food_delivery_data.csv   # Dataset (388 records, 55 features)
```

### Results
Both notebooks produce:
- Statistical summaries and distributions
- Feature importance visualizations
- ROC curves and confusion matrices
- Model performance metrics


## Performance Metrics

| Metric | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| Accuracy | 94.87% | 93.59% |
| Precision | 0.96 | 0.96 |
| Recall | 0.98 | 0.97 |
| F1-Score | 0.97 | 0.96 |
| ROC-AUC | High | High |


## Acknowledgments

- Dataset sourced from food delivery domain analysis
- Models built using scikit-learn and standard ML practices
- SHAP library used for feature importance visualization
