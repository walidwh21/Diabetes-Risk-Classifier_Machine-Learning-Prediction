# <div align="center">Diabetes Dataset Documentation</div>

<div align="justify">

## Overview

This document provides comprehensive information about the diabetes dataset used in the Diabetes Risk SVM project. The dataset is designed for binary classification to predict whether a patient has diabetes based on various medical predictor variables.

## Dataset Summary

- **Dataset Name**: Pima Indians Diabetes Database
- **Total Records**: 768 instances
- **Features**: 8 independent variables + 1 target variable
- **Target Classes**: 2 (Binary classification)
  - Class 0: No diabetes (500 instances, 65.1%)
  - Class 1: Has diabetes (268 instances, 34.9%)
- **Data Type**: Numerical features only
- **Missing Values**: None (explicitly)
- **File Format**: CSV (Comma Separated Values)

## Dataset Origin and Context

The dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to diagnostically predict whether a patient has diabetes based on certain diagnostic measurements included in the dataset. All patients in this dataset are females at least 21 years old of Pima Indian heritage.

## Feature Descriptions

### 1. Pregnancies

- **Type**: Integer
- **Range**: 0-17
- **Mean**: 3.85
- **Description**: Number of times the patient has been pregnant
- **Clinical Significance**: Higher number of pregnancies can be associated with increased diabetes risk due to gestational diabetes history

### 2. Glucose

- **Type**: Integer
- **Range**: 0-199 mg/dL
- **Mean**: 120.89 mg/dL
- **Description**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **Clinical Significance**: This is one of the primary diagnostic criteria for diabetes
- **Normal Range**:
  - Normal: < 140 mg/dL
  - Prediabetes: 140-199 mg/dL
  - Diabetes: ≥ 200 mg/dL
- **Note**: Zero values (5 instances) may represent missing data and should be handled appropriately

### 3. BloodPressure

- **Type**: Integer
- **Range**: 0-122 mmHg
- **Mean**: 69.11 mmHg
- **Description**: Diastolic blood pressure (bottom number in blood pressure reading)
- **Clinical Significance**: High blood pressure is often comorbid with diabetes
- **Normal Range**:
  - Normal: < 80 mmHg
  - High: ≥ 80 mmHg
- **Note**: Zero values (35 instances) are physiologically impossible and represent missing data

### 4. SkinThickness

- **Type**: Integer
- **Range**: 0-99 mm
- **Mean**: 20.54 mm
- **Description**: Triceps skin fold thickness
- **Clinical Significance**: Measure of subcutaneous fat, indicator of obesity which correlates with diabetes risk
- **Note**: Zero values (227 instances) likely represent missing measurements rather than actual zero thickness

### 5. Insulin

- **Type**: Integer
- **Range**: 0-846 μU/mL
- **Mean**: 79.80 μU/mL
- **Description**: 2-hour serum insulin level
- **Clinical Significance**: Insulin resistance is a key factor in Type 2 diabetes development
- **Normal Range**: 16-166 μU/mL (fasting)
- **Note**: Zero values (374 instances) are likely missing data as insulin is always present in blood

### 6. BMI (Body Mass Index)

- **Type**: Float
- **Range**: 0.0-67.1 kg/m²
- **Mean**: 31.99 kg/m²
- **Description**: Weight in kg/(height in meters)²
- **Clinical Significance**: Strong predictor of diabetes risk
- **Categories**:
  - Underweight: < 18.5
  - Normal: 18.5-24.9
  - Overweight: 25.0-29.9
  - Obese: ≥ 30.0
- **Note**: Zero values (11 instances) are physiologically impossible

### 7. DiabetesPedigreeFunction

- **Type**: Float
- **Range**: 0.078-2.42
- **Mean**: 0.47
- **Description**: Diabetes pedigree function score
- **Clinical Significance**: Represents genetic predisposition to diabetes based on family history
- **Calculation**: Uses information about diabetes history in relatives and their genetic relationship to the subject
- **Higher Values**: Indicate stronger genetic predisposition

### 8. Age

- **Type**: Integer
- **Range**: 21-81 years
- **Mean**: 33.24 years
- **Description**: Age of the patient in years
- **Clinical Significance**: Diabetes risk increases with age
- **Distribution**: Relatively young population with 75% under 41 years

## Target Variable

### Outcome

- **Type**: Binary Integer (0 or 1)
- **Values**:
  - 0: No diabetes (500 instances, 65.1%)
  - 1: Has diabetes (268 instances, 34.9%)
- **Description**: Binary classification target indicating diabetes diagnosis
- **Class Balance**: Moderately imbalanced dataset favoring non-diabetic cases

## Data Quality Issues

### Missing Values

While the dataset contains no explicit null values, several features have zero values that are physiologically impossible:

1. **Glucose**: 5 zero values (0.65%)
2. **BloodPressure**: 35 zero values (4.56%)
3. **SkinThickness**: 227 zero values (29.56%)
4. **Insulin**: 374 zero values (48.70%)
5. **BMI**: 11 zero values (1.43%)

These zero values should be treated as missing data and handled appropriately through:

- Imputation (mean, median, or model-based)
- Removal of affected records
- Domain-specific replacement strategies

### Data Distribution

- The dataset shows characteristics typical of the Pima Indian population
- Age distribution is skewed toward younger individuals
- BMI values indicate a population with higher obesity rates
- Glucose levels show expected patterns for diabetes screening

## Statistical Summary

| Feature                  | Count | Mean   | Std    | Min   | 25%   | 50%  | 75%    | Max  |
| ------------------------ | ----- | ------ | ------ | ----- | ----- | ---- | ------ | ---- |
| Pregnancies              | 768   | 3.85   | 3.37   | 0     | 1     | 3    | 6      | 17   |
| Glucose                  | 768   | 120.89 | 31.97  | 0     | 99    | 117  | 140.25 | 199  |
| BloodPressure            | 768   | 69.11  | 19.36  | 0     | 62    | 72   | 80     | 122  |
| SkinThickness            | 768   | 20.54  | 15.95  | 0     | 0     | 23   | 32     | 99   |
| Insulin                  | 768   | 79.80  | 115.24 | 0     | 0     | 30.5 | 127.25 | 846  |
| BMI                      | 768   | 31.99  | 7.88   | 0     | 27.30 | 32.0 | 36.6   | 67.1 |
| DiabetesPedigreeFunction | 768   | 0.47   | 0.33   | 0.078 | 0.24  | 0.37 | 0.63   | 2.42 |
| Age                      | 768   | 33.24  | 11.76  | 21    | 24    | 29   | 41     | 81   |

## Use Cases and Applications

### Primary Use Case

- **Binary Classification**: Predicting diabetes diagnosis based on medical measurements
- **Medical Screening**: Identifying high-risk individuals for further testing
- **Risk Assessment**: Evaluating diabetes probability for preventive care

### Machine Learning Applications

- **Classification Algorithms**: SVM, Random Forest, Logistic Regression, Neural Networks
- **Feature Selection**: Identifying most predictive features
- **Model Validation**: Cross-validation and performance evaluation
- **Preprocessing Techniques**: Handling missing values, feature scaling, outlier detection

## Preprocessing Recommendations

### 1. Missing Value Handling

- Replace zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI
- Consider domain knowledge for appropriate imputation strategies
- Document all preprocessing steps for reproducibility

### 2. Feature Scaling

- Standardize or normalize features due to different scales
- Consider robust scaling for features with outliers

### 3. Outlier Detection

- Identify and handle extreme values in Insulin and BMI
- Use clinical knowledge to determine appropriate thresholds

### 4. Feature Engineering

- Create categorical age groups
- BMI categories based on clinical standards
- Interaction features between related variables

## Ethical Considerations

### Population Bias

- Dataset represents only Pima Indian women aged 21+
- Results may not generalize to other populations
- Consider demographic limitations when applying models

### Medical Implications

- Model predictions should supplement, not replace, medical diagnosis
- High-stakes application requiring careful validation
- Consider false positive/negative implications

### Privacy and Consent

- Ensure appropriate use of medical data
- Follow HIPAA and other relevant privacy regulations
- Maintain patient confidentiality

## References and Further Reading

1. Original Dataset Source: UCI Machine Learning Repository
2. Pima Indians Diabetes Database documentation
3. Clinical guidelines for diabetes diagnosis and management
4. Machine learning approaches for medical diagnosis

## File Information

- **Filename**: `data/diabetes.csv`
- **Encoding**: UTF-8
- **Separator**: Comma (,)
- **Header**: Yes (first row contains column names)
- **File Size**: Approximately 24 KB

---

</div>

_Last Updated: July 25, 2025_
_Dataset Version: 1.0_
_Documentation Version: 1.0_
