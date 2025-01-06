
# Card Fraud Detection using (Machine Learning Approach with Regression Models  and Resampling Techniques)

## Problem Statement

Fraudulent transactions pose a significant threat to financial institutions. The challenge lies in identifying these fraudulent activities amidst a large volume of legitimate transactions. This task becomes more complex due to the highly imbalanced nature of transaction data, where fraudulent transactions are a minority. The goal of this project is to build a machine learning model that can detect fraudulent transactions by addressing class imbalance using resampling techniques.

### Challenges:
- **Imbalanced Data**: The dataset is highly skewed, with legitimate transactions far outnumbering fraudulent ones.
- **Model Bias**: Standard models often perform poorly in identifying fraudulent transactions because they tend to predict the majority class.
- **Accuracy vs. Detection**: High accuracy in models may be misleading if they cannot detect fraudulent transactions effectively.

### Objective:
The objective of this project is to build and evaluate machine learning models that detect fraudulent transactions accurately, using resampling techniques like undersampling and oversampling to address the imbalance between the classes. The goal is to improve precision, recall, and the F1 score of the fraud detection models.

### Importance:
Accurately detecting fraudulent transactions is crucial for financial institutions to minimize losses, improve security, and enhance customer trust. A reliable fraud detection system enables proactive measures to mitigate fraudulent activities.

## Approach

This project utilizes resampling techniques to balance the dataset and improve fraud detection accuracy. Three key resampling strategies were applied:
1. **Imbalanced Data**: No resampling; the model is trained on the original imbalanced dataset.
2. **Undersampled Data**: The majority class is undersampled to balance the dataset.
3. **Oversampled Data**: The minority class is oversampled to ensure a balanced dataset.

Several machine learning models were trained and evaluated, including:
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier

Evaluation metrics used to assess model performance include:
- **Accuracy**: Overall correct predictions.
- **Precision**: Accuracy of positive predictions (fraudulent transactions).
- **Recall**: Proportion of fraudulent transactions detected.
- **F1 Score**: Balanced measure of precision and recall.

## Dataset Information

The dataset used in this project is from the **Kaggle Credit Card Fraud Detection** challenge. It contains **284,807 rows** and **31 features** representing anonymized credit card transactions. The dataset is highly imbalanced, with only a small percentage of fraudulent transactions.

- **Features**: The dataset consists of anonymized features representing different characteristics of the transaction, such as transaction amount, timestamp, and user ID.
- **Target Variable**: The target variable indicates whether a transaction is fraudulent or not:
  - `0` represents a legitimate transaction.
  - `1` represents a fraudulent transaction.
- **Imbalance**: The dataset has a large imbalance, with **Class 0** (legitimate transactions) dominating the data, making it challenging for machine learning models to detect the minority **Class 1** (fraudulent transactions).

Dataset Reference:
- [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Project Summary

### Objective:
The goal is to develop a machine learning model that accurately identifies fraudulent transactions by addressing class imbalance.

### Scope:
Fraud detection is critical in the financial sector. This project leverages various resampling techniques to train models on more balanced datasets and improve the detection of fraudulent activities.

### Key Components:
- **Class Imbalance Handling**: Applying undersampling and oversampling techniques to balance the dataset and reduce model bias.
- **Model Training**: Using Logistic Regression, Random Forest, and Decision Tree models to detect fraud.
- **Evaluation Metrics**: Using precision, recall, accuracy, and F1 score to evaluate model performance.

### Implementation:
Machine learning algorithms were implemented in Python, using libraries such as `scikit-learn` for model training and evaluation. The performance of models was compared using different resampling strategies to understand their impact on fraud detection.

### Outcome:
The goal was to build a model capable of accurately detecting fraudulent transactions and ensuring that performance metrics like precision, recall, and F1 score were balanced.

## Results

# Model Performance Evaluation on Fraud Detection

The table below shows the evaluation metrics for different models across various resampling strategies.

| ****Model****                | ****Resampling**** | ****Accuracy Score**** | ****F1 Score**** | ****Precision Score**** | ****Recall Score**** |
|----------------------------|-------------------|------------------------|------------------|-------------------------|----------------------|
| Logistic Regression         | Imbalanced        | 0.999256               | 0.735484         | 0.890625                | 0.626374             |
| Random Forest Classifier    | Imbalanced        | 0.999438               | 0.812121         | 0.905405                | 0.736264             |
| Decision Tree Classifier    | Imbalanced        | 0.998948               | 0.701031         | 0.660194                | 0.747253             |
| Logistic Regression         | Undersampling     | 0.947368               | 0.948980         | 0.989362                | 0.911765             |
| Random Forest Classifier    | Undersampling     | 0.947368               | 0.948980         | 0.989362                | 0.911765             |
| Decision Tree Classifier    | Undersampling     | 0.931579               | 0.935323         | 0.949495                | 0.921569             |
| Logistic Regression         | Oversampling      | 0.944329               | 0.942553         | 0.972963                | 0.913987             |
| Random Forest Classifier    | Oversampling      | 0.999927               | 0.999927         | 0.999855                | 1.000000             |
| Decision Tree Classifier    | Oversampling      | 0.998210               | 0.998211         | 0.997459                | 0.998964             |


### Key Observations:
- **Imbalanced Classes**: The model performed with a very high accuracy but struggled with detecting fraudulent transactions, as indicated by low recall and F1 score.
- **Undersampling**: This approach improved recall and F1 score, leading to more effective fraud detection while maintaining a reasonable accuracy.
- **Oversampling**: This approach provided perfect recall, but accuracy was inflated due to the resampling of the minority class. Precision was high across all models, and F1 score showed optimal performance in detecting fraud.

## Conclusion

In this project, we addressed the class imbalance in fraud detection using resampling techniques, which significantly impacted the performance of the models. Key findings include:
- **Imbalanced Data**: While accuracy was high, the model struggled to detect fraudulent transactions, leading to poor recall and F1 scores.
- **Undersampled Data**: Provided a more realistic view of model performance with improved recall and a balanced F1 score, making it suitable for fraud detection.
- **Oversampled Data**: Led to high precision and recall with a balanced F1 score, though the high accuracy may not reflect real-world class distributions.

The results indicate that both undersampling and oversampling improve fraud detection compared to using imbalanced data, with undersampling providing a more balanced trade-off between recall and precision.

## Key Takeaways:
- **Imbalanced datasets** often lead to misleadingly high accuracy, but poor fraud detection.
- **Undersampling** and **oversampling** improve recall and F1 score, making the models more effective at detecting fraudulent transactions.
- **Oversampling** leads to a perfect recall but should be considered with caution as it may not reflect real-world class distributions.

## Next Steps
- **Hyperparameter Tuning**: Further fine-tuning of models and hyperparameters to improve performance.
- **Explore SMOTE**: Investigate the use of Synthetic Minority Over-sampling Technique (SMOTE) to generate synthetic fraud cases.
- **Real-world Application**: Deploy the models in production, where the class distribution may change over time, and retrain periodically to maintain performance.

## Installation

## Project Repository

You can access the full code and resources for this project on my [GitHub repository](https://github.com/Mazenasag/Detecting-Card-Fraud).

Feel free to clone the repository, explore the code, and contribute to the project.

