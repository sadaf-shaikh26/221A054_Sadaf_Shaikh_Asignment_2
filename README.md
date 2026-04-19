# 221A054_Sadaf_Shaikh_Asignment_2

(1) Problem Statement

The task is to perform sentiment analysis on tweets related to Donald Trump and classify them into Positive, Negative, or Neutral categories using machine learning models.

(2) Objective

To build and compare different classification models (Naïve Bayes, Logistic Regression, SVM) for predicting tweet sentiment and evaluate their performance using precision and recall.

(3) Dataset

Source: Manually created dataset (tweets inspired by X/Twitter)
Features:

Tweet (text data)
Sentiment (Positive / Negative / Neutral)

Size:

Total: 100 tweets
Training: 80 tweets
Testing: 20 tweets
(4) Methodology
Data Preprocessing
Loaded dataset using pandas
Converted text into numerical form using CountVectorizer
Split data into training (80%) and testing (20%)
EDA
Basic inspection of sentiment distribution
Observed balance between Positive, Negative, and Neutral classes
Model Building
Naïve Bayes
Logistic Regression
Support Vector Machine (SVM)
Evaluation
Used classification report
Compared precision, recall, and F1-score
(5) Results
Naïve Bayes
Accuracy: 65%
Good recall for Positive class (1.00)
Moderate performance overall
Logistic Regression
Accuracy: 65%
Best precision overall (macro avg: 0.77)
Balanced performance across classes
SVM
Accuracy: 45%
Poor recall for Negative and Neutral classes
Lowest overall performance
Insight
Logistic Regression performed best overall
Naïve Bayes performed well for Positive sentiment
SVM underperformed on this dataset
(6) How to Run
pip install -r requirements.txt
python analysis.py
(7) Conclusion

The sentiment analysis task was successfully implemented using three machine learning models. Logistic Regression showed the best overall performance, while SVM performed the worst. The results indicate that simpler models like Naïve Bayes and Logistic Regression are more effective for small text datasets.

(8) Student's Details

Name: Shaikh Sadaf
Roll No:64
UIN: 221A054
YEAR: TE-AIDS
