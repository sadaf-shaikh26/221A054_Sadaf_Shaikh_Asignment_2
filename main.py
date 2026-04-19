import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score, recall_score

# Load dataset
data = pd.read_excel("data/trump_tweets.xlsx")

# Features & labels
X = data['Tweet']
y = data['Sentiment']

# Split (80 train / 20 test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# Model 1: Naive Bayes
# ---------------------------
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
pred_nb = nb.predict(X_test_vec)

print("\nNaive Bayes Results:")
print(classification_report(y_test, pred_nb))

# ---------------------------
# Model 2: Logistic Regression
# ---------------------------
lr = LogisticRegression(max_iter=200)
lr.fit(X_train_vec, y_train)
pred_lr = lr.predict(X_test_vec)

print("\nLogistic Regression Results:")
print(classification_report(y_test, pred_lr))

# ---------------------------
# Model 3: SVM
# ---------------------------
svm = SVC()
svm.fit(X_train_vec, y_train)
pred_svm = svm.predict(X_test_vec)

print("\nSVM Results:")
print(classification_report(y_test, pred_svm))


import matplotlib.pyplot as plt

# Model names
models = ['Naive Bayes', 'Logistic Regression', 'SVM']

# Accuracy values (from your output)
accuracy = [0.65, 0.65, 0.45]

# Precision (macro avg from your results)
precision = [0.70, 0.77, 0.78]

# Recall (macro avg)
recall = [0.67, 0.67, 0.48]

# Plot
x = range(len(models))

plt.figure(figsize=(8,5))

plt.plot(x, accuracy, marker='o', label='Accuracy')
plt.plot(x, precision, marker='o', label='Precision')
plt.plot(x, recall, marker='o', label='Recall')

plt.xticks(x, models)
plt.title("Model Performance Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.legend()
plt.grid()

plt.show()
