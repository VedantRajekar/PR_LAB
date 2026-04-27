import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("/content/emails.csv")

# Select first 100 entries
data = data.head(100)

# Correct column names based on available data
# The dataset appears to be already pre-processed with word counts.
# 'Email No.' is an identifier, and 'Prediction' is likely the target label.
X = data.drop(['Email No.', 'Prediction'], axis=1)  # Features
y = data['Prediction']  # Labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# The dataset is already vectorized, so CountVectorizer is not needed.
# X_train_vec and X_test_vec are simply X_train and X_test
X_train_vec = X_train
X_test_vec = X_test

# Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
predictions = model.predict(X_test_vec)
print(predictions)