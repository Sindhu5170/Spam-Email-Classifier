import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'message': [
        'Congratulations! You won a lottery',
        'Claim your free gift now',
        'Meeting at 5 PM today',
        'Project submission tomorrow',
        'Win cash prizes instantly',
        'Call me when you are free',
        'Exclusive offer just for you',
        'Let us discuss the assignment'
    ],
    'label': [
        'spam',
        'spam',
        'ham',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham'
    ]
}

df = pd.DataFrame(data)

# Convert text into numerical vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])

y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

# User input
email = input("Enter Email Text: ")

email_vector = vectorizer.transform([email])

result = model.predict(email_vector)

print("Prediction:", result[0])
