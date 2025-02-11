import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset (Modify to load any dataset)
def load_dataset(file_path, text_column, label_column):
    data = pd.read_csv(file_path)
    data = data[[text_column, label_column]]
    data.columns = ['text', 'label']
    return data

# Example usage (change file_path, text_column, and label_column accordingly)
file_path = input("Enter the file path:")  # Update with actual dataset path
text_column = "message"  # Update with actual text column name
label_column = "category"  # Update with actual label column name
data = load_dataset(file_path, text_column, label_column)

# Convert labels to numerical format
encoder = LabelEncoder()
data['label'] = encoder.fit_transform(data['label'])

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Text Vectorization (Convert text to numerical format)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model Selection & Training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
