import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download("stopwords")

# Load data from CSV file
data = pd.read_csv("../Dataset/Preprocessed_Data.csv")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["Tweet"], data["stance"], test_size=0.2, random_state=42)

# Convert tweets to vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train SVM classifier
svm = SVC(kernel="rbf")
svm.fit(X_train_vectorized, y_train)

# Test SVM classifier
y_pred = svm.predict(X_test_vectorized)

# Print accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))
