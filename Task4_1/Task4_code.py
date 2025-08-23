# Cell 1: Import libraries
import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# Cell 2: Unzip your dataset
zip_path = r"D:\OIBSIP\Task4\Dataset_SpamDetector.zip"
extract_dir = r"D:\OIBSIP\Task4"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extracted files:", os.listdir(extract_dir))
# Cell 3: Load dataset
csv_path = os.path.join(extract_dir, "spam.csv")  # update if different
df = pd.read_csv(csv_path, encoding="latin-1")   # sometimes spam datasets need latin-1

print(df.head())
print(df.columns)
# Cell 4: Clean dataset properly
# Keep only the two useful columns
df = df[['v1', 'v2']].rename(columns={"v1": "label", "v2": "text"})

# Normalize labels to lower case
df['label'] = df['label'].str.lower()

print(df.head())
print(df['label'].value_counts())
# Cell 5: Drop nulls & duplicates
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print("Dataset shape after cleaning:", df.shape)
# Cell 6: Encode labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['label_num'] = le.fit_transform(df['label'])  # ham=0, spam=1

print(df[['label', 'label_num']].head())
# Cell 7: Train-test split
from sklearn.model_selection import train_test_split

X = df['text']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
# Cell 8: TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vectorized training shape:", X_train_vec.shape)
# Cell 9: Train model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_vec, y_train)
# Cell 10: Model evaluation
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Cell 11: Test with custom messages
test_msgs = [
    "Congratulations! You won a free ticket to Bahamas. Claim now!",
    "Hey, are we still meeting for lunch today?"
]

test_vec = vectorizer.transform(test_msgs)
preds = model.predict(test_vec)

for msg, label in zip(test_msgs, preds):
    print(f"Message: {msg}\nPrediction: {'Spam' if label == 1 else 'Ham'}\n")
# Transform test set into TF-IDF features
X_test_vec = vectorizer.transform(X_test)

# Predict on test set
y_pred = model.predict(X_test_vec)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Custom test messages
test_msgs = [
    "Congratulations! You won a free ticket to Bahamas. Claim now!",
    "Hey, are we still meeting for lunch today?",
    "Good morning, my Love ... I go to sleep now and wish you a great day."
]

# Transform using fitted vectorizer
test_vec = vectorizer.transform(test_msgs)

# Predict
preds = model.predict(test_vec)

# Show results
for msg, label in zip(test_msgs, preds):
    print(f"Message: {msg}\nPrediction: {'Spam' if label == 1 else 'Ham'}\n")
import joblib
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")
msg = input("Enter a message: ")
vec = vectorizer.transform([msg])
pred = model.predict(vec)[0]
print("Prediction:", "Spam" if pred == 1 else "Ham")
