import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================
# LOAD DATASET
# ==========================

print("Loading Dataset...")

df = pd.read_csv("dataset/email_dataset_classification.csv")

print("Dataset Loaded Successfully\n")

# ==========================
# HANDLE MISSING VALUES
# ==========================

print("Missing Values Before Cleaning:\n")
print(df.isnull().sum())

# Remove rows where label is missing
df = df.dropna(subset=["label"])

# Replace missing messages with empty string
df["clean_message"] = df["clean_message"].fillna("")

# Convert to string
df["clean_message"] = df["clean_message"].astype(str)

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

# ==========================
# FEATURES & LABELS
# ==========================

X = df["clean_message"]
y = df["label"]

# ==========================
# TF-IDF
# ==========================

print("\nCreating TF-IDF Features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X = vectorizer.fit_transform(X)

print("TF-IDF Created Successfully")

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# MODELS
# ==========================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Linear SVM":
        LinearSVC(),

    "Naive Bayes":
        MultinomialNB(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )
}

best_model = None
best_accuracy = 0
best_name = ""

# ==========================
# TRAINING
# ==========================

for name, model in models.items():

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, prediction))

    print("\nClassification Report")
    print(classification_report(y_test, prediction))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

# ==========================
# SAVE MODEL
# ==========================

joblib.dump(best_model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\n")
print("=" * 70)
print("BEST MODEL :", best_name)
print("BEST ACCURACY :", round(best_accuracy * 100, 2), "%")
print("=" * 70)

print("\nModel saved to model/spam_model.pkl")
print("Vectorizer saved to model/tfidf_vectorizer.pkl")