import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# ==============================
#  LOAD & PREPARE DATA
# ==============================
df1 = pd.read_csv("./data/label_tweets.csv")
df2 = pd.read_csv("./data/plabeled.csv")

# Cleanup
df1.drop_duplicates(inplace=True)
df2.drop_duplicates(inplace=True)
df1.drop("id", axis=1, inplace=True)

# Merge both datasets
df = pd.concat([df1, df2], ignore_index=True)

# Encode labels
df["label"] = df["label"].map({"Offensive": 1, "Non-offensive": 0})

X = df['full_text']
y = df['label']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ==============================
# TEXT VECTORIZATION
# ==============================
vectorizer = CountVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================
# DEFINE MODELS
# ==============================
models = {
    "MultinomialNB": MultinomialNB(),
    "LinearSVC": LinearSVC(),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = []

best_model = None
best_f1 = -1

# ==============================
# TRAIN + EVALUATE MODELS
# ==============================
for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train_vec, y_train)
    pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    results.append((name, acc, prec, rec, f1))
    print(f"{name} ➤ Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

    # choose best based on F1
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# Print comparison
print("\n MODEL COMPARISON:")
for r in results:
    print(f"{r[0]:15} | Acc={r[1]:.3f} | Prec={r[2]:.3f} | Rec={r[3]:.3f} | F1={r[4]:.3f}")

print(f"\n BEST MODEL SELECTED: {best_model_name} (F1 Score = {best_f1:.3f})")

# ==============================
# SAVE MODEL + VECTORIZER
# ==============================
joblib.dump(best_model, "cyberbullying_model.joblib")
joblib.dump(vectorizer, "cyber_vectorizer.joblib")

print("\n Model & Vectorizer saved successfully!")
