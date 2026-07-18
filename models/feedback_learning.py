import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

from models.similarity import similarity_matrix, users
from models.mbti import mbti_score


def interest_score(a, b):
    a = set(a.split(", "))
    b = set(b.split(", "))
    return len(a & b) / len(a | b) * 100


def location_score(a, b):
    return 100 if a == b else 0


feedback = pd.read_csv("data/feedback.csv")

features = []
labels = []

for _, row in feedback.iterrows():

    user1 = users[users["user_id"] == row["user_id"]].iloc[0]
    user2 = users[users["user_id"] == row["matched_user_id"]].iloc[0]

    idx1 = user1.name
    idx2 = user2.name

    semantic = similarity_matrix[idx1][idx2]

    mbti = mbti_score(user1["mbti"], user2["mbti"])

    interest = interest_score(
        user1["interests"],
        user2["interests"],
    )

    location = location_score(
        user1["location"],
        user2["location"],
    )

    features.append([
        semantic,
        mbti,
        interest,
        location,
    ])

    labels.append(row["action"])

X = pd.DataFrame(
    features,
    columns=[
        "Semantic",
        "MBTI",
        "Interest",
        "Location",
    ],
)

y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("\n" + "=" * 65)
print("PROFILE MATCHING FEEDBACK LEARNING")
print("=" * 65)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

print("\nPerformance Metrics")
print("-" * 65)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nConfusion Matrix")
print("-" * 65)
print(cm)

print("\nClassification Report")
print("-" * 65)
print(classification_report(y_test, predictions))

print("Learned Feature Weights")
print("-" * 65)

for feature, weight in zip(X.columns, model.coef_[0]):
    print(f"{feature:<10}: {weight:.4f}")

print("=" * 65)