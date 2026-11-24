"""
Proof-of-concept: Machine Learning-Based Intrusion Detection System
Trains a Random Forest classifier on the NSL-KDD benchmark dataset to
distinguish normal network traffic from attack traffic.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]


def load_dataset(path):
    df = pd.read_csv(path, names=COLUMNS)
    df = df.drop(columns=["difficulty"])
    # Binary classification: normal vs any attack
    df["label"] = df["label"].apply(lambda x: "normal" if x == "normal" else "attack")
    return df


def encode_features(train_df, test_df):
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        encoder.fit(pd.concat([train_df[col], test_df[col]]))
        train_df[col] = encoder.transform(train_df[col])
        test_df[col] = encoder.transform(test_df[col])
        encoders[col] = encoder

    label_encoder = LabelEncoder()
    label_encoder.fit(["normal", "attack"])
    train_df["label"] = label_encoder.transform(train_df["label"])
    test_df["label"] = label_encoder.transform(test_df["label"])
    encoders["label"] = label_encoder
    return train_df, test_df, encoders


def main():
    print("Loading NSL-KDD dataset...")
    train_df = load_dataset("data/KDDTrain+.txt")
    test_df = load_dataset("data/KDDTest+.txt")
    print(f"  Train records: {len(train_df)}, Test records: {len(test_df)}")

    train_df, test_df, encoders = encode_features(train_df, test_df)

    X_train, y_train = train_df.drop(columns=["label"]), train_df["label"]
    X_test, y_test = test_df.drop(columns=["label"]), test_df["label"]

    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\nEvaluating on held-out test set...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}  (of flagged attacks, how many were real)")
    print(f"Recall:    {rec:.4f}  (of real attacks, how many were caught)")

    labels_order = encoders["label"].transform(["normal", "attack"])
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    print("\nConfusion Matrix [rows=actual, cols=predicted] (order: normal, attack):")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))

    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print("Top 10 most important features for detecting intrusions:")
    print(importances.sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
