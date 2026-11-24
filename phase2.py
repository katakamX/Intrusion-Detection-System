"""
Phase 2: Machine Learning-Based Intrusion Detection System
- Multi-class classification: identify the ATTACK CATEGORY (DoS, Probe, R2L, U2R,
  or Normal), not just a binary normal/attack flag.
- Compares three algorithms: Random Forest, XGBoost, and a small Neural Network (MLP).
- Persists the best-performing model + encoders to model/ with joblib for reuse
  in the Phase 3 dashboard (no retraining needed there).
"""

import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

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

# Standard NSL-KDD mapping of individual attack names to one of 4 attack categories.
# (Any name not listed here falls back to "normal" -- the 'normal' label itself
# maps to "normal" via this same lookup since it's included below.)
ATTACK_CATEGORY = {
    "normal": "normal",
    # DoS - Denial of Service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "mailbomb": "DoS", "worm": "DoS",
    # Probe - surveillance/scanning
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L - Remote to Local (unauthorized access from a remote machine)
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    # U2R - User to Root (privilege escalation)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def load_dataset(path):
    df = pd.read_csv(path, names=COLUMNS)
    df = df.drop(columns=["difficulty"])
    df["label"] = df["label"].apply(lambda x: ATTACK_CATEGORY.get(x, "normal"))
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
    all_classes = sorted(set(ATTACK_CATEGORY.values()))
    label_encoder.fit(all_classes)
    train_df["label"] = label_encoder.transform(train_df["label"])
    test_df["label"] = label_encoder.transform(test_df["label"])
    encoders["label"] = label_encoder
    return train_df, test_df, encoders


def evaluate(name, model, X_test, y_test, label_names):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | Macro-Precision: {prec:.4f} | Macro-Recall: {rec:.4f} | Macro-F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))
    return {"name": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "model": model}


def main():
    print("Loading NSL-KDD dataset with multi-class attack categories (normal, DoS, Probe, R2L, U2R)...")
    train_df = load_dataset("data/KDDTrain+.txt")
    test_df = load_dataset("data/KDDTest+.txt")
    print(f"  Train records: {len(train_df)}, Test records: {len(test_df)}")
    print("  Class distribution (train):")
    print(train_df["label"].value_counts().to_string())

    train_df, test_df, encoders = encode_features(train_df, test_df)
    label_names = list(encoders["label"].classes_)

    X_train, y_train = train_df.drop(columns=["label"]), train_df["label"]
    X_test, y_test = test_df.drop(columns=["label"]), test_df["label"]

    # MLP (neural net) is sensitive to feature scale; the tree models are not,
    # but scaling doesn't hurt them, so we use the same scaled features for all three
    # to keep the comparison simple and consistent.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    print("\nTraining Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    results.append(evaluate("Random Forest", rf, X_test, y_test, label_names))

    print("\nTraining XGBoost...")
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=100, max_depth=6, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    results.append(evaluate("XGBoost", xgb, X_test, y_test, label_names))

    print("\nTraining Neural Network (MLP)...")
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    results.append(evaluate("Neural Network (MLP)", mlp, X_test_scaled, y_test, label_names))

    print("\n=== Model Comparison Summary ===")
    comparison = pd.DataFrame(
        [{"Model": r["name"], "Accuracy": r["accuracy"], "Macro-Precision": r["precision"],
          "Macro-Recall": r["recall"], "Macro-F1": r["f1"]} for r in results]
    ).sort_values("Macro-F1", ascending=False)
    print(comparison.to_string(index=False))

    best = max(results, key=lambda r: r["f1"])
    print(f"\nBest model by Macro-F1: {best['name']}")

    uses_scaled = best["name"] == "Neural Network (MLP)"
    best_X_test = X_test_scaled if uses_scaled else X_test
    cm = confusion_matrix(y_test, best["model"].predict(best_X_test))
    print(f"\nConfusion matrix for {best['name']} (rows=actual, cols=predicted, order={label_names}):")
    print(cm)

    joblib.dump(best["model"], "model/best_model.joblib")
    joblib.dump(encoders, "model/encoders.joblib")
    joblib.dump(scaler, "model/scaler.joblib")
    joblib.dump(uses_scaled, "model/best_model_uses_scaled_input.joblib")
    print(f"\nSaved best model ({best['name']}), encoders, and scaler to model/ for reuse in the Phase 3 dashboard.")


if __name__ == "__main__":
    main()
