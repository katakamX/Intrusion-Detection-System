"""
Phase 3: Streamlit dashboard for the ML-based Network Intrusion Detection System.

Loads the Phase 2 model (saved in model/) and lets a user classify network
traffic records either by sampling the bundled NSL-KDD test set or by
uploading their own CSV in the same raw format, with live alerts and charts.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from phase2 import COLUMNS, CATEGORICAL_COLUMNS, ATTACK_CATEGORY

FEATURE_COLUMNS = [c for c in COLUMNS if c not in ("label", "difficulty")]

st.set_page_config(page_title="ML-NIDS Dashboard", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model/best_model.joblib")
    encoders = joblib.load("model/encoders.joblib")
    scaler = joblib.load("model/scaler.joblib")
    uses_scaled = joblib.load("model/best_model_uses_scaled_input.joblib")
    return model, encoders, scaler, uses_scaled


@st.cache_data
def load_test_set():
    df = pd.read_csv("data/KDDTest+.txt", names=COLUMNS)
    df = df.drop(columns=["difficulty"])
    df["true_category"] = df["label"].apply(lambda x: ATTACK_CATEGORY.get(x, "normal"))
    return df


def preprocess(df, encoders):
    """Encode categorical feature columns using the saved encoders.
    Unseen categories (possible in a user-uploaded CSV) are mapped to the
    encoder's first known class rather than crashing the app."""
    out = df.copy()
    for col in CATEGORICAL_COLUMNS:
        encoder = encoders[col]
        known = set(encoder.classes_)
        fallback = encoder.classes_[0]
        safe_values = out[col].apply(lambda v: v if v in known else fallback)
        out[col] = encoder.transform(safe_values)
    return out[FEATURE_COLUMNS]


def predict(df, model, encoders, scaler, uses_scaled):
    X = preprocess(df, encoders)
    X_input = scaler.transform(X) if uses_scaled else X
    preds = model.predict(X_input)
    label_encoder = encoders["label"]
    return label_encoder.inverse_transform(preds)


def highlight_attacks(row):
    color = "" if row["Prediction"] == "normal" else "background-color: #ffcccc"
    return [color] * len(row)


def main():
    st.title("ML-Based Network Intrusion Detection System")
    st.caption("Phase 3 prototype — classifies network traffic as normal or one of 4 attack categories (DoS, Probe, R2L, U2R) using the Phase 2 model.")

    model, encoders, scaler, uses_scaled = load_artifacts()

    st.sidebar.header("Traffic Source")
    source = st.sidebar.radio("Choose input", ["Sample from NSL-KDD test set", "Upload CSV"])

    if source == "Sample from NSL-KDD test set":
        test_df = load_test_set()
        n = st.sidebar.slider("Number of records to sample", 10, 500, 50)
        if st.sidebar.button("Sample traffic") or "sampled_df" not in st.session_state:
            st.session_state["sampled_df"] = test_df.sample(n=n, random_state=np.random.randint(0, 10_000)).reset_index(drop=True)
        working_df = st.session_state["sampled_df"]
        has_ground_truth = True
    else:
        uploaded = st.sidebar.file_uploader(
            "Upload a CSV with no header, matching the NSL-KDD 41-feature format "
            "(optionally with a trailing label column)",
            type=["csv", "txt"],
        )
        if uploaded is None:
            st.info("Upload a CSV to classify, or switch to the bundled sample traffic in the sidebar.")
            return
        raw = pd.read_csv(uploaded, header=None)
        if raw.shape[1] == len(COLUMNS):
            raw.columns = COLUMNS
            raw = raw.drop(columns=["difficulty"])
            raw["true_category"] = raw["label"].apply(lambda x: ATTACK_CATEGORY.get(x, "normal"))
            has_ground_truth = True
        elif raw.shape[1] == len(FEATURE_COLUMNS):
            raw.columns = FEATURE_COLUMNS
            has_ground_truth = False
        else:
            st.error(f"Expected {len(FEATURE_COLUMNS)} feature columns (or {len(COLUMNS)} with label+difficulty), got {raw.shape[1]}.")
            return
        working_df = raw

    predictions = predict(working_df, model, encoders, scaler, uses_scaled)
    display_df = working_df[FEATURE_COLUMNS].copy()
    display_df.insert(0, "Prediction", predictions)
    if has_ground_truth:
        display_df.insert(1, "Actual", working_df["true_category"].values)

    alert_count = int((predictions != "normal").sum())
    total = len(predictions)

    col1, col2, col3 = st.columns(3)
    col1.metric("Records analyzed", total)
    col2.metric("Alerts raised", alert_count)
    if has_ground_truth:
        accuracy = float((predictions == working_df["true_category"].values).mean())
        col3.metric("Accuracy on this batch", f"{accuracy:.1%}")

    st.subheader("Prediction breakdown")
    counts = pd.Series(predictions).value_counts()
    st.bar_chart(counts)

    st.subheader("Classified traffic")
    st.dataframe(display_df.style.apply(highlight_attacks, axis=1), width="stretch")


if __name__ == "__main__":
    main()
