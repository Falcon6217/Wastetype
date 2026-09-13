import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Waste Type Classifier",
    page_icon="♻️",
    layout="wide",
)

# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 15%, rgba(112, 0, 255, 0.15), transparent 40%),
            radial-gradient(circle at 85% 85%, rgba(0, 230, 255, 0.15), transparent 40%),
            linear-gradient(135deg, #0f0c20 0%, #15102a 50%, #060b19 100%);
        color: #e2e8f0;
    }

    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 32, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.10);
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(0, 242, 254, 0.30);
        border-radius: 16px;
        padding: 16px;
    }

    .neon-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(
            90deg,
            #00f2fe 0%,
            #4facfe 35%,
            #00ff88 70%,
            #ff007f 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .neon-subtitle {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    .stButton > button {
        background: linear-gradient(
            90deg,
            #ff007f 0%,
            #7928ca 50%,
            #4facfe 100%
        ) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
    }

    .prediction-card {
        background: linear-gradient(
            135deg,
            rgba(0, 255, 136, 0.10),
            rgba(0, 242, 254, 0.10)
        );
        border: 1px solid #00ff88;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }

    .prediction-title {
        font-size: 2rem;
        font-weight: 800;
        color: #00ff88;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. CONSTANTS
# ============================================================

FEATURES = [
    "Weight",
    "Moisture",
    "Hardness",
    "Magnetic",
    "Biodegradable",
]

TARGET = "WasteType"

FEATURE_INFO = {
    "Weight": (1.0, 500.0, "Weight of the waste item in grams."),
    "Moisture": (0.0, 100.0, "Moisture content as a percentage."),
    "Hardness": (1.0, 10.0, "Hardness score from 1 to 10."),
    "Magnetic": (0, 1, "1 = magnetic, 0 = non-magnetic."),
    "Biodegradable": (0, 1, "1 = biodegradable, 0 = non-biodegradable."),
}

EXPECTED_CATEGORIES = [
    "Plastic",
    "Metal",
    "Paper",
    "Glass",
    "Organic",
]

# ============================================================
# 4. GENERATE DEMO DATA
# ============================================================

@st.cache_data
def generate_demo_data(num_samples=250, seed=42):
    rng = np.random.RandomState(int(seed))

    profiles = {
        "Plastic": {
            "weight": (5, 60),
            "moisture": (0, 10),
            "hardness": (2, 5),
            "magnetic": 0,
            "bio": 0,
        },
        "Metal": {
            "weight": (50, 400),
            "moisture": (0, 5),
            "hardness": (7, 10),
            "magnetic": 1,
            "bio": 0,
        },
        "Paper": {
            "weight": (2, 40),
            "moisture": (10, 40),
            "hardness": (1, 3),
            "magnetic": 0,
            "bio": 1,
        },
        "Glass": {
            "weight": (80, 500),
            "moisture": (0, 5),
            "hardness": (6, 9),
            "magnetic": 0,
            "bio": 0,
        },
        "Organic": {
            "weight": (20, 300),
            "moisture": (50, 95),
            "hardness": (1, 3),
            "magnetic": 0,
            "bio": 1,
        },
    }

    rows = []

    # Make sure every class gets data.
    per_class = max(10, int(num_samples) // len(profiles))

    for label, profile in profiles.items():
        for _ in range(per_class):
            weight = rng.uniform(*profile["weight"])
            moisture = rng.uniform(*profile["moisture"])
            hardness = rng.uniform(*profile["hardness"])

            magnetic = profile["magnetic"]
            if rng.rand() < 0.05:
                magnetic = 1 - magnetic

            biodegradable = profile["bio"]
            if rng.rand() < 0.05:
                biodegradable = 1 - biodegradable

            rows.append(
                [
                    weight,
                    moisture,
                    hardness,
                    magnetic,
                    biodegradable,
                    label,
                ]
            )

    df = pd.DataFrame(rows, columns=FEATURES + [TARGET])

    return (
        df.sample(frac=1, random_state=int(seed))
        .reset_index(drop=True)
    )


# ============================================================
# 5. CLEAN AND VALIDATE DATA
# ============================================================

def clean_dataset(df):
    df = df.copy()

    # Remove accidental spaces from column names.
    df.columns = df.columns.astype(str).str.strip()

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        return None, (
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Keep only the columns used by this application.
    df = df[required_columns].copy()

    # Convert numerical columns safely.
    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Clean target.
    df[TARGET] = df[TARGET].astype(str).str.strip()

    # Remove missing/invalid rows.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_columns)

    if df.empty:
        return None, "The dataset has no valid rows after cleaning."

    # Convert binary columns to integers.
    for column in ["Magnetic", "Biodegradable"]:
        df[column] = df[column].round().astype(int)

    # Check binary columns.
    for column in ["Magnetic", "Biodegradable"]:
        invalid = ~df[column].isin([0, 1])
        if invalid.any():
            return None, (
                f"Column '{column}' must contain only 0 or 1."
            )

    # Remove empty target values.
    df = df[df[TARGET].str.len() > 0]

    if df.empty:
        return None, "WasteType contains no valid class labels."

    return df.reset_index(drop=True), None


# ============================================================
# 6. TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models(df, test_size, random_state):
    X = df[FEATURES].copy()
    y_raw = df[TARGET].copy()

    # At least two classes are required for classification.
    unique_classes = y_raw.unique()

    if len(unique_classes) < 2:
        raise ValueError(
            "The dataset must contain at least two different WasteType classes."
        )

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # Check whether every class has enough samples for stratification.
    class_counts = pd.Series(y).value_counts()

    stratify_value = y

    # If a class has only one sample, stratified splitting is impossible.
    if class_counts.min() < 2:
        stratify_value = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=stratify_value,
    )

    # Decision Tree
    decision_tree = DecisionTreeClassifier(
        random_state=int(random_state),
        max_depth=6,
        min_samples_split=2,
    )

    decision_tree.fit(X_train, y_train)
    dt_pred = decision_tree.predict(X_test)

    # Logistic Regression with scaling.
    logistic_regression = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=int(random_state),
                ),
            ),
        ]
    )

    logistic_regression.fit(X_train, y_train)
    lr_pred = logistic_regression.predict(X_test)

    return (
        encoder,
        X_train,
        X_test,
        y_train,
        y_test,
        decision_tree,
        dt_pred,
        logistic_regression,
        lr_pred,
    )


# ============================================================
# 7. SIDEBAR - DATA SOURCE
# ============================================================

st.sidebar.markdown("## ⚙️ Control Panel")

data_source = st.sidebar.radio(
    "Dataset Source",
    [
        "Use built-in demo data",
        "Upload waste_data.csv",
    ],
)

data = None

if data_source == "Upload waste_data.csv":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        help=(
            "CSV must contain: "
            + ", ".join(FEATURES + [TARGET])
        ),
    )

    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
            data, error_message = clean_dataset(uploaded_data)

            if error_message:
                st.sidebar.error(error_message)
                st.stop()

            st.sidebar.success(
                f"Loaded {len(data)} valid rows."
            )

        except Exception as exc:
            st.sidebar.error(
                f"Could not read the CSV file: {exc}"
            )
            st.stop()

    else:
        st.sidebar.info(
            "Upload a CSV to train the classifier."
        )
        st.sidebar.caption(
            "Showing demo data until a CSV is uploaded."
        )
        data = generate_demo_data()

else:
    num_samples = st.sidebar.slider(
        "Number of demo samples",
        min_value=50,
        max_value=1000,
        value=250,
        step=50,
    )

    seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        value=42,
        step=1,
    )

    data = generate_demo_data(
        num_samples=num_samples,
        seed=seed,
    )

# ============================================================
# 8. TRAIN/TEST SETTINGS
# ============================================================

test_size = st.sidebar.slider(
    "Test set size",
    min_value=0.10,
    max_value=0.50,
    value=0.20,
    step=0.05,
)

random_state = st.sidebar.number_input(
    "Train/test split seed",
    min_value=0,
    value=42,
    step=1,
)

# ============================================================
# 9. TRAIN MODELS
# ============================================================

try:
    (
        encoder,
        X_train,
        X_test,
        y_train,
        y_test,
        dt,
        dt_pred,
        lr,
        lr_pred,
    ) = train_models(
        data,
        test_size,
        random_state,
    )
except ValueError as exc:
    st.error(f"Model training error: {exc}")
    st.stop()

class_names = list(encoder.classes_)

# ============================================================
# 10. SIDEBAR FOOTER
# ============================================================

st.sidebar.markdown("---")
st.sidebar.caption(
    "Powered by Streamlit and Scikit-learn."
)

# ============================================================
# 11. HEADER
# ============================================================

st.markdown(
    '<div class="neon-title">♻️ Waste Type Classifier</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="neon-subtitle">'
    "Machine learning classification using Decision Tree "
    "and Logistic Regression"
    "</div>",
    unsafe_allow_html=True,
)

# ============================================================
# 12. TABS
# ============================================================

(
    tab_predict,
    tab_explore,
    tab_performance,
    tab_compare,
    tab_about,
) = st.tabs(
    [
        "🔮 Predict",
        "📊 Data Insights",
        "📈 Performance",
        "⚖️ Model Battle",
        "ℹ️ Info",
    ]
)

# ============================================================
# TAB 1 - PREDICTION
# ============================================================

with tab_predict:
    st.subheader("Interactive Feature Sandbox")

    model_choice = st.radio(
        "Selected Model",
        [
            "Decision Tree",
            "Logistic Regression",
        ],
        horizontal=True,
    )

    col1, col2 = st.columns(2, gap="large")

    input_values = {}

    with col1:
        st.markdown("##### 🎛️ Material Parameters")

        for feature in FEATURES:
            low, high, help_text = FEATURE_INFO[feature]

            if feature in ["Magnetic", "Biodegradable"]:
                input_values[feature] = st.selectbox(
                    feature,
                    options=[0, 1],
                    format_func=lambda value: (
                        "Yes (1)" if value == 1 else "No (0)"
                    ),
                    help=help_text,
                )
            else:
                default_value = float(data[feature].mean())

                step_value = max(
                    0.01,
                    float(high - low) / 100,
                )

                input_values[feature] = st.slider(
                    feature,
                    min_value=float(low),
                    max_value=float(high),
                    value=float(
                        np.clip(
                            default_value,
                            low,
                            high,
                        )
                    ),
                    step=step_value,
                    help=help_text,
                )

    input_df = pd.DataFrame(
        [input_values],
        columns=FEATURES,
    )

    selected_model = (
        dt
        if model_choice == "Decision Tree"
        else lr
    )

    predicted_class = selected_model.predict(input_df)[0]
    predicted_label = encoder.inverse_transform(
        [predicted_class]
    )[0]

    probabilities = selected_model.predict_proba(input_df)[0]

    # Map probabilities correctly to encoder class labels.
    model_classes = selected_model.classes_

    probability_map = {}

    for encoded_class, probability in zip(
        model_classes,
        probabilities,
    ):
        label = encoder.inverse_transform(
            [int(encoded_class)]
        )[0]
        probability_map[label] = float(probability)

    probability_df = pd.DataFrame(
        {
            "WasteType": list(probability_map.keys()),
            "Probability": list(probability_map.values()),
        }
    ).sort_values(
        "Probability",
        ascending=False,
    )

    with col2:
        st.markdown("##### 🎯 Classification Result")

        st.markdown(
            f"""
            <div class="prediction-card">
                <span style="color:#a0aec0;">
                    Predicted Category
                </span>
                <div class="prediction-title">
                    🏷️ {predicted_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Confidence")

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.barh(
            probability_df["WasteType"],
            probability_df["Probability"],
        )

        ax.set_xlabel("Probability")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()

        for index, value in enumerate(
            probability_df["Probability"]
        ):
            ax.text(
                min(value + 0.01, 0.95),
                index,
                f"{value * 100:.1f}%",
                va="center",
            )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --------------------------------------------------------
    # BATCH PREDICTION
    # --------------------------------------------------------

    st.markdown("---")
    st.subheader("📁 Batch Prediction")

    batch_file = st.file_uploader(
        "Upload a CSV containing the five feature columns",
        type=["csv"],
        key="batch_upload",
    )

    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            batch_df.columns = (
                batch_df.columns.astype(str).str.strip()
            )

            missing_columns = [
                feature
                for feature in FEATURES
                if feature not in batch_df.columns
            ]

            if missing_columns:
                st.error(
                    "Missing columns: "
                    + ", ".join(missing_columns)
                )
            else:
                prediction_input = batch_df[
                    FEATURES
                ].copy()

                for feature in FEATURES:
                    prediction_input[feature] = pd.to_numeric(
                        prediction_input[feature],
                        errors="coerce",
                    )

                invalid_rows = prediction_input.isna().any(
                    axis=1
                )

                if invalid_rows.any():
                    st.error(
                        f"{int(invalid_rows.sum())} row(s) contain "
                        "invalid or missing numeric values."
                    )
                else:
                    predictions = selected_model.predict(
                        prediction_input
                    )

                    batch_df["predicted_WasteType"] = (
                        encoder.inverse_transform(
                            predictions
                        )
                    )

                    st.dataframe(
                        batch_df,
                        use_container_width=True,
                    )

                    csv_data = batch_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "⬇️ Download Predictions",
                        data=csv_data,
                        file_name="waste_predictions.csv",
                        mime="text/csv",
                    )

        except Exception as exc:
            st.error(
                f"Batch prediction failed: {exc}"
            )

# ============================================================
# TAB 2 - DATA INSIGHTS
# ============================================================

with tab_explore:
    st.subheader("📊 Data Explorer")

    st.write(
        f"Dataset contains **{len(data)} rows** "
        f"and **{len(data.columns)} columns**."
    )

    st.dataframe(
        data.head(20),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Target Class Distribution")

        counts = data[TARGET].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(
            counts.index,
            counts.values,
        )

        ax.set_xlabel("Waste Type")
        ax.set_ylabel("Number of Samples")
        ax.tick_params(axis="x", rotation=30)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("##### Feature Correlation Matrix")

        correlation = data[FEATURES].corr()

        fig, ax = plt.subplots(figsize=(6, 4))

        image = ax.imshow(
            correlation.values,
            interpolation="nearest",
            aspect="auto",
        )

        ax.set_xticks(
            range(len(FEATURES))
        )
        ax.set_yticks(
            range(len(FEATURES))
        )

        ax.set_xticklabels(
            FEATURES,
            rotation=45,
            ha="right",
        )
        ax.set_yticklabels(FEATURES)

        fig.colorbar(image, ax=ax)

        # Display correlation values.
        for row in range(len(FEATURES)):
            for column in range(len(FEATURES)):
                ax.text(
                    column,
                    row,
                    f"{correlation.iloc[row, column]:.2f}",
                    ha="center",
                    va="center",
                )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("##### Feature Distribution")

    selected_feature = st.selectbox(
        "Select a feature",
        FEATURES,
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    data.boxplot(
        column=selected_feature,
        by=TARGET,
        ax=ax,
    )

    plt.suptitle("")
    ax.set_title(
        f"{selected_feature} by Waste Type"
    )
    ax.set_xlabel("Waste Type")
    ax.set_ylabel(selected_feature)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# TAB 3 - MODEL PERFORMANCE
# ============================================================

with tab_performance:
    st.subheader("📈 Model Performance")

    model_tab_choice = st.radio(
        "Evaluate Model",
        [
            "Decision Tree",
            "Logistic Regression",
        ],
        horizontal=True,
        key="performance_model",
    )

    if model_tab_choice == "Decision Tree":
        predictions = dt_pred
        classifier = dt
    else:
        predictions = lr_pred
        classifier = lr

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Overall Accuracy",
        f"{accuracy * 100:.2f}%",
    )

    metric2.metric(
        "Training Samples",
        len(X_train),
    )

    metric3.metric(
        "Testing Samples",
        len(X_test),
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Confusion Matrix")

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=range(len(class_names)),
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        image = ax.imshow(
            matrix,
            interpolation="nearest",
            aspect="auto",
        )

        ax.set_xticks(
            range(len(class_names))
        )
        ax.set_yticks(
            range(len(class_names))
        )

        ax.set_xticklabels(
            class_names,
            rotation=45,
            ha="right",
        )
        ax.set_yticklabels(class_names)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        fig.colorbar(image, ax=ax)

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("##### Classification Report")

        report = classification_report(
            y_test,
            predictions,
            labels=range(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).T.round(3)

        st.dataframe(
            report_df,
            use_container_width=True,
        )

    if model_tab_choice == "Decision Tree":
        st.markdown("##### 🌳 Decision Tree Structure")

        fig, ax = plt.subplots(
            figsize=(16, 9)
        )

        plot_tree(
            dt,
            feature_names=FEATURES,
            class_names=class_names,
            filled=True,
            rounded=True,
            ax=ax,
            fontsize=8,
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# TAB 4 - MODEL COMPARISON
# ============================================================

with tab_compare:
    st.subheader("⚖️ Head-to-Head Model Comparison")

    dt_accuracy = accuracy_score(
        y_test,
        dt_pred,
    )

    lr_accuracy = accuracy_score(
        y_test,
        lr_pred,
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Decision Tree",
        f"{dt_accuracy * 100:.2f}%",
    )

    col2.metric(
        "Logistic Regression",
        f"{lr_accuracy * 100:.2f}%",
    )

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "Decision Tree",
                "Logistic Regression",
            ],
            "Accuracy": [
                dt_accuracy,
                lr_accuracy,
            ],
        }
    )

    st.dataframe(
        comparison_df.assign(
            Accuracy_Percentage=(
                comparison_df["Accuracy"] * 100
            ).round(2)
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    bars = ax.bar(
        comparison_df["Model"],
        comparison_df["Accuracy"],
    )

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)

    for bar, value in zip(
        bars,
        comparison_df["Accuracy"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.02, 0.98),
            f"{value * 100:.1f}%",
            ha="center",
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if dt_accuracy > lr_accuracy:
        st.success(
            "🏆 Decision Tree has the higher test accuracy."
        )
    elif lr_accuracy > dt_accuracy:
        st.success(
            "🏆 Logistic Regression has the higher test accuracy."
        )
    else:
        st.info(
            "Both models have the same test accuracy."
        )

# ============================================================
# TAB 5 - ABOUT
# ============================================================

with tab_about:
    st.subheader("ℹ️ About Waste Type Classifier")

    st.markdown(
        """
        ### Project Overview

        This application predicts the type of waste from
        physical characteristics of a waste item.

        ### Input Features

        - **Weight** – item weight in grams
        - **Moisture** – moisture percentage
        - **Hardness** – material hardness score
        - **Magnetic** – whether the material is magnetic
        - **Biodegradable** – whether the material is biodegradable

        ### Waste Categories

        The built-in demo dataset contains:

        - Plastic
        - Metal
        - Paper
        - Glass
        - Organic

        ### Machine Learning Models

        **Decision Tree Classifier**

        Uses a tree-based series of decisions to classify
        the waste type.

        **Logistic Regression**

        Uses scaled numerical features and class
        probabilities to predict the waste type.

        ### CSV Format

        Your training CSV should contain exactly these
        required columns:

        `Weight, Moisture, Hardness, Magnetic, Biodegradable, WasteType`

        Example:

        `50, 5, 4, 0, 0, Plastic`

        `150, 2, 9, 1, 0, Metal`

        `20, 25, 2, 0, 1, Paper`
        """
    )

# ============================================================
# 6. END
# ============================================================
