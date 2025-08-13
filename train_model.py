# train_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score, accuracy_score
import logging
import os
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_and_preprocess_data(file_path):
    """Loads and preprocesses the dataset robustly."""
    logging.info("[Step 1] Loading dataset...")
    df = pd.read_csv(file_path)

    # Remove columns not needed
    exclude_cols = ['Year', 'Suitable_Crops', 'Fertilizer_Plan', 'Irrigation_Plan', 
                    'Market_Price_Index', 'Previous_Crop']
    df = df.drop(columns=exclude_cols, errors='ignore')

    # Check target column
    if 'Primary_Crop' not in df.columns:
        raise ValueError("❌ 'Primary_Crop' column is missing in the dataset!")

    X = df.drop(columns='Primary_Crop')
    y = df['Primary_Crop']

    # Numeric and categorical separation
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # Handle numeric missing values
    X_num = X[numerical_cols].copy()

    if not X_num.empty:
        num_cols_with_na = X_num.columns[X_num.isnull().any()].tolist()

        if num_cols_with_na:
            # Drop fully empty numeric columns
            fully_empty = [col for col in num_cols_with_na if X_num[col].isnull().all()]
            if fully_empty:
                logging.warning(f"Dropping fully empty numeric columns: {fully_empty}")
                X_num.drop(columns=fully_empty, inplace=True)
                num_cols_with_na = [c for c in num_cols_with_na if c not in fully_empty]

            if num_cols_with_na:
                logging.info(f"[Step 2] Imputing numeric columns {num_cols_with_na} using KNNImputer...")
                imputer = KNNImputer(n_neighbors=5)
                imputed_values = imputer.fit_transform(X_num[num_cols_with_na])
                for i, col in enumerate(num_cols_with_na):
                    X_num[col] = imputed_values[:, i]
        else:
            logging.info("[Step 2] No missing numeric values to impute.")
    else:
        logging.warning("No numeric columns found in dataset!")

    # Handle categorical missing values
    if categorical_cols:
        X_cat = X[categorical_cols].copy()
        for col in categorical_cols:
            X_cat[col] = X_cat[col].fillna("Unknown")
        logging.info("[Step 3] One-hot encoding categorical features...")
        X_cat_encoded = pd.get_dummies(X_cat, columns=categorical_cols, drop_first=True)
    else:
        X_cat_encoded = pd.DataFrame()
        logging.warning("No categorical columns found in dataset!")

    # Combine numeric + categorical
    X_processed = pd.concat([X_num, X_cat_encoded], axis=1)
    X_processed = X_processed.fillna(0)  # Final safety net for any stray NaNs

    logging.info(f"[Step 4] Final features: X={X_processed.shape}, y={y.shape}")
    return X_processed, y

def filter_and_label_data(X, y, min_samples=100):
    """Removes underrepresented crops and encodes labels."""
    logging.info("[Step 5] Filtering crops by minimum sample count...")
    crop_counts = y.value_counts()
    crops_to_keep = crop_counts[crop_counts >= min_samples].index

    X_filtered = X[y.isin(crops_to_keep)]
    y_filtered = y[y.isin(crops_to_keep)]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_filtered)

    logging.info(f"Crops kept: {list(crops_to_keep)}")
    logging.info(f"Filtered dataset shape: X={X_filtered.shape}, y={len(y_encoded)}, classes={len(le.classes_)}")
    return X_filtered, y_encoded, le.classes_, list(X_filtered.columns)

def train_model(X, y, num_classes, feature_cols):
    """Builds and trains the MLP model."""
    logging.info("[Step 6] Building MLP model...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    logging.info("[Step 7] Training model...")
    model.fit(X, y, epochs=100, batch_size=32, verbose=1)
    return model

def evaluate_model(model, X, y, classes):
    """Evaluates model performance."""
    logging.info("[Step 8] Evaluating model...")
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(ss.split(X, y))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    top3_hits = sum(1 for i in range(len(y_test)) 
                    if y_test[i] in predictions[i].argsort()[-3:][::-1])
    top3_acc = top3_hits / len(y_test)

    logging.info(f"Accuracy: {acc:.3f} | Macro-F1: {f1:.3f} | Top-3 Accuracy: {top3_acc:.3f}")

def save_model(model, classes, feature_cols):
    """Saves model and metadata."""
    logging.info("[Step 9] Saving model...")
    model.save('croprecommender_mlp.h5')
    np.savez('croprecommender_mlp.npz', classes=classes, feature_cols=feature_cols)

if __name__ == '__main__':
    X_initial, y_initial = load_and_preprocess_data('apcrop_dataset_realistic.csv')
    X_filtered, y_encoded, classes, feature_cols = filter_and_label_data(X_initial, y_initial)
    model = train_model(X_filtered, y_encoded, len(classes), feature_cols)
    evaluate_model(model, X_filtered, y_encoded, classes)
    save_model(model, classes, feature_cols)
