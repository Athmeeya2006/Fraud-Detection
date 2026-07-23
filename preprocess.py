"""
========================================================================
TOPICS 3 & 4: Data Preprocessing
========================================================================
WHAT WE LEARN HERE:
  - Feature Scaling: WHY and HOW to normalize Time & Amount
  - Train/Test Splitting: WHY temporal splits beat random splits
  - SMOTE: HOW to create synthetic fraud examples for training

KEY CONCEPTS:
  - StandardScaler: transforms features to mean=0, std=1
  - Temporal split: train on past, test on future (realistic)
  - SMOTE: generates new fraud examples by interpolating between
    existing fraud examples in feature space
  - NEVER apply SMOTE to the test set!
========================================================================
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def scale_features(train_df, test_df):
    """
    TOPIC: Feature Scaling
    ----------------------
    WHY: Logistic Regression (and many models) are sensitive to feature scales.
         Amount ranges from 0 to ~25,000 while V1-V28 are ~(-5, 5).
         Without scaling, Amount would dominate the model.

    HOW: StandardScaler transforms each feature to have mean=0 and std=1.
         Formula: X_scaled = (X - mean) / std

    WHAT WE SCALE: Only Time and Amount.
         V1-V28 are already PCA-transformed (which includes scaling).

    CRITICAL - NO DATA LEAKAGE:
         The scaler is FIT on the training set only, then used to TRANSFORM
         both train and test. Fitting on the full dataset (train + test)
         would leak information about the future (test) into preprocessing,
         inflating results. Each feature gets its own scaler so the mean/std
         are independent.
    """
    print("=" * 60)
    print("PREPROCESSING STEP 1: Feature Scaling (fit on train only)")
    print("=" * 60)

    train_df = train_df.copy()
    test_df = test_df.copy()

    time_scaler = StandardScaler()
    amount_scaler = StandardScaler()

    # Fit on TRAIN, transform BOTH (no leakage)
    train_df['Time_scaled'] = time_scaler.fit_transform(train_df[['Time']])
    test_df['Time_scaled'] = time_scaler.transform(test_df[['Time']])

    train_df['Amount_scaled'] = amount_scaler.fit_transform(train_df[['Amount']])
    test_df['Amount_scaled'] = amount_scaler.transform(test_df[['Amount']])

    # Drop original unscaled columns
    train_df = train_df.drop(['Time', 'Amount'], axis=1)
    test_df = test_df.drop(['Time', 'Amount'], axis=1)

    print(f"  Scaler fit on {len(train_df):,} training rows only")
    print("  After scaling  - Amount_scaled: train mean≈0, std≈1")
    print("                   Time_scaled:   train mean≈0, std≈1")
    print(f"  Feature columns: {[c for c in train_df.columns if c != 'Class']}")

    return train_df, test_df, time_scaler, amount_scaler


def temporal_train_test_split(df, test_ratio=0.2):
    """
    TOPIC: Train/Test Splitting (Temporal)
    ----------------------------------------
    WHY TEMPORAL (not random)?
      In real fraud detection, you train on PAST data and predict FUTURE
      transactions. Random splitting would leak future patterns into
      training - that's unrealistic and gives inflated results.

    HOW: Sort by time (already sorted in this dataset), then take the
         first 80% as training, last 20% as testing.

    WHAT'S THE RISK OF RANDOM SPLIT?
      Random split mixes future and past data. Your model might learn
      patterns that only exist in the future, making it appear better
      than it actually is. This is called "data leakage."
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING STEP 2: Temporal Train/Test Split")
    print("=" * 60)

    # The data is already sorted by Time. Split BEFORE scaling so the
    # scaler never sees the test set (see scale_features).
    df = df.sort_values('Time').reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    print(f"  Total samples: {len(df):,}")
    print(f"  Training: {len(train):,} ({len(train)/len(df)*100:.1f}%)")
    print(f"  Testing:  {len(test):,} ({len(test)/len(df)*100:.1f}%)")
    print(f"\n  Training fraud: {train['Class'].sum()} / {len(train)} "
          f"({train['Class'].mean()*100:.3f}%)")
    print(f"  Testing fraud:  {test['Class'].sum()} / {len(test)} "
          f"({test['Class'].mean()*100:.3f}%)")

    return train, test


def apply_smote(X_train, y_train, random_state=42):
    """
    TOPIC: SMOTE (Synthetic Minority Over-sampling TEchnique)
    ----------------------------------------------------------
    WHY? With only ~400 fraud examples in training, models don't see
         enough fraud to learn the pattern. SMOTE creates NEW synthetic
         fraud examples.

    HOW DOES SMOTE WORK?
      1. Pick a fraud example
      2. Find its k nearest fraud neighbors
      3. Draw a line between them in feature space
      4. Place a new synthetic point somewhere on that line

      Example: If fraud_1 = [1, 2, 3] and fraud_2 = [3, 4, 5],
      a synthetic point might be [2, 3, 4] (midpoint).

    CRITICAL RULE: Apply SMOTE ONLY to training data!
      If you SMOTE the test set, you're testing on fake data.
      Your metrics would be meaningless.

    ALTERNATIVE: class_weight='balanced' in models (we use both to compare)
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING STEP 3: SMOTE (Synthetic Oversampling)")
    print("=" * 60)

    print("\n  Before SMOTE:")
    print(f"    Legit:  {(y_train == 0).sum():,}")
    print(f"    Fraud:  {(y_train == 1).sum():,}")

    smote = SMOTE(random_state=random_state, sampling_strategy=1.0)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print("\n  After SMOTE:")
    print(f"    Legit:  {(y_resampled == 0).sum():,}")
    print(f"    Fraud:  {(y_resampled == 1).sum():,}")
    print(f"    Total:  {len(y_resampled):,}")
    print(f"\n  SMOTE created {(y_resampled == 1).sum() - (y_train == 1).sum():,} "
          f"synthetic fraud examples")

    return X_resampled, y_resampled


def preprocess(data_path='creditcard.csv', test_ratio=0.2, random_state=42):
    """
    Full preprocessing pipeline (leakage-safe ordering):
    1. Load data
    2. Temporal train/test split  (split FIRST)
    3. Scale features (fit scaler on train only, transform both)
    4. Apply SMOTE to training data only

    Returns a dict consumed by every downstream module. Note that
    ``amount_test`` carries the ORIGINAL (unscaled) test-set transaction
    amounts so the cost/real-time modules can reason in real dollars
    without re-reading the raw CSV.
    """
    print("\n" + "=" * 60)
    print("FULL PREPROCESSING PIPELINE")
    print("=" * 60)

    df = pd.read_csv(data_path)

    # Split first, THEN scale (prevents test data leaking into the scaler)
    train_df, test_df = temporal_train_test_split(df, test_ratio=test_ratio)

    # Keep the original, unscaled test amounts for the $ analyses
    amount_test = test_df['Amount'].to_numpy()

    train_scaled, test_scaled, time_scaler, amount_scaler = scale_features(
        train_df, test_df
    )

    X_train = train_scaled.drop('Class', axis=1)
    y_train = train_scaled['Class']
    X_test = test_scaled.drop('Class', axis=1)
    y_test = test_scaled['Class']

    X_train_smote, y_train_smote = apply_smote(
        X_train, y_train, random_state=random_state
    )

    return {
        'X_train': X_train,           # Original training data (no SMOTE)
        'X_test': X_test,
        'y_train': y_train,           # Original training labels
        'y_test': y_test,
        'X_train_smote': X_train_smote,  # SMOTE-resampled training data
        'y_train_smote': y_train_smote,  # SMOTE-resampled training labels
        'amount_test': amount_test,      # Original unscaled test amounts ($)
        'feature_names': list(X_train.columns),
        'time_scaler': time_scaler,
        'amount_scaler': amount_scaler,
    }


if __name__ == '__main__':
    data = preprocess()
    print("\n✅ Preprocessing complete!")
    print(f"   X_train shape:       {data['X_train'].shape}")
    print(f"   X_train_smote shape: {data['X_train_smote'].shape}")
    print(f"   X_test shape:        {data['X_test'].shape}")
