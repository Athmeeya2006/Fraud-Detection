"""
========================================================================
FULL PIPELINE: Run Everything End-to-End
========================================================================
This script ties together all the topics:

  1. EDA               → Understand the data (eda.py)
  2. Preprocess        → Scale, split, SMOTE (preprocess.py)
  3. Train (Base)      → LogReg, RandomForest, XGBoost (models.py)
  4. Train (Advanced)  → LightGBM, Neural Network (advanced_models.py)
  5. Evaluate          → Metrics, curves, threshold tuning (evaluate.py)
  6. Cross-Validation  → K-Fold stability analysis (cross_validation.py)
  7. SHAP Explainability → Feature importance & why (feature_importance.py)
  8. Anomaly Detection → Unsupervised fraud finding (anomaly_detection.py)
  9. Cost Analysis     → Business dollar impact (cost_analysis.py)
  10. Real-Time Sim    → Production scoring demo (realtime_simulator.py)

RUN WITH:
  cd /home/athmeeya/Fraud-Detection
  venv/bin/python run_pipeline.py

All output plots go to outputs/
========================================================================
"""

import time
import os


def main():
    start_time = time.time()

    print("╔" + "═" * 58 + "╗")
    print("║    CREDIT CARD FRAUD DETECTION - FULL PIPELINE          ║")
    print("║    (10 Phases • Supervised + Unsupervised + Business)   ║")
    print("╚" + "═" * 58 + "╝")

    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # ─── PHASE 1: EDA ─────────────────────────────────────────────────
    print("\n\n" + "█" * 60)
    print("█  PHASE 1: EXPLORATORY DATA ANALYSIS")
    print("█" * 60)

    from eda import run_eda
    run_eda(output_dir=output_dir)

    # ─── PHASE 2: PREPROCESSING ───────────────────────────────────────
    print("\n\n" + "█" * 60)
    print("█  PHASE 2: PREPROCESSING (Scale → Split → SMOTE)")
    print("█" * 60)

    from preprocess import preprocess
    data = preprocess()

    # ─── PHASE 3: BASE MODEL TRAINING ─────────────────────────────────
    print("\n\n" + "█" * 60)
    print("█  PHASE 3: BASE MODEL TRAINING (LogReg, RF, XGBoost)")
    print("█" * 60)

    from models import train_all_models
    base_models = train_all_models(data)

    # ─── PHASE 4: ADVANCED MODEL TRAINING ─────────────────────────────
    print("\n\n" + "█" * 60)
    print("█  PHASE 4: ADVANCED MODEL TRAINING (LightGBM, NeuralNet)")
    print("█" * 60)

    from advanced_models import train_advanced_models
    advanced_models = train_advanced_models(data)

    # Combine all models
    all_models = {**base_models, **advanced_models}

    # ─── PHASE 5: EVALUATION ──────────────────────────────────────────
    print("\n\n" + "█" * 60)
    print("█  PHASE 5: EVALUATION & THRESHOLD TUNING")
    print("█" * 60)

    from evaluate import evaluate_all
    all_metrics, optimal_threshold = evaluate_all(all_models, data, output_dir=output_dir)

    # Find best model
    best = max(all_metrics, key=lambda x: x['AUPRC'])
    best_model_name = best['Model']
    best_model = all_models[best_model_name]
    print(f"\n  🏆 Best model: {best_model_name} (AUPRC={best['AUPRC']:.4f})")

    # ─── PHASE 6: CROSS-VALIDATION ────────────────────────────────────
    print("\n\n" + "█" * 60)
    print("█  PHASE 6: CROSS-VALIDATION (Stability Analysis)")
    print("█" * 60)

    from cross_validation import run_stratified_cv
    cv_results, fold_data = run_stratified_cv(n_splits=5, output_dir=output_dir)

    # ─── PHASE 7: SHAP EXPLAINABILITY ─────────────────────────────────
    print("\n\n" + "█" * 60)
    print("█  PHASE 7: SHAP EXPLAINABILITY (Why does the model decide?)")
    print("█" * 60)

    from feature_importance import run_feature_importance
    shap_values, shap_sample = run_feature_importance(
        best_model, best_model_name, data, output_dir=output_dir
    )

    # ─── PHASE 8: ANOMALY DETECTION ───────────────────────────────────
    print("\n\n" + "█" * 60)
    print("█  PHASE 8: UNSUPERVISED ANOMALY DETECTION")
    print("█" * 60)

    from anomaly_detection import run_anomaly_detection
    anomaly_metrics, anomaly_scores = run_anomaly_detection(data, output_dir=output_dir)

    # ─── PHASE 9: COST ANALYSIS ───────────────────────────────────────
    print("\n\n" + "█" * 60)
    print("█  PHASE 9: COST-SENSITIVE BUSINESS ANALYSIS")
    print("█" * 60)

    from cost_analysis import run_cost_analysis
    cost_df, cost_threshold, cost_results = run_cost_analysis(
        best_model, best_model_name, data,
        investigation_cost=25.0, chargeback_rate=1.5,
        output_dir=output_dir
    )

    # ─── PHASE 10: REAL-TIME SIMULATION ───────────────────────────────
    print("\n\n" + "█" * 60)
    print("█  PHASE 10: REAL-TIME FRAUD SCORING SIMULATION")
    print("█" * 60)

    from realtime_simulator import run_realtime_simulation
    run_realtime_simulation(
        best_model, best_model_name, data,
        n_transactions=100, output_dir=output_dir
    )

    # ─── DONE ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n\n" + "╔" + "═" * 58 + "╗")
    print(f"║  PIPELINE COMPLETE in {elapsed:.1f}s" + " " * max(36 - len(f"{elapsed:.1f}"), 0) + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n📂 All plots saved to: {os.path.abspath(output_dir)}/")
    print(f"📊 Results CSV: {output_dir}/results.csv")
    print("\nPlots generated:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png') or f.endswith('.csv'):
            print(f"   📈 {f}")


if __name__ == '__main__':
    main()
