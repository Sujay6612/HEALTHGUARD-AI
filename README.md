# HeartIQ Predictor

This project is now split into two responsibilities:

- `app.py`: serves the Flask app and shows prediction plus model-quality metadata.
- `train_model.py`: trains and evaluates candidate models on a real dataset, then saves the best model.

## Added app features

- Patient search by ID or name
- CSV export for a patient's saved visit history
- Visit trend charts for risk, blood pressure, cholesterol, and ECG
- Local patient tracking with comparison against previous visits

## To improve actual accuracy

1. Add the Cleveland dataset at:
   - `data/processed.cleveland.data`
2. If you downloaded the official UCI `heart+disease.zip`, extract `processed.cleveland.data` from the ZIP and place it in `data/`.
3. Make sure the dataset includes these fields, using either the canonical names below or one of the supported aliases:
   - `age`
   - `sex` or `gender`
   - `trestbps`, `resting_bp`, `restbp`, `restingbloodpressure`, or `blood_pressure`
   - `chol`, `cholesterol`, `serumcholestoral`, or `serum_cholesterol`
   - `restecg`, `resting_ecg`, `ecg`, `restecgresults`, or `restingecg`
   - target column named one of: `target`, `condition`, `output`, `label`, `num`, or `HeartDisease`
4. Supported target formats include:
   - binary `0/1`
   - text labels like `Yes/No`, `True/False`, `Positive/Negative`, `Presence/Absence`
   - severity labels like `0-4`, which are converted to binary automatically (`0` = no disease, `1-4` = disease)
5. Run:

```powershell
.venv\Scripts\python.exe train_model.py
```

6. Start the app:

```powershell
.venv\Scripts\python.exe app.py
```

## What the training script does

- Loads the real dataset
- Trains multiple tuned candidate models using the same 5 app inputs
- Selects the best model by cross-validated ROC-AUC
- Saves:
  - `models/model.pkl`
  - `models/scaler.pkl` for backward compatibility
  - `models/model_metadata.json` with accuracy, precision, recall, and ROC-AUC

## Important note

Until you retrain with a real dataset, the current saved model should still be treated as unverified for real medical use.
