# Diabetes prediction with logistic regression

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prantopi/diabetes-prediction/blob/main/diabetes_predictml.ipynb)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)

A logistic regression model that predicts whether a patient has diabetes from eight health measurements. This was one of my first machine learning projects (2023).

## Data

`diabetic.csv` holds 100,000 patient records with these columns:

| Column | Type | Notes |
|---|---|---|
| `gender` | text | Female / Male / Other |
| `age` | number | years |
| `hypertension` | 0 / 1 | |
| `heart_disease` | 0 / 1 | |
| `smoking_history` | text | never, current, former, ever, not current, No Info |
| `bmi` | number | body mass index |
| `HbA1c_level` | number | average blood sugar over ~3 months |
| `blood_glucose_level` | number | mg/dL |
| `diabetes` | 0 / 1 | the label |

## How it works

1. Load the CSV with pandas.
2. Encode `gender` as a boolean and map each `smoking_history` value to an integer.
3. Fit scikit-learn's `LogisticRegression` (`max_iter=5000`) on the eight features.
4. Print accuracy and compare predictions with the real diagnosis for 50 patients.

## Result

**96.0% accuracy**, measured on the same data the model was trained on.

## Run it

```bash
pip install pandas scikit-learn
python diabetic_predict.py
```

Or open the notebook in Colab with the badge above.

## What I'd do differently now

- Hold out a test set (`train_test_split`) so the accuracy reflects unseen patients.
- Only about 8.5% of patients in the data have diabetes, so a model that always says "no" already scores about 91%. Precision, recall and a confusion matrix would show more.
- One-hot encode `smoking_history` instead of mapping it to integers, which implies an order that doesn't exist.
- The gender check compares against `"male"`, but the data uses `"Male"`, so that feature is always false.

## Files

| File | What it is |
|---|---|
| `diabetes_predictml.ipynb` | Colab notebook with outputs |
| `diabetic_predict.py` | Same code as a script |
| `diabetic.csv` | Dataset |
| `gender.ipnyb` | Early exploration notebook |
