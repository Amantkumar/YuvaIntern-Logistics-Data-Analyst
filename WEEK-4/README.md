# Yuva Intern – Week 4: Predictive Modeling and Optimization in Logistics

Author: Aman Tiwari
Internship Track: Logistics Data Analyst
Task: Week 4 – Predictive Modeling and Optimization

## Objective
Forecast delivery time using Python and translate predictive insights into practical logistics optimization strategies.

## Dataset
The analysis uses the cleaned illustrative logistics dataset continued from Week 3. It contains shipment-level information such as region, vehicle type, traffic level, distance, shipment volume, promised time, delivery time, transportation cost, and delivery status.

## Prediction Problem
Target: `delivery_time_hours`

The model predicts actual delivery time from operational, geographic, traffic, vehicle, cost, and calendar features.

## Models
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Evaluation
- MAE
- RMSE
- R²
- 5-fold cross-validation
- GridSearchCV hyperparameter tuning

## Optimization
An illustrative scenario estimates the effect of:
1. reducing route distance by 10% through route planning improvements;
2. reducing high-traffic exposure through scheduling, represented by a High-to-Medium traffic scenario.

These are scenario assumptions for analytical demonstration, not measured real-world savings.

## Files
- `YuvaIntern_Week4_Predictive_Modeling_Optimization_Aman_Tiwari.docx`
- `YuvaIntern_Week4_Logistics_Predictive_Modeling.py`
- `logistics_data_cleaned.csv`
- `model_comparison_metrics.csv`
- `five_fold_cv_rmse.csv`
- `hyperparameter_tuning_summary.csv`
- `optimization_scenario_summary.csv`
- `Week4_Model_Results.txt`
- `charts/`

## Data Note
The logistics dataset is simulated/illustrative and should not be interpreted as actual company operational data.
