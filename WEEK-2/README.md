# Yuva Intern – Week 2: Data Collection, Cleaning and Preprocessing

## Project
**Logistics Data Analyst Internship – Week 2**

**Author:** Aman Tiwari

## Objective
This week focuses on simulating a logistics data preprocessing pipeline. The work covers data collection simulation, data inspection, missing-value handling, duplicate removal, data-type correction, outlier detection/treatment, and normalization using Python.

The Week 1 logistics dataset is used as the base. Public logistics datasets were reviewed as references for realistic supply-chain fields, while the included working files are clearly marked as an illustrative/simulated dataset.

## Files
- `logistics_data_raw_week2.csv` – raw simulated collection with controlled duplicate records and the missing values already present in the Week 1 base data.
- `logistics_data_cleaned.csv` – cleaned dataset after duplicate removal, data-type correction, missing-value treatment and outlier capping.
- `logistics_data_preprocessed.csv` – cleaned dataset plus Min-Max normalized numerical features.
- `YuvaIntern_Week2_Logistics_Preprocessing.py` – complete Python preprocessing pipeline.
- `YuvaIntern_Week2_Data_Collection_Cleaning_Preprocessing_Aman_Tiwari.docx` – final Week 2 report.

## Preprocessing Workflow
1. Load and inspect the raw dataset.
2. Check data types, missing values and duplicates.
3. Remove exact duplicate records.
4. Convert dates and numeric columns to appropriate types.
5. Impute missing numerical values using the median.
6. Standardize categorical text values.
7. Detect outliers using the IQR rule.
8. Cap extreme values instead of automatically deleting valid-looking logistics observations.
9. Apply Min-Max normalization to selected continuous variables.
10. Export cleaned and preprocessed datasets.

## Dataset
The simulated logistics data contains shipment ID, delivery date, region, vehicle type, traffic level, distance, shipment volume, promised delivery time, actual delivery time, transportation cost, on-time status and delay status.

## Result
The raw simulated file contains **503 rows**. After removing **3 exact duplicate records**, the cleaned dataset contains **500 rows**. Missing numeric values are handled using median imputation, and selected numerical features are normalized to a 0–1 range.

## Tools
- Python
- Pandas
- NumPy
- Scikit-learn

## Next Step
The preprocessed dataset will be used as the input for Week 3: Advanced Data Analysis and Visualization in Logistics.
