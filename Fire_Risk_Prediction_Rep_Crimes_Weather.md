# Combined Fire Risk Prediction Report

## 1. Overview and Purpose

This document combines findings from two analyses aimed at predicting fire risk in Montreal buildings:

**Fire Risk Prediction Project** using property evaluation data and spatial crime data to determine whether crime patterns can predict the likelihood of fire incidents.

**Weather and Evaluation Dataset Fire Risk Report** exploring whether combining weather data with building characteristics improves monthly fire occurrence predictions.

---

## 2. Datasets Provided

- [Montreal firefighters intervene](https://donnees.montreal.ca/en/dataset/interventions-service-securite-incendie-montreal)
- [Fire stations on the island of Montreal](https://donnees.montreal.ca/dataset/casernes-pompiers)
- [Land assessment units](https://donnees.montreal.ca/dataset/unites-evaluation-fonciere)
- [Criminal acts](https://donnees.montreal.ca/dataset/actes-criminels)
- [Census Data](https://www12.statcan.gc.ca/census-recensement/index-eng.cfm)
- [Citizen Service Requests (311 Requests)](https://donnees.montreal.ca/dataset/requete-311)
- [Punctual address](https://donnees.montreal.ca/dataset/adresses-ponctuelles)

---

## 3. Exploratory Data Analysis (EDA)

Basic EDA was performed in all the Data sets used, Also performed missingness analysis on “construction year = 9999” in the “evaluation_with_fire_and_coordinates.csv”.  
<Add all the EDA Performed here>

---

## 4. Data Sources

**Fire Risk Prediction Project Data:**

- **Property Evaluation Dataset:** Property attributes, including fire occurrence labels.
- **Crime Dataset:** Detailed crime incidents (type, date, latitude, longitude).
- **Merged Dataset:** Each property enriched with crime features:
  - Number of crimes within ~150m
  - Average distance to nearby crimes
  - Most common nearby crime type

**Weather Fire Risk Report Data:**

- **Evaluation Foncière Dataset:** Building-level attributes (units, construction year, area, etc.).
- **Weather Dataset:** Monthly aggregated weather variables per station:
  - Minimum and maximum temperatures
  - Mean temperature
  - Average wind speed and gusts

These datasets were merged by aligning timestamps and geographic references.

---

## 5. Feature Engineering

**Crime Features:**

- num_crimes_nearby
- avg_distance_to_crimes
- One-hot encoded crime categories:
  - Mischief
  - Vehicle theft
  - Qualified theft
  - Unknown

**Building Characteristics:**

- MUNICIPALITE
- ETAGE_HORS_SOL
- NOMBRE_LOGEMENT
- AGE_BATIMENT
- RATIO_SURFACE
- DENSITE_LOGEMENT
- CATEGORIE_UEF
- CODE_UTILISATION

**Weather Features:**

- temperature_2m_min (°C)
- temperature_2m_max (°C)
- temperature_2m_mean (°C)
- wind_speed_10m_mean (km/h)
- wind_gusts_10m_mean (km/h)

---

## 6. Modeling Approaches

**Fire Risk Prediction Project:**

- **Random Forest Classifier**
  - **Performance:**
    - Precision Fire: 87%
    - Recall Fire: 83%
    - F1-score Fire: 85%
    - Accuracy: 81%
  - **Feature Importance:**
    - avg_distance_to_crimes: ~60%
    - num_crimes_nearby: ~35%
    - Crime types: ~5%

- **XGBoost Classifier**
  - **Performance:**
    - Precision Fire: 85%
    - Recall Fire: 80%
    - F1-score Fire: 82%
    - Accuracy: 78%
  - **Feature Importance:**
    - avg_distance_to_crimes: 2384 splits
    - num_crimes_nearby: 2130 splits
    - Crime types combined: ~700 splits

**Weather Fire Risk Report:**

- **Random Forest Classifier**
  - **Performance:**
    - Accuracy: 99%
    - Extremely high recall and precision, likely due to repeated monthly records per building.

  **Confusion Matrix:**

  |                | Predicted No Fire | Predicted Fire |
  |----------------|-------------------|----------------|
  | Actual No Fire | 73,803             | 0              |
  | Actual Fire    | 834                | 58,120         |

- **Time Series Forecasting**
  - Aggregated monthly fire counts to visualize seasonal patterns.
  - Used for exploratory analysis, not per-building classification.

{Additional Note -> all the train-test splits were made randomly, i.e. 70-30 train-test split.}

---

## 7. Temporal Modeling

**Fire Risk Prediction Project Considerations:**

- **Panel Approach:**
  - One row per property per month.
  - Enables time series modeling.

  **Example:**

  | ID_UEV | YEAR | MONTH | Features | fire_occurred |
  |--------|------|-------|----------|---------------|
  | 001    | 2020 | 1     | …        | 0             |
  | 001    | 2020 | 2     | …        | 0             |
  | 001    | 2020 | 3     | …        | 1             |

- **Fire Month Approach:**
  - One row per property, fire_month indicating month of fire (1–12) or 13 if none.

  **Example:**

  | ID_UEV | Features | fire_month |
  |--------|----------|------------|
  | 001    | …        | 3          |
  | 002    | …        | 13         |

**Weather Fire Risk Report:**

- Implemented Panel Approach for temporal modeling.
- Captured seasonal and lagged effects.
- Noted risk of overfitting due to repeated patterns.

---

## 8. Interpretation

- Crime proximity and frequency are strong predictors of fire risk.
- Combining weather variables and building attributes yields extremely high recall in per-month prediction.
- Temporal models show seasonal fire patterns and potential for forecasting.
- Very high model accuracy in the weather-based panel dataset suggests caution (potential overfitting).

---

## 9. Next Steps

Recommendations across both datasets:

- Further optimize thresholds to balance precision and recall.
- Use temporal modeling frameworks (Panel and Fire Month).
- Visualize model outputs (e.g., monthly risk scores) on maps.
- Experiment with XGBoost classifiers for weather + building data.

---

## 10. Conclusion

Combining crime data, property evaluation data, and weather observations provides complementary insights into fire risk. Both approaches demonstrate high predictive potential:

- Random Forest models performed robustly on crime-based predictors.
- Weather and building data yielded high per-month predictive accuracy, albeit with possible overfitting.
- Temporal modeling is critical to capturing seasonal trends and improving future prediction granularity.
