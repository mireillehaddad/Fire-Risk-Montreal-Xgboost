# 🔥 Fire Risk Analysis with Weather and Property Data

This project aims to predict fire risk using weather and property data from Montreal and forecast future fire incidents using time-series analysis.

---

## 📁 Datasets Used

- `Weather Data Only.csv`: Contains historical weather measurements by station.
- `Weather Data According to Stations.csv`: Metadata including station coordinates and IDs.
- `evaluation_with_fire_and_coordinates_and_date.csv`: Fire incident records mapped with coordinates and dates.

---

## ⚙️ Process Flow

1. **Data Cleaning and Merging**
   - Combined weather data using `Weather_dataset_cleaned.ipynb` to produce `weather_with_station_names.csv`.

2. **Modeling**
   - **Random Forest Classifier** (`ML_Model_Weather_with_Evaluation_Fonciere_latest.ipynb`):
     - Trained to classify whether a fire is likely at a given location based on weather and property features.
     - Most important features: temperature (mean, max, min).
   - **Time-Series Forecasting** (`Time_Series_Forecastingr_Weather_with_Evaluation_Fonciere.ipynb`):
     - Built to forecast number of monthly fires.
     - Used Prophet to capture seasonality, trends, and variability.
     - Observed a decreasing trend with clear seasonal peaks.

---

## 🧠 Why These Models?

- **Random Forest**: Easy to interpret, handles mixed data well, and shows feature importance.
- **Prophet Time-Series**: Good for monthly/yearly trends, interpretable components (trend & seasonality).

---

## 📊 Results

### Random Forest Performance

```
  precision    recall  f1-score   support

       False       0.99      1.00      0.99     73803
        True       1.00      0.99      0.99     58954

    accuracy                           0.99    132757
   macro avg       0.99      0.99      0.99    132757
weighted avg       0.99      0.99      0.99    132757
```

📌 **Feature Importance Plot**  
![Feature Importance](./5a428794-8eed-4cc3-a007-75d3fde4450f.png)

---

### Time-Series Forecast Outputs

- Fire counts plotted monthly.  
  ![Monthly Fires](./24d36db8-30f8-491f-a1b8-5950c339dca3.png)

- Forecasts show a slight decline in expected fire counts.  
  ![Forecast](./03bc4e1b-2f89-41c5-b483-030f8eb3a349.png)

- Seasonal component indicates more fires in summer months.  
  ![Decomposition](./8847883f-dadb-40e8-88d5-66de326af706.png)

---

## ✅ Summary

I successfully combined weather and property data to:

- Predict the **likelihood of fire** using Random Forest.
- Forecast **monthly fire incidents** using Time-Series modeling to support proactive planning.
