# Weather Trend Forecasting

## About This Project
This project uses the Global Weather Repository dataset to look at weather patterns and forecast temperature trends over time. I used it as a way to practice a full data science workflow, including data cleaning, visualization, forecasting, and model evaluation.

I wanted to keep the project practical and easy to follow. Instead of making it too complicated, I focused on understanding the dataset first, exploring the patterns in the data, building a reasonable model, and then explaining the results clearly.

## PM Accelerator Mission
PM Accelerator’s mission is to break down financial barriers and achieve educational fairness. I included this project as a way to apply data science in a practical and accessible way.

## Dataset
The dataset used in this project is the **Global Weather Repository** dataset from Kaggle. It contains daily weather information for cities around the world, including features related to temperature, precipitation, wind, air quality, and other weather conditions.

The original dataset is available on Kaggle. Because the raw file is relatively large, I did not upload it directly to this repository. To run the project, please download the dataset from Kaggle and place the CSV file inside the `data/` folder.

Expected file name: `GlobalWeatherRepository.csv`

## What I Worked On
In this project, I:
- cleaned and inspected the dataset
- checked missing values and possible outliers
- explored temperature and precipitation trends
- used the `last_updated` feature for time-based analysis
- built and compared forecasting models
- looked at feature importance to better understand the model

## Project Structure
- `data/` contains the dataset
- `notebooks/` contains the main analysis code
- `outputs/figures/` contains the saved plots
- `requirements.txt` lists the libraries used for the project

## How to Run
1. Install the required libraries:
   `pip install -r requirements.txt`

2. Run the main script:
   `python notebooks/weather_trend_forecasting.py`

## Results
I compared a simple baseline model with a Random Forest model for temperature forecasting.

### Baseline model
- MAE: 10.2418
- RMSE: 14.0818
- R²: -0.5930

### Random Forest model
- MAE: 1.2883
- RMSE: 2.2715
- R²: 0.9585

The Random Forest model performed much better than the baseline. This suggests that the time-based features and weather-related variables were useful for forecasting temperature.

## Feature Importance
The most important features in the Random Forest model were:
- `temp_roll3`
- `temp_lag1`
- `temp_lag2`
- `pressure_mb`
- `humidity`

This suggests that recent temperature history, along with a few weather-related conditions, played the biggest role in short-term forecasting.

## Note
This project is meant to be simple, readable, and easy to follow. I tried to keep both the code and the explanation clear.

## Author
Siqi Wang
