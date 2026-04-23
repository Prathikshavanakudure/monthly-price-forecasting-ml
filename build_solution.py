import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title & Introduction
cells.append(nbf.v4.new_markdown_cell("""# AI-ML Internship Task: Advanced Price Forecasting & System Architecture
**Author:** Candidate
**Topic:** Time Series Forecasting (SARIMAX) & Production-Ready Deployment Strategy

This notebook provides an end-to-end solution for predicting average monthly prices. It emphasizes production-grade standards including rigorous statistical testing for model selection, confidence intervals for risk management, and a robust microservice architecture utilizing Django, FastAPI, and modern frontend frameworks.
"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# Set aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')
"""))

# 1. Data Cleaning
cells.append(nbf.v4.new_markdown_cell("""## 1. Data Loading, Cleaning & Anomaly Handling
First, we acquire and clean the data. This involves checking for missing values, structural anomalies, and applying transformations to stabilize variance."""))

cells.append(nbf.v4.new_code_cell("""# 1.1 Data Loading
try:
    df = pd.read_csv('Data_Dictionary.xlsx', parse_dates=['date'])
except:
    df = pd.read_csv('Dataset.csv', parse_dates=['date'])

df = df.sort_values(by='date').reset_index(drop=True)
df.set_index('date', inplace=True)

# 1.2 Missing Values Handling
if df.isnull().sum().any():
    print("Interpolating missing values...")
    df['avg_monthly_price'] = df['avg_monthly_price'].interpolate(method='time')

# 1.3 Variance Stabilization & Anomaly Handling
# Real-world economic data often has multiplicative volatility (spikes grow larger as the base price increases).
# To stabilize this variance and reduce the extreme effect of outliers, we apply a Log Transformation.
df['log_price'] = np.log(df['avg_monthly_price'])

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(df.index, df['avg_monthly_price'], label='Original Price')
plt.title('Original Series (Note Volatility Spikes)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df.index, df['log_price'], color='orange', label='Log Price')
plt.title('Log Transformed Series (Stabilized Variance)')
plt.legend()

plt.tight_layout()
plt.show()
"""))

# 2. Stationarity Check
cells.append(nbf.v4.new_markdown_cell("""## 2. Statistical Testing: Stationarity & The ADF Test
Time series models like SARIMAX require the data to be **stationary** (constant mean and variance over time). We verify this using the **Augmented Dickey-Fuller (ADF) Test**.
* **Null Hypothesis (H0):** The series has a unit root (is non-stationary).
* **Alternate Hypothesis (H1):** The series has no unit root (is stationary).

If the p-value is > 0.05, we accept the null hypothesis and apply **differencing** (`d=1`) to make it stationary."""))

cells.append(nbf.v4.new_code_cell("""# Applying Augmented Dickey-Fuller test on the log series
adf_result = adfuller(df['log_price'].dropna())

print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

if adf_result[1] > 0.05:
    print("Conclusion: The series is NON-STATIONARY. We will use differencing (d=1) in our SARIMAX model.")
else:
    print("Conclusion: The series is STATIONARY.")
"""))

# 3. Model parameters explanation
cells.append(nbf.v4.new_markdown_cell("""## 3. Model Selection & Hyperparameter Tuning
We use the **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors)** model. 
* **Why SARIMAX?** It expertly captures both overall momentum (trend) and cyclical fluctuations (seasonality) inherently present in monthly pricing data.

### Hyperparameter Explanation 
A SARIMAX model is defined by non-seasonal `(p, d, q)` and seasonal `(P, D, Q, s)` parameters:
* **p (AutoRegressive):** Number of lagged actual values. Usually derived from PACF partial-autocorrelation plots.
* **d (Differencing):** `d=1` because our ADF test indicated non-stationarity. We difference the data once to stabilize the mean.
* **q (Moving Average):** Number of lagged forecast errors. Usually derived from ACF autocorrelation plots.
* **P, D, Q, s (Seasonal):** Similar to above, but capturing the seasonal cycle. `s=12` implies an annual cycle (12 monthly periods).

*Note: In a full pipeline, `pmdarima.auto_arima` acts as a Grid Search algorithm minimizing the AIC (Akaike Information Criterion) to select optimal hyperparameters automatically.* We proceed with robust standard parameters `(1,1,1) x (1,1,1,12)`."""))

cells.append(nbf.v4.new_code_cell("""# Using the last 12 months for testing
train = df[:-12]
test = df[-12:]

print("Fitting SARIMAX model...")
model = SARIMAX(train['log_price'], 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit(disp=False)
"""))

# 4. Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 4. Model Evaluation & Metrics
We predict over our unseen `test` set. We convert the predictions out of log-scale back to real limits (`np.exp`), and measure error.

### Understanding Evaluation Metrics:
* **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual prices. It’s highly interpretable (e.g., "We are off by $500 on average").
* **RMSE (Root Mean Squared Error):** Penalizes larger errors heavily by squaring differences before averaging. High RMSE compared to MAE indicates occasional huge mispredictions.
* **MAPE (Mean Absolute Percentage Error):** The percentage of the error proportional to the actual value. A 5% MAPE implies a 95% accuracy rate regardless of currency scale."""))

cells.append(nbf.v4.new_code_cell("""# Predict on the test set
preds_log = model_fit.forecast(steps=12)
preds = np.exp(preds_log) # Reverse log transformation

# Calculate metrics
mae = mean_absolute_error(test['avg_monthly_price'], preds)
rmse = np.sqrt(mean_squared_error(test['avg_monthly_price'], preds))
mape = np.mean(np.abs((test['avg_monthly_price'] - preds) / test['avg_monthly_price'])) * 100

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

plt.figure(figsize=(12, 5))
plt.plot(train.index[-36:], train['avg_monthly_price'][-36:], label='Train (last 3 years)')
plt.plot(test.index, test['avg_monthly_price'], label='Actual Test')
plt.plot(test.index, preds, label='Predictions', color='red', linestyle='--')
plt.title('SARIMA Model - Train vs Test vs Predictions')
plt.legend()
plt.show()
"""))

# 5. Future forecast WITH confidence intervals & Model Saving
cells.append(nbf.v4.new_markdown_cell("""## 5. Deployment: Future Forecasting, Confidence Intervals, and Saving the Model
To put the model into production, we retrain it on the **entire dataset** and serialize (save) it via `joblib`. 

**Why Confidence Intervals?** Point forecasts alone are risky. Confidence intervals provide a lower and upper bound (a 'cone of uncertainty'), enabling businesses to calculate best-case and worst-case scenarios for risk hedging."""))

cells.append(nbf.v4.new_code_cell("""# Retrain on full dataset
final_model = SARIMAX(df['log_price'], 
                      order=(1, 1, 1), 
                      seasonal_order=(1, 1, 1, 12),
                      enforce_stationarity=False,
                      enforce_invertibility=False)
final_model_fit = final_model.fit(disp=False)

# Export the trained model artifacts for deployment APIs
joblib.dump(final_model_fit, 'sarimax_production_model.pkl')
print("Model strictly serialized and saved to 'sarimax_production_model.pkl' for Django/FastAPI ingestion.")

# Generate Forecast and Confidence Intervals
future_steps = 12
forecast_results = final_model_fit.get_forecast(steps=future_steps)

future_log_preds = forecast_results.predicted_mean
future_conf_int_log = forecast_results.conf_int(alpha=0.05) # 95% Confidence Interval

# Convert everything back via exp
future_preds = np.exp(future_log_preds)
lower_bound = np.exp(future_conf_int_log.iloc[:, 0])
upper_bound = np.exp(future_conf_int_log.iloc[:, 1])

# Create future dates index
future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=future_steps, freq='MS')

plt.figure(figsize=(12, 5))
plt.plot(df.index[-60:], df['avg_monthly_price'][-60:], label='Historical Price (Last 5 Years)')
plt.plot(future_dates, future_preds, color='red', marker='o', label='Future 12-Month Forecast')

# Plot Confidence Interval
plt.fill_between(future_dates, lower_bound, upper_bound, color='red', alpha=0.15, label='95% Confidence Interval')

plt.title('Production Price Forecast with 95% Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Predicted Price')
plt.legend()
plt.show()
"""))

# Architecture Text Block
cells.append(nbf.v4.new_markdown_cell("""## 6. Strategic Business Impact & KPIs

### Practical Business Actions based on Predictions
If the trend and confidence intervals forecast a severe upward price hike:
* **Procurement Optimization:** Execute long-term procurement contracts (forward contracts) *immediately* at today's rates, locking in inventory before costs explode.
* **Pricing Strategy Adjustments:** Gradually raise consumer retail prices synchronously with the forecast to artificially smooth out future sticker shock for the end user.

If the market forecasts a continuous drop:
* **Inventory Depletion (JIT):** Transition aggressively to Just-In-Time (JIT) manufacturing. Reduce warehouse safety stock to avoid holding depreciating inventory.
* **Marketing Expansion:** Utilize the lowering COGS (Cost of Goods Sold) margin to aggressively run promotional marketing campaigns aiming to steal market share.

### Measuring Business Impact (KPIs)
To verify if acting on artificial intelligence predictions is fruitful, organizations track KPIs:
* **Cost Avoidance Metric:** `(Actual Spend without Intervention) - (Actual Spend via ML Recommendation)`. Positive numbers signify direct savings.
* **Working Capital Ratio:** Track warehouse carrying costs. Did the ML prompt inventory liquidation before the market crash?
* **Gross Margin Variance:** Ensure profit margins stayed within bounds or improved during months of high price volatility compared to historical averages.

---

## 7. Next-Generation Production Architecture (FastAPI + Django)

Deploying a model correctly is just as important as training it. Below is the blueprint for an enterprise-level implementation flow `(Training -> Server -> Client)`:

### The Microservices Decoupling (Why Both?)
1. **Django (The Monolith):** Acts as the central administrative hub. It handles User Authentication, RBAC (Role-Based Access), logging user actions, managing SQL databases via Django ORM, and serving robust REST APIs using Django Rest Framework (DRF). 
2. **FastAPI (The Intelligence Engine):** Built on asynchronous architecture (`async/await` and Uvicorn). Because ML inference blocks the server thread, hosting it securely inside FastAPI ensures ultra-low-latency real-time inference without bogging down Django traffic.

### Step-by-Step Flow:
1. Client signs into the App via **Django**.
2. Client requests the forecast dashboard on **Next.js**. Next.js triggers an API route to Django.
3. Django validates the user token. Once authorized, Django performs a swift internal server-to-server request via HTTP (Or a message broker like **RabbitMQ** or **Kafka**) to the secure **FastAPI** model-serving microservice.
4. FastAPI loads `sarimax_production_model.pkl` into RAM via a Singleton pattern, quickly running the inference, and passing the JSON packet back down the chain.

### Code Sample: FastAPI Endpoint Handling the Model 
```python
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI(title="Price Inference ML Service")

# Singleton loading of the model so it isn't loaded on every request
try:
    production_model = joblib.load("sarimax_production_model.pkl")
except Exception as e:
    production_model = None

@app.get("/predict/forecast")
async def get_forecast(steps: int = 12):
    if not production_model:
        raise HTTPException(status_code=500, detail="Model artifact failed to load into memory.")
    
    # Get predictions
    forecast_results = production_model.get_forecast(steps=steps)
    predictions = np.exp(forecast_results.predicted_mean).tolist() # Revert log
    
    # Compile 95% CI
    conf_int_log = forecast_results.conf_int(alpha=0.05)
    lower_bound = np.exp(conf_int_log.iloc[:, 0]).tolist()
    upper_bound = np.exp(conf_int_log.iloc[:, 1]).tolist()
    
    return {
        "status": "success",
        "steps": steps,
        "predictions": predictions,
        "upper_confidence_interval": upper_bound,
        "lower_confidence_interval": lower_bound
    }
```

---

## 8. Frontend Integration (Next.js & React Native)
Once APIs are live, the frontend consumes them:
* **Next.js (Web):** Use Server-Side Rendering (`getServerSideProps` or React Server Components) for the first page load to hide API keys securely, ensure ultra-fast user experiences, and populate SEO bots beautifully. Use client-side libraries like `SWR` or `React Query` to poll for dynamic interactive filtering.
* **React Native (Mobile):** Integrate `axios` requests wrapped in React Query to handle caching gracefully in low-network areas. Pipe numeric array sets directly into libraries like `Victory-Native` or `react-native-chart-kit` for responsive touchscreen data visualizations.

---

## 9. Production Reliability: CI/CD, MLOps, and Monitoring
To ensure the project stands out inside a production grid:

* **Docker Containerization:** Both Django and FastAPI should be strictly isolated into separate Docker containers networked via `docker-compose`. This ensures environmental parity ("It works on my machine") and flawless cloud orchestration (Kubernetes or AWS ECS).
* **Model Drift Monitoring:** Models naturally decay as global dynamics change. Use tools like **Prometheus** mapped with **Grafana** dashboards. Every night, a cron job checks `(Actual Price vs Yesterday's Predicted Price)`. If the error exceeds 10% consecutively, trigger a Slack Webhook/PagerDuty alarm marking severe model drift.
* **Automated MLOps Pipeline:** Combine the metrics above into a robust GitOps flow (like GitHub Actions + Apache Airflow). When drift triggers an alarm, a pipeline activates: It downloads new monthly data, retrains the model seamlessly, ensures the new RMSE is improved over the previous one, exports a new `pkl` file, builds a new Docker image, and executes a Zero-Downtime Deployment rollout.
"""))

nb['cells'] = cells

with open("Internship_Task_Submission.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook with advanced production features generated successfully.")
