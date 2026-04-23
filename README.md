 **Monthly Price Forecasting using Machine Learning**
 **Project Overview**

This project focuses on predicting average monthly prices for the next 12 months using time series forecasting techniques. The goal is to help businesses make better decisions based on future price trends.

**Features**
Data cleaning and preprocessing
Handling missing values and anomalies
Time series forecasting using SARIMAX
Model evaluation using MAE, RMSE, and MAPE
Future price prediction with confidence intervals
Deployment architecture using Django and FastAPI

 **Tech Stack**
Python
Pandas, NumPy
Matplotlib, Seaborn
Statsmodels (SARIMAX)
Scikit-learn
Joblib
Django & FastAPI

📂 **Project Structure**
├── Assignment.ipynb        # Main notebook with full solution
├── build_solution.py       # Python script
├── Dataset.csv             # Input dataset
├── Data_Dictionary.xlsx    # Dataset explanation
├── README.md               # Project documentation
├── requirements.txt        # Dependencies

**How to Run**
Clone the repository
git clone https://github.com/Prathikshavanakudure/monthly-price-forecasting-ml.git
Install dependencies
pip install -r requirements.txt
Run the notebook
jupyter notebook Assignment.ipynb

 **Model Details**
Model Used: SARIMAX (Time Series Forecasting)
Handles trend and seasonality
Log transformation used for variance stabilization
Differencing applied for stationarity

 **Evaluation Metrics**
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
MAPE (Mean Absolute Percentage Error)

**Deployment Architecture**
Django: Handles authentication, database, APIs
FastAPI: Serves ML model for real-time predictions
Frontend: Can be integrated with Next.js / React Native
Business Impact
Helps in predicting price trends
Supports better inventory and pricing decisions
Reduces financial risk using forecast insights

**Author:** Prathiksha V

**Contact:**
Email: prathikshavanakudure@gmail.com
Phone: 8867033641
