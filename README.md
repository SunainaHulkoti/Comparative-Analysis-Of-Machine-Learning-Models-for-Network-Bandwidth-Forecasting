# Infrastructure Dashboard

A real-time infrastructure monitoring dashboard built with Streamlit, utilizing and comparing machine learning models for predictive analytics, anomaly detection, and packet loss detection on network bandwidth and other metrics.

## Features

- **Real-time Data Fetching**: Automatically fetches infrastructure data and refreshes every 5 seconds.
- **Anomaly Detection**: Uses statistical methods to detect anomalies in bandwidth data.
- **Packet Loss Detection**: Identifies potential packet loss issues.
- **Predictive Modeling**: Employs GRU, LSTM, and Prophet models for time series forecasting.
- **Interactive Visualizations**: Displays data using Plotly charts for easy interpretation.
- **Model Evaluation**: Provides metrics like MAE and RMSE for model performance.

## Results

Based on backtesting with historical data, the LSTM model outperformed GRU and Prophet in terms of prediction accuracy for bandwidth metrics, achieving the lowest MAE and RMSE on test sets.

### Performance Comparison
- **LSTM**: Best overall performance.
- **GRU**: Close second, slightly higher error rates.
- **Prophet**: Higher errors, suitable for trend-based forecasting but less precise for short-term predictions.

For detailed results across both trials, refer to the PDF report in the `results/` directory.

[Download Model Results PDF](results/model_results.pdf)


## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd MajorProject
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the trained models in the `artifacts/` directory:
   - `scaler_in.pkl`: Scaler for data normalization.
   - `gru_in.pt`: Trained GRU model.
   - `lstm_in.pt`: Trained LSTM model.
   - `prophet_model.pkl`: Trained Prophet model.

## Usage

Run the Streamlit app:
```
streamlit run app.py
```

Open your browser to the provided URL (usually `http://localhost:8501`) to view the dashboard.

## Project Structure

- `app.py`: Main Streamlit application.
- `models.py`: Definitions for GRU and LSTM models, loading functions.
- `utils.py`: Utility functions for data fetching, preparation, and detection.
- `train_models.py`: Script for training the models (if needed).
- `lastlogger.py`: Logging utilities.
- `requirements.txt`: Python dependencies.
- `artifacts/`: Directory containing trained models and scalers.

## Dependencies

- streamlit
- pymongo
- pandas
- numpy
- plotly
- prophet
- scikit-learn
- torch
- matplotlib
- python-dateutil
- joblib
- streamlit-autorefresh

## License

This project is licensed under the MIT License.