import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import torch
import joblib
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import fetch_infra, prepare_series, detect_anomaly, detect_packet_loss
from models import GRUModel, LSTMModel, load_torch_model, load_scaler
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Infrastructure Dashboard", layout="wide")

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="dataRefresh")

# ---------------------------
# Load Models
# ---------------------------
ART = "artifacts"
SCALER_PATH = f"{ART}/scaler_in.pkl"
GRU_PATH = f"{ART}/gru_in.pt"
LSTM_PATH = f"{ART}/lstm_in.pt"
PROPHET_PATH = f"{ART}/prophet_model.pkl"

try:
    scaler = load_scaler(SCALER_PATH)
except:
    scaler = None

# Prophet
try:
    prophet_model = joblib.load(PROPHET_PATH)
except:
    prophet_model = None

# GRU
try:
    gru_model = load_torch_model(GRUModel, GRU_PATH, input_dim=1, hidden_dim=64)
except:
    gru_model = None

# LSTM
try:
    lstm_model = load_torch_model(LSTMModel, LSTM_PATH, input_dim=1, hidden_dim=64)
except:
    lstm_model = None


# ---------------------------
# Helper: RNN prediction loop
# ---------------------------
def rnn_predict(model, scaled_series, seq_len=5):
    if len(scaled_series) < seq_len + 1:
        return None, None, None

    preds_scaled = []
    y_true_scaled = []
    ts_list = []

    for i in range(len(scaled_series) - seq_len):
        seq = scaled_series[i:i+seq_len]
        target = scaled_series[i+seq_len]

        inp = torch.tensor(seq, dtype=torch.float32).reshape(1, seq_len, 1)
        pred = model(inp).item()

        preds_scaled.append(pred)
        y_true_scaled.append(target)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    y_true_scaled = np.array(y_true_scaled).reshape(-1, 1)

    preds = scaler.inverse_transform(preds_scaled).flatten()
    y_true = scaler.inverse_transform(y_true_scaled).flatten()

    return y_true, preds


# ---------------------------
# Helper: Prophet evaluation
# ---------------------------
def evaluate_prophet(df):
    if df.empty or len(df) < 20 or prophet_model is None:
        return None, None, None

    df2 = df.rename(columns={"timestamp": "ds", "bandwidth_in": "y"})
    df2 = df2[["ds", "y"]]

    train_size = int(len(df2) * 0.8)
    train_df = df2[:train_size]
    test_df = df2[train_size:]

    m = prophet_model
    future = m.make_future_dataframe(periods=len(test_df), freq='s')
    forecast = m.predict(future)

    pred = forecast["yhat"].iloc[-len(test_df):].to_numpy()
    true = test_df["y"].to_numpy()
    ts = test_df["ds"].to_numpy()

    return ts, true, pred


# ---------------------------
# Compute metrics (no squared arg)
# ---------------------------
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


# ---------------------------
# UI
# ---------------------------
st.title("🔎 Real-Time Infrastructure Plots")

df = fetch_infra(limit=5000)

if df.empty:
    st.warning("No data found in MongoDB collection. Insert logs then refresh.")
    st.stop()

st.success(f"Loaded {len(df)} records — Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")

# Raw logs
with st.expander("📄 Raw Logs"):
    st.dataframe(df)

# Raw graph
import plotly.express as px

with st.expander("📈 Bandwidth In / Out"):

    # Prepare data for hover
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # ----------- Bandwidth In Plot -----------
    fig_in = px.line(
        df,
        x="timestamp",
        y="bandwidth_in",
        color_discrete_sequence=['#1f77b4'],
        custom_data=["device_id"]
    )

    fig_in.update_traces(
        hovertemplate=
        "<b>Bandwidth In</b><br>" +
        "Device: %{customdata[0]}<br>" +
        "Time: %{x|%Y-%m-%d %H:%M:%S}<br>" +
        "Value: %{y} Mbps<br>"
    )

    st.plotly_chart(fig_in, use_container_width=True)

    # ----------- Bandwidth Out Plot -----------
    fig_out = px.line(
        df,
        x="timestamp",
        y="bandwidth_out",
        color_discrete_sequence=['#ff7f0e'],
        custom_data=["device_id"]
    )

    fig_out.update_traces(
        hovertemplate=
        "<b>Bandwidth Out</b><br>" +
        "Device: %{customdata[0]}<br>" +
        "Time: %{x|%Y-%m-%d %H:%M:%S}<br>" +
        "Value: %{y} Mbps<br>"
    )

    st.plotly_chart(fig_out, use_container_width=True)
# ---------------------------------------
# Time-series preparation
# ---------------------------------------
series = prepare_series(df, "bandwidth_in")
if len(series) < 30:
    st.error("Not enough data for ML models.")
    st.stop()

scaled = scaler.transform(series.reshape(-1,1)).flatten()

split_idx = int(len(series) * 0.8)
ts_test = df["timestamp"].iloc[split_idx:].to_numpy()


# ---------------------------
# MODEL PERFORMANCE TABLE
# ---------------------------
st.subheader("Model Performance ")

# Compute baseline for % calculation (mean + 2*std)
mean_val = df[["bandwidth_in", "bandwidth_out"]].mean().mean()
std_val = df[["bandwidth_in", "bandwidth_out"]].std().mean()
baseline = df[["bandwidth_in", "bandwidth_out"]].max().max() * 1.05  

# Prepare results list as before
results = []

# ----- Prophet -----
ts_p, y_p_true, y_p_pred = evaluate_prophet(df)
if y_p_true is not None:
    mae, rmse = compute_metrics(y_p_true, y_p_pred)
    results.append(["Prophet", mae, rmse])
else:
    results.append(["Prophet", None, None])

# ----- GRU -----
if gru_model is not None:
    y_g_true, y_g_pred = rnn_predict(gru_model, scaled)
    if y_g_true is not None:
        mae, rmse = compute_metrics(y_g_true, y_g_pred)
        results.append(["GRU", mae, rmse])
    else:
        results.append(["GRU", None, None])

# ----- LSTM -----
if lstm_model is not None:
    y_l_true, y_l_pred = rnn_predict(lstm_model, scaled)
    if y_l_true is not None:
        mae, rmse = compute_metrics(y_l_true, y_l_pred)
        results.append(["LSTM", mae, rmse])
    else:
        results.append(["LSTM", None, None])

# Build DataFrame
res_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE"])
res_df.index = res_df.index + 1
res_df.index.name = "S.No"

# Convert MAE and RMSE to "accuracy %" using max bandwidth as baseline
res_df["MAE %"] = ((1 - res_df["MAE"] / baseline) * 100).clip(lower=0).round(2)
res_df["RMSE %"] = ((1 - res_df["RMSE"] / baseline) * 100).clip(lower=0).round(2)

# Reorder columns: Model | MAE | MAE % | RMSE | RMSE %
res_df = res_df[["Model", "MAE", "MAE %", "RMSE", "RMSE %"]]

# Display table
st.table(res_df)

# ---------------------------------------
# Plots
# ---------------------------------------
st.subheader("Separate model plots (actual vs predicted)")

# Helper for proper plotting
def plot_model(ts, y_true, y_pred, title):

    if ts is None or y_true is None:
        st.write(f"{title}: Not enough data.")
        return
    
    # Build the DataFrame
    df_plot = pd.DataFrame({
        "timestamp": ts[:len(y_true)],
        "Actual": y_true[:len(ts)],
        "Predicted": y_pred[:len(ts)]
    })

    # Create plotly figure
    fig = px.line(
        df_plot,
        x="timestamp",
        y=["Actual", "Predicted"],
        labels={"value": "Bandwidth (Mbps)", "timestamp": "Time"},
        color_discrete_sequence=["#1f77b4", "#ff4d4d"]
    )

    # Custom hover
    fig.update_traces(
        hovertemplate=
        "<b>%{fullData.name}</b><br>" +
        "Time: %{x|%Y-%m-%d %H:%M:%S}<br>" +
        "Value: %{y} Mbps<br>"
    )

    fig.update_layout(
        title=title,
        legend_title_text="",
        hovermode="x unified"
    )

    # Show it
    st.plotly_chart(fig, use_container_width=True)

# Prophet
st.markdown("### Prophet")
plot_model(ts_p, y_p_true, y_p_pred, "Prophet")

# GRU
st.markdown("### GRU")
if gru_model is not None:
    plot_model(ts_test, y_g_true, y_g_pred, "GRU")

# LSTM
st.markdown("### LSTM")
if lstm_model is not None:
    plot_model(ts_test, y_l_true, y_l_pred, "LSTM")


# ---------------------------------------
# Alerts
# ---------------------------------------
values = (df["bandwidth_in"] + df["bandwidth_out"]).to_numpy()
timestamps = df["timestamp"].to_numpy()
# ---------------------------------------
# Anomaly Detection
# ---------------------------------------
st.subheader("🚨 Anomaly Detection")

anomaly_alerts = detect_anomaly(values, timestamps)

if anomaly_alerts:
    filtered_anomaly = [a for a in anomaly_alerts if a['severity'] in ("HIGH", "MEDIUM")]

    if filtered_anomaly:
        for a in filtered_anomaly:
            z = a['peak_zscore']
            if z > 6:
                z_html = f"<span style='color:red; font-weight:bold;'>{z:.2f}</span>"
            elif z > 4:
                z_html = f"<span style='color:orange; font-weight:bold;'>{z:.2f}</span>"
            else:
                z_html = f"<span style='font-weight:bold;'>{z:.2f}</span>"

            title = f"{a['severity']} — {a['attack_start']} → {a['attack_end']}"

            with st.expander(title):
                st.markdown(f"""
**Peak Bandwidth:** {a['peak_bandwidth']:.2f} Mbps  
**Z-score:** {z_html}  
                """, unsafe_allow_html=True)

    else:
        st.success("No Anomaly detected.")
else:
    st.success("No Anomaly detected.")

# ---------------------------
# Packet Loss Alerts
# ---------------------------
st.subheader("⚠️ Packet Loss Alerts")
pkt_alerts = detect_packet_loss(df, loss_col="packet_loss", threshold_pct=0.1) 
if pkt_alerts:
    filtered_pkt = [p for p in pkt_alerts if p['severity'] in ("HIGH", "MEDIUM")]

    if filtered_pkt:
        for p in filtered_pkt:
            title = f"{p['severity']} — {p['start_ts']} → {p['end_ts']}"
            with st.expander(title):
                st.write(f"""
**Duration:** {p['duration_seconds']} sec  
**Peak Loss:** {p['peak_loss_pct']} %  
**Recommendation:** {p['recommendation']}  
                """)
    else:
        st.info("No packet loss.")
else:
    st.info("No packet loss events.")
