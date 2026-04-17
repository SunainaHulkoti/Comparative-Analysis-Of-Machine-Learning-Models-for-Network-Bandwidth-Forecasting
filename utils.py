# utils.py

import os
import pandas as pd
from pymongo import MongoClient
import numpy as np


# ---------------------- DB CONNECTION ----------------------
MONGO_URI = "mongodb://localhost:27017"

# Names from Compass
DB_NAME = "DATABASE_NETWORK"
INFRA_COLLECTION = "infrastructure_logs"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
infra_col = db[INFRA_COLLECTION]


# ---------------------- FETCH FUNCTIONS ----------------------
def fetch_infra(limit=None):
    """Fetch infrastructure logs"""
    cursor = infra_col.find().sort("timestamp", -1)
    if limit:
        cursor = cursor.limit(limit)

    data = list(cursor)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df.sort_values("timestamp")


# ---------------------- TIME SERIES PREP ----------------------
def prepare_series(df, col="bandwidth_in"):
    if df.empty or col not in df.columns:
        return np.array([])
    return df[col].astype(float).to_numpy()


# ---------------------- ALERT LOGIC ----------------------
def detect_anomaly(values, timestamps, z_thresh=3.0, rate_thresh=5.0, window=30):
    """
    Professional DDoS detector using hybrid:
    - Z-score anomaly detection
    - Absolute bandwidth threshold (rate_thresh Mbps)
    """
    if len(values) < window:
        return []

    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        "ts": timestamps,
        "bw": values
    })

    df["mean"] = df["bw"].rolling(window).mean()
    df["std"] = df["bw"].rolling(window).std()
    df["z"] = (df["bw"] - df["mean"]) / df["std"]
    df["z"] = df["z"].fillna(0)

    df["ddos_flag"] = (df["z"] >= z_thresh) | (df["bw"] >= rate_thresh)

    alerts = []
    active_attack = None

    for i, row in df.iterrows():
        if row["ddos_flag"]:

            if active_attack is None:
                # Attack START
                active_attack = {
                    "start_ts": row["ts"],
                    "peak_bw": row["bw"],
                    "peak_z": row["z"]
                }
            else:
                # Update peak during ongoing attack
                if row["bw"] > active_attack["peak_bw"]:
                    active_attack["peak_bw"] = row["bw"]
                    active_attack["peak_z"] = row["z"]

        else:
            if active_attack:
                # Attack END
                active_attack["end_ts"] = df.loc[i - 1, "ts"]

                duration = (active_attack["end_ts"] - active_attack["start_ts"]).total_seconds()

                severity = (
                    "CRITICAL" if active_attack["peak_z"] > 6 else
                    "HIGH" if active_attack["peak_z"] > 4 else
                    "MEDIUM"
                )

                alerts.append({
                    "attack_start": active_attack["start_ts"],
                    "attack_end": active_attack["end_ts"],
                    "duration_seconds": duration,
                    "peak_bandwidth": float(active_attack["peak_bw"]),
                    "peak_zscore": float(active_attack["peak_z"]),
                    "severity": severity
                })

                active_attack = None

    return alerts


def detect_packet_loss(df, loss_col="packet_loss", threshold_pct=1.0):
    """
    Professional packet loss analyzer.
    Detects sustained high packet-loss events, not single spikes.
    """
    if df.empty or loss_col not in df.columns:
        return []

    alerts = []
    active = None

    for i, row in df.iterrows():
        loss = float(row[loss_col])

        if loss >= threshold_pct:
            # Start new event
            if active is None:
                active = {
                    "start_ts": row["timestamp"],
                    "peak_loss": loss
                }
            else:
                active["peak_loss"] = max(active["peak_loss"], loss)

        else:
            if active:
                # End event
                end_ts = row["timestamp"]
                duration = (end_ts - active["start_ts"]).total_seconds()
                peak = active["peak_loss"]

                severity = (
                    "CRITICAL" if peak > 20 else
                    "HIGH" if peak > 10 else
                    "MEDIUM" if peak > 5 else
                    "LOW"
                )

                suggestion = (
                    "Severe congestion or link failure — check routing or hardware."
                    if peak > 20 else
                    "Major congestion — increase buffer or reduce traffic burst."
                    if peak > 10 else
                    "Moderate jitter — consider QoS tuning."
                    if peak > 5 else
                    "Minor — monitor the interface."
                )

                alerts.append({
                    "start_ts": active["start_ts"],
                    "end_ts": end_ts,
                    "duration_seconds": duration,
                    "peak_loss_pct": peak,
                    "severity": severity,
                    "recommendation": suggestion
                })

                active = None

    return alerts
# ---------------------- MISC UTILS ----------------------
def ensure_dir(path):
    """Create directory if it doesn't exist."""
    import os
    if not os.path.exists(path):
        os.makedirs(path)
