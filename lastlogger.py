import time
import socket
import threading
from datetime import datetime
import psutil
from scapy.all import sniff, IP, ICMP
from pymongo import MongoClient

# ==================== CONFIG ====================
INTERFACE = "Ethernet"              # change according to your Ubuntu interface
INTERVAL = 1.0                    # seconds

# ✅ MongoDB config (no authentication)
client = MongoClient("mongodb://172.16.24.31:27017/")
db = client["DATABASE_NETWORK"]
collection = db["infrastructure_logs"]

# Map of known IPs to device names
ip_to_name = {
    "172.16.19.174": "window1",
    "172.16.24.28": "ubuntu",
    "172.16.24.29": "windows2"

}

# Thread safety
lock = threading.Lock()
ping_requests = {}
ping_replies = {}

# ==================== FUNCTIONS ====================

def get_iface_stats(iface):
    io = psutil.net_io_counters(pernic=True)
    if iface not in io:
        raise KeyError(f"Interface '{iface}' not found. Available: {list(io.keys())}")
    s = io[iface]
    return s.bytes_sent, s.bytes_recv


def pkt_callback(pkt):
    if IP in pkt and ICMP in pkt:
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        icmp_type = pkt[ICMP].type

        with lock:
            # ICMP echo-request
            if icmp_type == 8:
                ping_requests[src_ip] = ping_requests.get(src_ip, 0) + 1

            # ICMP echo-reply
            elif icmp_type == 0:
                ping_replies[dst_ip] = ping_replies.get(dst_ip, 0) + 1


def resolve_name(ip):
    return ip_to_name.get(ip, ip)


def write_logs_to_mongodb(logs):
    """
    ✅ Inserts multiple logs at once (efficient)
    """
    if logs:
        try:
            collection.insert_many(logs)
        except Exception as e:
            print("❌ MongoDB insert error:", e)


def main():
    print(f"✅ MongoDB logger started on interface '{INTERFACE}'.")
    print("Logs will be stored in MongoDB collection 'infrastructure_logs'.")
    print("Press Ctrl+C to stop.\n")

    sniff_thread = threading.Thread(
        target=lambda: sniff(iface=INTERFACE, filter="icmp", prn=pkt_callback, store=False)
    )
    sniff_thread.daemon = True
    sniff_thread.start()

    prev_sent, prev_recv = get_iface_stats(INTERFACE)

    try:
        while True:
            time.sleep(INTERVAL)

            curr_sent, curr_recv = get_iface_stats(INTERFACE)
            delta_sent = curr_sent - prev_sent
            delta_recv = curr_recv - prev_recv

            bandwidth_out = (delta_sent * 8) / (INTERVAL * 1_000_000)
            bandwidth_in = (delta_recv * 8) / (INTERVAL * 1_000_000)

            prev_sent, prev_recv = curr_sent, curr_recv

            with lock:
                req_snapshot = ping_requests.copy()
                rep_snapshot = ping_replies.copy()
                ping_requests.clear()
                ping_replies.clear()

            logs = []

            for ip, req_count in req_snapshot.items():
                rep_count = rep_snapshot.get(ip, 0)

                packet_loss = ((req_count - rep_count) / req_count) * 100 if req_count > 0 else 0.0

                log = {
                    "timestamp": datetime.now(),           # store as datetime object
                    "device_id": resolve_name(ip),
                    "device_ip": ip,
                    "bandwidth_in": round(bandwidth_in, 3),
                    "bandwidth_out": round(bandwidth_out, 3),
                    "packet_loss": round(packet_loss, 2)
                }

                logs.append(log)
                print(log)

            if logs:
                write_logs_to_mongodb(logs)

    except KeyboardInterrupt:
        print("\n🛑 Logging stopped by user.")

# ==================== RUN ====================
if __name__ == "__main__":
    main()