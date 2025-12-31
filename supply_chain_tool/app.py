import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Supply Chain Optimization Tool", layout="wide")

st.title("Supply Chain / Logistics Optimization Tool (MVP)")
st.caption("Forecasting → Inventory Policy → Simulation → KPIs")

# -----------------------------
# Helpers
# -----------------------------
def moving_average_forecast(series, window=7):
    return series.rolling(window=window, min_periods=1).mean()

def exp_smoothing_forecast(series, alpha=0.3):
    # simple exponential smoothing
    f = []
    prev = series.iloc[0]
    for x in series:
        prev = alpha * x + (1 - alpha) * prev
        f.append(prev)
    return pd.Series(f, index=series.index)

def z_from_service_level(sl):
    # rough Z mapping (good enough for interview demo)
    # common service levels: 0.90, 0.95, 0.97, 0.98, 0.99
    table = {
        0.80: 0.84,
        0.85: 1.04,
        0.90: 1.28,
        0.95: 1.65,
        0.97: 1.88,
        0.98: 2.05,
        0.99: 2.33
    }
    # pick nearest
    keys = np.array(sorted(table.keys()))
    nearest = keys[np.argmin(np.abs(keys - sl))]
    return table[float(nearest)]

def eoq(annual_demand, order_cost, holding_cost_per_unit_per_year):
    if holding_cost_per_unit_per_year <= 0:
        return np.nan
    return np.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit_per_year)

def simulate_inventory(demand, lead_time_days, reorder_point, order_qty, initial_on_hand=0):
    """
    Simple simulation:
    - Daily demand consumes on-hand (no backorders, lost sales)
    - If inventory position <= ROP, place order of Q (arrives after lead time)
    - Inventory position = on_hand + on_order
    """
    on_hand = initial_on_hand
    pipeline = []  # list of (arrival_day_index, qty)
    on_order = 0

    records = []
    stockouts = 0
    lost_units = 0
    total_demand = 0

    for i, d in enumerate(demand):
        # receive arriving orders
        arriving = [q for (arr_i, q) in pipeline if arr_i == i]
        if arriving:
            received = sum(arriving)
            on_hand += received
            on_order -= received
        pipeline = [(arr_i, q) for (arr_i, q) in pipeline if arr_i != i]

        # demand occurs
        total_demand += d
        if d > on_hand:
            stockouts += 1
            lost_units += (d - on_hand)
            on_hand = 0
        else:
            on_hand -= d

        inventory_position = on_hand + on_order

        # reorder logic
        ordered_today = 0
        if inventory_position <= reorder_point:
            ordered_today = order_qty
            arrival = i + lead_time_days
            pipeline.append((arrival, order_qty))
            on_order += order_qty

        records.append({
            "day": i,
            "demand": d,
            "on_hand": on_hand,
            "on_order": on_order,
            "inv_position": inventory_position,
            "ordered_today": ordered_today
        })

    df = pd.DataFrame(records)
    fill_rate_proxy = (total_demand - lost_units) / total_demand if total_demand > 0 else np.nan
    return df, {
        "stockout_days": stockouts,
        "lost_units": lost_units,
        "total_demand": total_demand,
        "fill_rate_proxy": fill_rate_proxy,
        "avg_on_hand": df["on_hand"].mean()
    }

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("1) Data")
uploaded = st.sidebar.file_uploader("Upload demand CSV", type=["csv"])
st.sidebar.caption("CSV must contain a demand column. Optional date column.")

default_days = 90

st.sidebar.header("2) Forecast")
forecast_method = st.sidebar.selectbox("Method", ["Moving Average", "Exponential Smoothing"])
ma_window = st.sidebar.slider("MA Window (days)", 3, 30, 7)
alpha = st.sidebar.slider("Smoothing alpha", 0.05, 0.95, 0.30)

st.sidebar.header("3) Inventory Planning")
lead_time = st.sidebar.slider("Lead time (days)", 1, 60, 10)
service_level = st.sidebar.select_slider("Target service level", options=[0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99], value=0.95)

use_eoq = st.sidebar.checkbox("Use EOQ for order quantity", value=True)
order_cost = st.sidebar.number_input("Order cost ($/order)", min_value=0.0, value=50.0, step=10.0)
holding_cost = st.sidebar.number_input("Holding cost ($/unit/year)", min_value=0.0, value=2.0, step=0.5)
manual_q = st.sidebar.number_input("Manual order quantity (units)", min_value=1, value=200, step=10)

st.sidebar.header("4) Simulation")
initial_on_hand = st.sidebar.number_input("Initial on-hand (units)", min_value=0, value=200, step=10)
sim_horizon = st.sidebar.slider("Simulation horizon (days)", 30, 365, 120)

# -----------------------------
# Load / generate data
# -----------------------------
if uploaded:
    df_raw = pd.read_csv(uploaded)
    cols = [c.lower() for c in df_raw.columns]
    df_raw.columns = cols

    # find demand column
    demand_col = None
    for c in ["demand", "qty", "quantity", "units", "sales"]:
        if c in df_raw.columns:
            demand_col = c
            break
    if demand_col is None:
        st.error("Could not find a demand column. Add a column named demand/qty/quantity/units/sales.")
        st.stop()

    date_col = None
    for c in ["date", "day", "ds"]:
        if c in df_raw.columns:
            date_col = c
            break

    if date_col:
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
        df_raw = df_raw.sort_values(date_col)

    demand_series = df_raw[demand_col].fillna(0).astype(float).reset_index(drop=True)
else:
    # generate sample demand
    np.random.seed(7)
    base = 40
    season = 10 * np.sin(np.linspace(0, 8*np.pi, default_days))
    noise = np.random.normal(0, 8, default_days)
    demand_series = pd.Series(np.maximum(0, base + season + noise).round())
    st.info("No file uploaded — using sample demand data.")

# -----------------------------
# Forecast
# -----------------------------
if forecast_method == "Moving Average":
    forecast = moving_average_forecast(demand_series, window=ma_window)
else:
    forecast = exp_smoothing_forecast(demand_series, alpha=alpha)

# Use last N days to estimate variability
hist = demand_series.tail(min(len(demand_series), 120))
demand_mean = float(hist.mean())
demand_std = float(hist.std(ddof=1)) if len(hist) > 1 else 0.0

# Demand during lead time (assume independent daily demand)
mu_LT = demand_mean * lead_time
sigma_LT = demand_std * np.sqrt(lead_time)

z = z_from_service_level(service_level)
safety_stock = z * sigma_LT
reorder_point = mu_LT + safety_stock

annual_demand = demand_mean * 365
eoq_qty = eoq(annual_demand, order_cost, holding_cost) if use_eoq else np.nan
order_qty = int(round(eoq_qty)) if (use_eoq and not np.isnan(eoq_qty) and eoq_qty > 0) else int(manual_q)

# -----------------------------
# Simulation demand input
# -----------------------------
# For simulation horizon, use forecast as expected demand (rounded)
sim_demand = forecast.tail(sim_horizon).reset_index(drop=True)
if len(sim_demand) < sim_horizon:
    # extend by repeating last forecast value
    last_val = float(sim_demand.iloc[-1]) if len(sim_demand) else demand_mean
    sim_demand = pd.concat([sim_demand, pd.Series([last_val] * (sim_horizon - len(sim_demand)))], ignore_index=True)
sim_demand = sim_demand.clip(lower=0).round().astype(int).tolist()

sim_df, kpis = simulate_inventory(
    demand=sim_demand,
    lead_time_days=int(lead_time),
    reorder_point=float(reorder_point),
    order_qty=int(order_qty),
    initial_on_hand=int(initial_on_hand)
)

# -----------------------------
# Layout: results
# -----------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Demand & Forecast")
    plot_df = pd.DataFrame({
        "demand": demand_series.tail(180).reset_index(drop=True),
        "forecast": forecast.tail(180).reset_index(drop=True),
    })

    fig = plt.figure()
    plt.plot(plot_df["demand"].values, label="Demand")
    plt.plot(plot_df["forecast"].values, label="Forecast")
    plt.legend()
    plt.title("Recent Demand vs Forecast")
    plt.xlabel("Day")
    plt.ylabel("Units")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Simulation (Inventory Over Time)")
    fig2 = plt.figure()
    plt.plot(sim_df["on_hand"].values, label="On-hand")
    plt.plot(sim_df["inv_position"].values, label="Inventory position")
    plt.axhline(reorder_point, linestyle="--", label="Reorder point")
    plt.legend()
    plt.title("Inventory Trajectory")
    plt.xlabel("Day")
    plt.ylabel("Units")
    st.pyplot(fig2, clear_figure=True)

with col2:
    st.subheader("Inventory Policy Outputs")
    m1, m2 = st.columns(2)
    m1.metric("Avg daily demand", f"{demand_mean:.1f}")
    m2.metric("Daily demand stdev", f"{demand_std:.1f}")

    m3, m4 = st.columns(2)
    m3.metric("Lead time (days)", f"{lead_time}")
    m4.metric("Target service level", f"{service_level:.2f}")

    m5, m6 = st.columns(2)
    m5.metric("Safety stock", f"{safety_stock:.0f}")
    m6.metric("Reorder point (ROP)", f"{reorder_point:.0f}")

    m7, m8 = st.columns(2)
    m7.metric("Order quantity (Q)", f"{order_qty}")
    m8.metric("Annual demand (est.)", f"{annual_demand:.0f}")

    st.divider()
    st.subheader("Operational KPIs (Simulation)")
    k1, k2 = st.columns(2)
    k1.metric("Stockout days", f"{kpis['stockout_days']}")
    k2.metric("Fill-rate proxy", f"{kpis['fill_rate_proxy']*100:.1f}%")

    k3, k4 = st.columns(2)
    k3.metric("Lost units", f"{kpis['lost_units']:.0f}")
    k4.metric("Avg on-hand", f"{kpis['avg_on_hand']:.1f}")

    st.caption("Note: This is a simple lost-sales simulation to demonstrate operational tradeoffs quickly.")

st.subheader("Simulation Table (first 50 days)")
st.dataframe(sim_df.head(50), use_container_width=True)

st.caption("MVP assumptions: independent daily demand; fixed lead time; one SKU; single location; lost sales (no backorders).")
