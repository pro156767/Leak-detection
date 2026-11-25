import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import io
import zipfile

# ==========================================
# PAGE CONFIG & THEME-ADAPTIVE STYLING
# ==========================================
st.set_page_config(
    page_title="Pipeline Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: var(--text-color);
        opacity: 0.78;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem;
        color: var(--text-color);
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 4px;
        font-weight: 500;
        color: var(--text-color);
        opacity: 0.65;
    }
    .stTabs [aria-selected="true"] {
        color: var(--text-color) !important;
        opacity: 1 !important;
        font-weight: 600;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-color);
        margin: 1rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

if "feedback_log" not in st.session_state:
    st.session_state["feedback_log"] = []


# ==========================================
# 1. MULTI-SCENARIO DEMO DATA GENERATOR
# ==========================================
@st.cache_data
def generate_demo_data(n_stations=5, n_steps=450, dt_sec=5.0, L=100.0, D_mm=150.0, e_mm=0.045):
    np.random.seed(42)
    freq_ms = max(100, int(round(dt_sec * 1000)))
    timestamps = pd.date_range("2026-11-10 08:00:00", periods=n_steps, freq=f"{freq_ms}ms")
    data = {"timestamp": timestamps}

    D = D_mm / 1000.0
    e = e_mm / 1000.0
    A = np.pi * (D / 2.0) ** 2

    base_Q = 0.035
    base_P = 550000.0
    base_T = 28.0

    Q_inlet = base_Q + np.random.normal(0, 0.00015, n_steps)
    P_inlet = base_P + np.random.normal(0, 320.0, n_steps)
    T_inlet = base_T + np.sin(np.linspace(0, 2 * np.pi, n_steps)) * 0.4 + np.random.normal(0, 0.05, n_steps)

    P_inlet[130:133] += 16000.0
    Q_inlet[130:133] += 0.0022

    P_stations = np.zeros((n_steps, n_stations))
    Q_stations = np.zeros((n_steps, n_stations))
    T_stations = np.zeros((n_steps, n_stations))

    P_stations[:, 0] = P_inlet
    Q_stations[:, 0] = Q_inlet
    T_stations[:, 0] = T_inlet

    n_zones = n_stations - 1
    leak_zone_idx = min(1, n_zones - 1)
    scaling_zone_idx = min(2, n_zones - 1) if n_zones >= 3 else -1
    precursor_spikes = [(150, 152), (195, 197), (230, 232), (255, 257)]

    for t in range(n_steps):
        for z in range(n_zones):
            T_curr = T_stations[t, z] - 0.03 * (z + 1) + np.random.normal(0, 0.02)
            T_stations[t, z + 1] = T_curr

            rho = 1000.0 * (1.0 - ((T_curr - 4.0) ** 2) / (119000.0 + 1365.0 * T_curr - 4.0 * (T_curr ** 2)))
            mu = 2.414e-5 * (10.0 ** (247.8 / (T_curr + 273.15 - 140.0)))

            q_in = Q_stations[t, z]
            leak_q = 0.0
            extra_dp = 0.0

            # Zone 1: Short Noise Spike
            if z == 0 and ((130 <= t <= 132) or (210 <= t <= 211)):
                extra_dp += 4200.0

            # Zone 2: Recurring Transients -> Incipient & Sustained Leak
            if z == leak_zone_idx:
                for (s_start, s_end) in precursor_spikes:
                    if s_start <= t <= s_end:
                        leak_q = base_Q * 0.032
                        extra_dp += 2400.0
                if 280 <= t <= 330:
                    prog = (t - 280) / 50.0
                    leak_q = base_Q * 0.048 * prog
                    extra_dp += 2800.0 * prog
                elif t > 330:
                    leak_q = base_Q * 0.058
                    extra_dp += 3400.0

            # Zone 3: Gradual Pipe Scaling (Slow EP rise, balanced flow)
            if z == scaling_zone_idx and t >= 140:
                extra_dp += 11.5 * (t - 140)

            q_out = q_in - leak_q + np.random.normal(0, 0.00008)
            Q_stations[t, z + 1] = max(q_out, 0.0001)

            q_eff = q_in - 0.65 * leak_q
            v = q_eff / A
            Re = (rho * v * D) / mu
            f = 64.0 / max(Re, 1.0) if Re < 2300 else 0.25 / (np.log10((e / D) / 3.7 + 5.74 / (Re ** 0.9))) ** 2

            dp_pred = f * (L / D) * (0.5 * rho * (v ** 2))
            P_stations[t, z + 1] = P_stations[t, z] - dp_pred - extra_dp + np.random.normal(0, 115.0)

    for s in range(n_stations):
        idx = s + 1
        data[f"P{idx}"] = np.round(P_stations[:, s], 2)
        data[f"Q{idx}"] = np.round(Q_stations[:, s], 6)
        data[f"T{idx}"] = np.round(T_stations[:, s], 2)

    return pd.DataFrame(data)


# ==========================================
# 2. PHYSICS & BEHAVIOR ENGINE
# ==========================================
def water_density(T_c):
    return 1000.0 * (1.0 - ((T_c - 4.0) ** 2) / (119000.0 + 1365.0 * T_c - 4.0 * (T_c ** 2)))


def water_viscosity(T_c):
    return 2.414e-5 * (10.0 ** (247.8 / (T_c + 273.15 - 140.0)))


def rolling_linear_slope(values, window, dt_sec):
    n = len(values)
    slopes = np.zeros(n)
    x = np.arange(window) * dt_sec
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-12:
        return slopes
    for i in range(window - 1, n):
        y = values[i - window + 1: i + 1]
        slopes[i] = np.sum((x - x_mean) * (y - y.mean())) / denom
    return slopes


def compute_physics_zone(df, zone_idx, L, D_mm, e_mm, dt_sec=5.0, drift_window=30):
    s_in, s_out = zone_idx, zone_idx + 1
    P_in, P_out = df[f"P{s_in}"].astype(float), df[f"P{s_out}"].astype(float)
    Q_in, Q_out = df[f"Q{s_in}"].astype(float), df[f"Q{s_out}"].astype(float)
    T_in, T_out = df[f"T{s_in}"].astype(float), df[f"T{s_out}"].astype(float)

    D, e = D_mm / 1000.0, e_mm / 1000.0
    Area = np.pi * (D / 2.0) ** 2

    T_avg = 0.5 * (T_in + T_out)
    rho_in, rho_out, rho_avg = water_density(T_in), water_density(T_out), water_density(T_avg)
    mu_avg = water_viscosity(T_avg)

    mass_flow_diff = (rho_in * Q_in) - (rho_out * Q_out)
    velocity = np.clip(0.5 * (Q_in + Q_out) / Area, 1e-6, None)
    Re = (rho_avg * velocity * D) / mu_avg

    f_lam = 64.0 / np.clip(Re, 1.0, None)
    f_turb = 0.25 / (np.log10((e / D) / 3.7 + 5.74 / (np.clip(Re, 1.0, None) ** 0.9))) ** 2
    f = np.where(Re < 2300.0, f_lam, f_turb)

    dp_measured = P_in - P_out
    dp_predicted = f * (L / D) * (0.5 * rho_avg * (velocity ** 2))
    ep_residual = dp_measured - dp_predicted

    safe_dt = max(dt_sec, 0.1)
    dEP_dt = np.diff(ep_residual.values, prepend=ep_residual.values[0]) / safe_dt

    # Rolling Median removes short transient spikes before computing long-term scaling drift
    med_win = min(15, max(5, len(df) // 15))
    if med_win % 2 == 0:
        med_win += 1
    ep_smooth = ep_residual.rolling(window=med_win, min_periods=1).median().values
    mf_smooth = mass_flow_diff.rolling(window=med_win, min_periods=1).median().values

    ep_long_slope = rolling_linear_slope(ep_smooth, min(drift_window, max(5, len(df) // 4)), safe_dt)

    return pd.DataFrame({
        "timestamp": df["timestamp"],
        f"Z{zone_idx}_dP_meas": dp_measured,
        f"Z{zone_idx}_dP_pred": dp_predicted,
        f"Z{zone_idx}_EP": ep_residual,
        f"Z{zone_idx}_EP_Smooth": ep_smooth,
        f"Z{zone_idx}_MassFlowDiff": mass_flow_diff,
        f"Z{zone_idx}_MF_Smooth": mf_smooth,
        f"Z{zone_idx}_dEP_dt": dEP_dt,
        f"Z{zone_idx}_EP_LongSlope": ep_long_slope
    })


def compute_spc(series, dt_sec=5.0, baseline_ratio=0.25, k_sigma=3.0, min_ooc_duration_sec=25.0):
    n_base = max(15, int(len(series) * baseline_ratio))
    base = series.iloc[:n_base]
    cl = float(base.mean())
    std = max(float(base.std()), 1e-6)

    ucl, lcl = cl + k_sigma * std, cl - k_sigma * std
    u_warn, l_warn = cl + (k_sigma * 0.67) * std, cl - (k_sigma * 0.67) * std

    vals = series.values
    n = len(vals)
    beyond = (vals > ucl) | (vals < lcl)
    warn = (vals > u_warn) | (vals < l_warn)

    duration_sec = np.zeros(n)
    status = ["Normal"] * n

    i = 0
    while i < n:
        if beyond[i]:
            j = i
            while j < n and beyond[j]:
                j += 1
            for k in range(i, j):
                dur = (k - i + 1) * dt_sec
                duration_sec[k] = dur
                status[k] = "OOC" if dur >= min_ooc_duration_sec else "Transient"
            i = j
        else:
            status[i] = "Warning" if warn[i] else "Normal"
            i += 1

    return {
        "CL": cl, "UCL": ucl, "LCL": lcl, "U_WARN": u_warn,
        "duration_sec": duration_sec, "beyond_limit": beyond, "status": status
    }


def classify_behavior(phys_df, zone_idx, ep_spc, mf_spc, dt_sec=5.0, scaling_slope_thresh=0.8, min_scaling_dur_sec=120.0):
    """
    ลำดับการจำแนกพฤติกรรม:
    1. Suspected Leak: ค่าออกนอก Control Limit ต่อเนื่อง (>= OOC Duration) ร่วมกับมีการสูญเสียมวลน้ำ
    2. Transient: ค่าพุ่งออกนอก Control Limit สั้นๆ (< OOC Duration) — แยกออกก่อนเพื่อไม่ให้ปนกับ Scaling
    3. Pipe Scaling: ค่า EP (หลังกรอง Spike ด้วย Median) ไต่ระดับสูงขึ้นช้าๆ ต่อเนื่องนานกว่า min_scaling_dur_sec โดยมวลน้ำยังสมดุล
    """
    n = len(phys_df)
    ep_smooth = phys_df[f"Z{zone_idx}_EP_Smooth"].values
    mf_vals = phys_df[f"Z{zone_idx}_MassFlowDiff"].values
    mf_smooth = phys_df[f"Z{zone_idx}_MF_Smooth"].values
    ep_slopes = phys_df[f"Z{zone_idx}_EP_LongSlope"].values

    ep_st, mf_st = ep_spc["status"], mf_spc["status"]
    mf_uwarn, ep_uwarn, ep_ucl = mf_spc["U_WARN"], ep_spc["U_WARN"], ep_spc["UCL"]

    # Accumulate continuous scaling duration on spike-free smoothed EP
    scaling_dur = np.zeros(n)
    for i in range(n):
        is_scaling_cond = (
            ((ep_smooth[i] > ep_uwarn) and (ep_slopes[i] >= scaling_slope_thresh))
            or (ep_smooth[i] > ep_ucl)
        ) and (mf_smooth[i] <= mf_uwarn)

        if is_scaling_cond:
            scaling_dur[i] = (scaling_dur[i - 1] + dt_sec) if i > 0 else dt_sec
        else:
            scaling_dur[i] = 0.0

    combined, behavior = [], []
    for i in range(n):
        mf_warn = mf_vals[i] > mf_uwarn

        if mf_st[i] == "OOC" or (ep_st[i] == "OOC" and mf_warn):
            combined.append("Suspected Leak")
            behavior.append("Transient / Step (Leak)")
        elif ep_st[i] == "Transient" or mf_st[i] == "Transient":
            combined.append("Transient")
            behavior.append("Precursor Spike" if mf_warn else "Short Noise")
        elif scaling_dur[i] >= min_scaling_dur_sec:
            combined.append("Pipe Scaling")
            behavior.append("Gradual Drift (Scaling)")
        elif ep_st[i] == "Warning" or mf_st[i] == "Warning":
            combined.append("Warning")
            behavior.append("Early Drift")
        else:
            combined.append("Normal")
            behavior.append("Stable")

    return combined, behavior


# ==========================================
# 3. PREDICTIVE, ML & FEEDBACK ENGINE
# ==========================================
def run_predictive(phys_df, n_zones, spc_results, dt_sec, descaling_limit=4500.0):
    summary = {}
    n_samples = len(phys_df)
    total_hr = max((n_samples * dt_sec) / 3600.0, 1e-4)

    for z in range(1, n_zones + 1):
        # A. Next Leak Forecast from Transient Frequency
        mf_b = spc_results[z]["mf"]["beyond_limit"]
        ep_b = spc_results[z]["ep"]["beyond_limit"]
        t_mask = mf_b | (ep_b & (phys_df[f"Z{z}_MassFlowDiff"].values > spc_results[z]["mf"]["U_WARN"]))

        starts, in_ev = [], False
        for i in range(n_samples):
            if t_mask[i] and not in_ev:
                starts.append(i)
                in_ev = True
            elif not t_mask[i]:
                in_ev = False

        n_ev = len(starts)
        avg_interval_min = None
        next_leak_min = None

        if n_ev >= 3:
            ia_sec = np.diff(starts) * dt_sec
            avg_interval_min = float(np.mean(ia_sec)) / 60.0
            reg = LinearRegression().fit(np.arange(len(ia_sec)).reshape(-1, 1), ia_sec)
            next_leak_min = max(dt_sec, float(reg.predict([[len(ia_sec)]])[0])) / 60.0
            leak_risk = "High (Accelerating)" if reg.coef_[0] < 0 else "Moderate"
        elif n_ev == 2:
            avg_interval_min = float((starts[1] - starts[0]) * dt_sec) / 60.0
            next_leak_min = avg_interval_min
            leak_risk = "Moderate"
        else:
            leak_risk = "Low"

        # B. Descaling Forecast from Spike-Free Smoothed EP
        w = min(120, n_samples)
        y_ep = phys_df[f"Z{z}_EP_Smooth"].iloc[-w:].values
        y_mf = phys_df[f"Z{z}_MF_Smooth"].iloc[-w:].values
        t_sec = np.arange(w) * dt_sec

        reg_s = LinearRegression().fit(t_sec.reshape(-1, 1), y_ep)
        slope_s = float(reg_s.coef_[0])
        curr_ep = float(y_ep[-1])
        balanced = float(np.mean(np.abs(y_mf))) <= abs(spc_results[z]["mf"]["U_WARN"])

        if slope_s > 0.15 and balanced and curr_ep > spc_results[z]["ep"]["U_WARN"]:
            if curr_ep >= descaling_limit:
                descaling_min = 0.0
                scaling_risk = "Critical (Clean Now)"
            else:
                descaling_min = ((descaling_limit - curr_ep) / slope_s) / 60.0
                scaling_risk = "Active Buildup"
        elif slope_s > 0.15 and not balanced:
            descaling_min = None
            scaling_risk = "Masked by Leak"
        else:
            descaling_min = None
            scaling_risk = "Normal"

        summary[z] = {
            "transients": n_ev,
            "freq_hr": n_ev / total_hr,
            "avg_interval_min": avg_interval_min,
            "next_leak_min": next_leak_min,
            "leak_risk": leak_risk,
            "scale_slope_hr": slope_s * 3600.0,
            "descaling_min": descaling_min,
            "scaling_risk": scaling_risk
        }
    return summary


def run_feedback_ml(phys_df, n_zones, feedback_log, baseline_ratio=0.25):
    if not feedback_log:
        return None, 0

    fb_df = pd.DataFrame(feedback_log)
    feats = ["EP", "MassFlowDiff", "dP_meas", "dEP_dt", "EP_LongSlope"]

    n_base = max(15, int(len(phys_df) * baseline_ratio))
    stride = max(1, n_base // 15)
    normal_rows = [{
        "EP": float(phys_df["Z1_EP"].iloc[i]),
        "MassFlowDiff": float(phys_df["Z1_MassFlowDiff"].iloc[i]),
        "dP_meas": float(phys_df["Z1_dP_meas"].iloc[i]),
        "dEP_dt": float(phys_df["Z1_dEP_dt"].iloc[i]),
        "EP_LongSlope": float(phys_df["Z1_EP_LongSlope"].iloc[i]),
        "Label": "False Alarm"
    } for i in range(0, n_base, stride)]

    train_df = pd.concat([pd.DataFrame(normal_rows), fb_df[feats + ["Label"]]], ignore_index=True)
    if train_df["Label"].nunique() < 2:
        return None, len(fb_df)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_df[feats])
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X_tr, train_df["Label"])

    preds = {}
    for z in range(1, n_zones + 1):
        X_z = pd.DataFrame({
            "EP": phys_df[f"Z{z}_EP"].values,
            "MassFlowDiff": phys_df[f"Z{z}_MassFlowDiff"].values,
            "dP_meas": phys_df[f"Z{z}_dP_meas"].values,
            "dEP_dt": phys_df[f"Z{z}_dEP_dt"].values,
            "EP_LongSlope": phys_df[f"Z{z}_EP_LongSlope"].values
        })
        X_zs = scaler.transform(X_z)
        preds[z] = {
            "pred": clf.predict(X_zs),
            "conf": np.round(np.max(clf.predict_proba(X_zs), axis=1) * 100.0, 1)
        }
    return preds, len(fb_df)


def run_iso_forest(phys_df, n_zones, baseline_ratio=0.25, contamination=0.08):
    cols = []
    for z in range(1, n_zones + 1):
        cols.extend([f"Z{z}_EP", f"Z{z}_MassFlowDiff", f"Z{z}_EP_LongSlope"])
    X = phys_df[cols].fillna(0.0)
    n_tr = max(20, int(len(X) * baseline_ratio))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X.iloc[:n_tr])
    X_all = scaler.transform(X)

    iso = IsolationForest(n_estimators=120, contamination=contamination, random_state=42)
    iso.fit(X_tr)
    scores = -iso.decision_function(X_all)
    s_min, s_max = scores.min(), scores.max()
    intensity = np.clip((scores - s_min) / max(s_max - s_min, 1e-9) * 100.0, 0.0, 100.0)
    return np.round(intensity, 1)


def build_zip(sample_df):
    buf = io.BytesIO()
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception:
        code = "# app.py"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("app.py", code)
        zf.writestr("requirements.txt", "streamlit>=1.35.0\npandas>=2.0.0\nnumpy>=1.24.0\nplotly>=5.18.0\nscikit-learn>=1.3.0\n")
        zf.writestr("sample_sensor_data.csv", sample_df.to_csv(index=False).encode("utf-8"))
    buf.seek(0)
    return buf


# ==========================================
# 4. MINIMAL SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### Pipeline Monitor")
    data_source = st.radio("Data Source", ["Demo Dataset", "Upload CSV"], label_visibility="collapsed")

    if data_source == "Demo Dataset":
        n_stations_demo = st.slider("Stations", 3, 8, 5)
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Timing & Thresholds**")
    sampling_rate = st.number_input("Sampling Rate (sec)", min_value=0.5, max_value=3600.0, value=5.0, step=0.5)
    ooc_duration = st.number_input("Out-of-Control Duration (sec)", min_value=1.0, max_value=3600.0, value=25.0, step=5.0)

    with st.expander("Pipe & Model Settings"):
        param_L = st.number_input("Spacing L (m)", 1.0, 5000.0, 100.0, 10.0)
        param_D = st.number_input("Diameter D (mm)", 10.0, 2000.0, 150.0, 5.0)
        param_e = st.number_input("Roughness e (mm)", 0.001, 10.0, 0.045, 0.005, format="%.4f")
        baseline_pct = st.slider("Baseline (%)", 15, 50, 25, 5) / 100.0
        spc_sigma = st.slider("SPC Sigma", 2.0, 4.5, 3.0, 0.25)
        scaling_slope_thresh = st.number_input("Scaling Slope (Pa/s)", 0.05, 50.0, 0.8, 0.1)
        scaling_min_dur = st.number_input("Scaling Duration (sec)", 30.0, 7200.0, 120.0, 30.0)
        descaling_limit = st.number_input("Descaling Limit EP (Pa)", 500.0, 50000.0, 4500.0, 250.0)
        ml_contam = st.slider("ML Contamination", 0.01, 0.25, 0.08, 0.01)

    st.markdown("---")
    st.download_button(
        "Download Sample CSV",
        data=generate_demo_data(5, dt_sec=sampling_rate, L=param_L, D_mm=param_D, e_mm=param_e).to_csv(index=False).encode("utf-8"),
        file_name="sample_pipeline_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.download_button(
        "Export Project ZIP",
        data=build_zip(generate_demo_data(5, dt_sec=sampling_rate)),
        file_name="pipeline_monitor.zip",
        mime="application/zip",
        use_container_width=True
    )

# Load Data
if data_source == "Demo Dataset":
    raw_df = generate_demo_data(n_stations=n_stations_demo, dt_sec=sampling_rate, L=param_L, D_mm=param_D, e_mm=param_e)
else:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        if "timestamp" in raw_df.columns:
            raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
        else:
            freq_ms = max(100, int(round(sampling_rate * 1000)))
            raw_df["timestamp"] = pd.date_range("2026-11-10 08:00:00", periods=len(raw_df), freq=f"{freq_ms}ms")
    else:
        raw_df = generate_demo_data(n_stations=5, dt_sec=sampling_rate, L=param_L, D_mm=param_D, e_mm=param_e)

p_cols = [c for c in raw_df.columns if c.startswith("P") and c[1:].isdigit()]
n_stations = len(p_cols)
n_zones = n_stations - 1
if n_zones < 1:
    st.error("CSV must contain at least 2 stations (P1, P2, Q1, Q2, T1, T2).")
    st.stop()


# ==========================================
# 5. RUN CALCULATIONS
# ==========================================
phys_dfs = []
for z in range(1, n_zones + 1):
    z_df = compute_physics_zone(raw_df, z, param_L, param_D, param_e, dt_sec=sampling_rate)
    if z > 1:
        z_df = z_df.drop(columns=["timestamp"])
    phys_dfs.append(z_df)
phys_df = pd.concat(phys_dfs, axis=1)

spc_results = {}
for z in range(1, n_zones + 1):
    ep_spc = compute_spc(phys_df[f"Z{z}_EP"], dt_sec=sampling_rate, baseline_ratio=baseline_pct, k_sigma=spc_sigma, min_ooc_duration_sec=ooc_duration)
    mf_spc = compute_spc(phys_df[f"Z{z}_MassFlowDiff"], dt_sec=sampling_rate, baseline_ratio=baseline_pct, k_sigma=spc_sigma, min_ooc_duration_sec=ooc_duration)
    comb, behav = classify_behavior(
        phys_df, z, ep_spc, mf_spc,
        dt_sec=sampling_rate,
        scaling_slope_thresh=scaling_slope_thresh,
        min_scaling_dur_sec=scaling_min_dur
    )
    spc_results[z] = {"ep": ep_spc, "mf": mf_spc, "status": comb, "behavior": behav}
    phys_df[f"Z{z}_Status"] = comb
    phys_df[f"Z{z}_Behavior"] = behav

pred_summary = run_predictive(phys_df, n_zones, spc_results, dt_sec=sampling_rate, descaling_limit=descaling_limit)
ml_anomaly_score = run_iso_forest(phys_df, n_zones, baseline_ratio=baseline_pct, contamination=ml_contam)
fb_preds, fb_count = run_feedback_ml(phys_df, n_zones, st.session_state["feedback_log"], baseline_ratio=baseline_pct)


# ==========================================
# 6. HEADER & 3 CLEAN TABS
# ==========================================
head_l, head_r = st.columns([3, 1])
with head_l:
    st.markdown("## Multi-Zone Pipeline Monitor")
with head_r:
    if fb_preds is not None:
        st.caption(f"ML Model: Retrained ({fb_count} feedbacks)")
    else:
        st.caption(f"ML Model: Unsupervised ({fb_count} feedbacks)")

tab_overview, tab_monitor, tab_predict = st.tabs(["Overview", "Zone Monitor & Feedback", "Predictive & Logs"])

STATUS_COLORS = {
    "Normal": "#10B981",
    "Warning": "#F59E0B",
    "Transient": "#F97316",
    "Pipe Scaling": "#A855F7",
    "Suspected Leak": "#EF4444"
}


# ------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------
with tab_overview:
    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        step_idx = st.slider("Inspect Time Step", 0, len(phys_df) - 1, len(phys_df) - 1, label_visibility="collapsed")
    with ctrl_col2:
        st.caption(f"Time: `{phys_df['timestamp'].iloc[step_idx]}` (Step {step_idx})")

    leaks = sum(1 for z in range(1, n_zones + 1) if spc_results[z]["status"][step_idx] == "Suspected Leak")
    scalings = sum(1 for z in range(1, n_zones + 1) if spc_results[z]["status"][step_idx] == "Pipe Scaling")
    transients = sum(1 for z in range(1, n_zones + 1) if spc_results[z]["status"][step_idx] == "Transient")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Suspected Leaks", f"{leaks} / {n_zones}")
    c2.metric("Pipe Scaling Zones", f"{scalings} / {n_zones}")
    c3.metric("Active Transients", f"{transients}")
    c4.metric("ML Anomaly Score", f"{ml_anomaly_score[step_idx]:.0f} / 100")

    st.markdown('<div class="section-title">Pipeline Status Map</div>', unsafe_allow_html=True)
    fig_map = go.Figure()
    for z in range(1, n_zones + 1):
        st_z = spc_results[z]["status"][step_idx]
        col_z = STATUS_COLORS.get(st_z, "#94A3B8")
        ep_v = phys_df[f"Z{z}_EP"].iloc[step_idx]
        mf_v = phys_df[f"Z{z}_MassFlowDiff"].iloc[step_idx]

        fig_map.add_trace(go.Scatter(
            x=[z, z + 1], y=[1, 1], mode="lines",
            line=dict(color=col_z, width=14),
            hoverinfo="text",
            text=f"Zone {z}: {st_z}<br>EP: {ep_v:.0f} Pa<br>Flow Diff: {mf_v:.4f} kg/s"
        ))
        fig_map.add_annotation(
            x=z + 0.5, y=1.18, showarrow=False,
            text=f"<b>Zone {z}</b><br><span style='font-size:11px;color:{col_z};'>{st_z}</span>"
        )

    fig_map.add_trace(go.Scatter(
        x=list(range(1, n_stations + 1)), y=[1] * n_stations,
        mode="markers+text",
        marker=dict(size=26, color="#1E293B", line=dict(width=2, color="#94A3B8")),
        text=[f"S{s}" for s in range(1, n_stations + 1)],
        textfont=dict(color="white", size=11),
        hoverinfo="skip"
    ))
    fig_map.update_layout(
        height=170, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[0.6, n_stations + 0.4]),
        yaxis=dict(visible=False, range=[0.75, 1.38]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown('<div class="section-title">Zone Diagnostics</div>', unsafe_allow_html=True)
    rows = []
    for z in range(1, n_zones + 1):
        p = pred_summary[z]
        r = {
            "Zone": f"Zone {z} (S{z}-S{z+1})",
            "Status": spc_results[z]["status"][step_idx],
            "Pattern": spc_results[z]["behavior"][step_idx],
            "EP (Pa)": round(phys_df[f"Z{z}_EP"].iloc[step_idx], 1),
            "Flow Diff (kg/s)": round(phys_df[f"Z{z}_MassFlowDiff"].iloc[step_idx], 4),
            "Scaling Rate (Pa/hr)": round(p["scale_slope_hr"], 1),
            "Transients": p["transients"]
        }
        if fb_preds is not None:
            r["Feedback ML"] = f"{fb_preds[z]['pred'][step_idx]} ({fb_preds[z]['conf'][step_idx]}%)"
        rows.append(r)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ------------------------------------------
# TAB 2: ZONE MONITOR & INTERACTIVE FEEDBACK
# ------------------------------------------
with tab_monitor:
    sel_col1, sel_col2 = st.columns([2, 3])
    with sel_col1:
        z = st.selectbox(
            "Select Zone",
            options=list(range(1, n_zones + 1)),
            format_func=lambda x: f"Zone {x} — {spc_results[x]['status'][-1]}"
        )
    with sel_col2:
        st.caption("คลิกที่จุดสีผิดปกติบนกราฟด้านล่างเพื่อส่งผลตรวจสอบจริงให้โมเดล ML เรียนรู้")

    z_status = np.array(spc_results[z]["status"])
    ep_info, mf_info = spc_results[z]["ep"], spc_results[z]["mf"]

    def make_chart(y_vals, spc_d, title, y_unit, line_col):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=phys_df["timestamp"], y=y_vals, mode="lines",
            name="Value", line=dict(color=line_col, width=1.8)
        ))
        fig.add_hline(y=spc_d["UCL"], line_dash="dot", line_color="#EF4444", line_width=1.2)
        fig.add_hline(y=spc_d["CL"], line_dash="solid", line_color="#94A3B8", line_width=1)
        fig.add_hline(y=spc_d["LCL"], line_dash="dot", line_color="#EF4444", line_width=1.2)

        for label, sym, sz in [("Suspected Leak", "x", 9), ("Pipe Scaling", "square", 7), ("Transient", "diamond", 8)]:
            m = (z_status == label)
            if np.any(m):
                fig.add_trace(go.Scatter(
                    x=phys_df["timestamp"][m], y=y_vals[m], mode="markers",
                    name=label, marker=dict(color=STATUS_COLORS[label], symbol=sym, size=sz)
                ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            yaxis_title=y_unit,
            height=300,
            hovermode="x unified",
            clickmode="event+select",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=35, b=10),
            xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.18)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.18)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11))
        )
        return fig

    ch1, ch2 = st.columns(2)
    with ch1:
        ep_sel = st.plotly_chart(
            make_chart(phys_df[f"Z{z}_EP"], ep_info, f"Zone {z} — Pressure Residual (EP)", "Pa", "#818CF8"),
            use_container_width=True, on_select="rerun", selection_mode="points", key=f"ep_z{z}"
        )
    with ch2:
        mf_sel = st.plotly_chart(
            make_chart(phys_df[f"Z{z}_MassFlowDiff"], mf_info, f"Zone {z} — Mass Flow Difference", "kg/s", "#38BDF8"),
            use_container_width=True, on_select="rerun", selection_mode="points", key=f"mf_z{z}"
        )

    clicked_idx = None
    for sel in [ep_sel, mf_sel]:
        if sel and "selection" in sel and sel["selection"].get("points"):
            pt = sel["selection"]["points"][0]
            if "x" in pt:
                ts_click = pd.to_datetime(pt["x"])
                clicked_idx = int(np.argmin(np.abs(pd.to_datetime(phys_df["timestamp"]) - ts_click)))
                break

    anom_idx = np.where(z_status != "Normal")[0]
    with st.container(border=True):
        fb_h1, fb_h2 = st.columns([2, 2])
        with fb_h1:
            st.markdown("**Technician Verification (Feedback Loop)**")
        with fb_h2:
            if clicked_idx is None and len(anom_idx) > 0:
                clicked_idx = st.selectbox(
                    "Or pick anomaly point",
                    options=anom_idx.tolist(),
                    format_func=lambda idx: f"Step {idx} | {z_status[idx]} | EP {phys_df[f'Z{z}_EP'].iloc[idx]:.0f} Pa",
                    label_visibility="collapsed"
                )

        if clicked_idx is not None:
            fb_ts = phys_df["timestamp"].iloc[clicked_idx]
            fb_ep = float(phys_df[f"Z{z}_EP"].iloc[clicked_idx])
            fb_mf = float(phys_df[f"Z{z}_MassFlowDiff"].iloc[clicked_idx])
            fb_dp = float(phys_df[f"Z{z}_dP_meas"].iloc[clicked_idx])
            fb_dep = float(phys_df[f"Z{z}_dEP_dt"].iloc[clicked_idx])
            fb_slp = float(phys_df[f"Z{z}_EP_LongSlope"].iloc[clicked_idx])

            st.caption(f"Selected: Zone {z} | Step {clicked_idx} ({fb_ts}) | System: **{z_status[clicked_idx]}** | EP: {fb_ep:.0f} Pa | Δm: {fb_mf:.4f} kg/s")

            b1, b2, b3 = st.columns(3)
            def save_fb(label_name):
                st.session_state["feedback_log"].append({
                    "Timestamp": str(fb_ts), "Zone": f"Zone {z}", "Step": clicked_idx,
                    "System": z_status[clicked_idx], "Label": label_name,
                    "EP": fb_ep, "MassFlowDiff": fb_mf, "dP_meas": fb_dp,
                    "dEP_dt": fb_dep, "EP_LongSlope": fb_slp
                })
                st.rerun()

            if b1.button("Confirm: Leak", use_container_width=True, type="primary"):
                save_fb("Confirmed Leak")
            if b2.button("Confirm: Scaling", use_container_width=True):
                save_fb("Confirmed Scaling")
            if b3.button("False Alarm (Normal)", use_container_width=True):
                save_fb("False Alarm")
        else:
            st.caption("No anomalies in this zone. Click any point on the chart above to label manually.")


# ------------------------------------------
# TAB 3: PREDICTIVE MAINTENANCE & LOGS
# ------------------------------------------
with tab_predict:
    st.markdown('<div class="section-title">Predictive Maintenance Forecast</div>', unsafe_allow_html=True)
    p_rows = []
    for z in range(1, n_zones + 1):
        p = pred_summary[z]
        p_rows.append({
            "Zone": f"Zone {z}",
            "Transients": p["transients"],
            "Avg Interval (min)": f"{p['avg_interval_min']:.1f}" if p["avg_interval_min"] is not None else "-",
            "Est. Next Leak (min)": f"{p['next_leak_min']:.1f}" if p["next_leak_min"] is not None else "-",
            "Leak Risk": p["leak_risk"],
            "Scaling Rate (Pa/hr)": f"{p['scale_slope_hr']:.1f}",
            "Est. Descaling (min)": "Now" if p["descaling_min"] == 0.0 else (f"{p['descaling_min']:.1f}" if p["descaling_min"] is not None else "-"),
            "Scaling Status": p["scaling_risk"]
        })
    st.dataframe(pd.DataFrame(p_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">ML Anomaly Intensity (Multi-Zone)</div>', unsafe_allow_html=True)
    fig_ml = go.Figure()
    fig_ml.add_trace(go.Scatter(
        x=phys_df["timestamp"], y=ml_anomaly_score, mode="lines",
        line=dict(color="#38BDF8", width=1.8), fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.12)"
    ))
    fig_ml.update_layout(
        height=220, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Score (0-100)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.18)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    )
    st.plotly_chart(fig_ml, use_container_width=True)

    st.markdown('<div class="section-title">Technician Feedback Log</div>', unsafe_allow_html=True)
    l_col, r_col = st.columns([3, 1])
    with l_col:
        if st.session_state["feedback_log"]:
            fb_table = pd.DataFrame(st.session_state["feedback_log"])
            st.dataframe(fb_table[["Timestamp", "Zone", "Step", "System", "Label", "EP", "MassFlowDiff"]], use_container_width=True, hide_index=True)
        else:
            st.caption("No feedback recorded yet. Go to 'Zone Monitor & Feedback' and click a point on the chart.")
    with r_col:
        if st.session_state["feedback_log"]:
            st.download_button(
                "Export Feedback CSV",
                data=pd.DataFrame(st.session_state["feedback_log"]).to_csv(index=False).encode("utf-8"),
                file_name="feedback_log.csv", mime="text/csv", use_container_width=True
            )
            if st.button("Clear Feedback", use_container_width=True):
                st.session_state["feedback_log"] = []
                st.rerun()
        up_fb = st.file_uploader("Import Feedback CSV", type=["csv"])
        if up_fb is not None:
            imp = pd.read_csv(up_fb)
            if {"Timestamp", "Zone", "Label", "EP", "MassFlowDiff", "dP_meas", "dEP_dt", "EP_LongSlope"}.issubset(imp.columns):
                st.session_state["feedback_log"] = imp.to_dict("records")
                st.rerun()
