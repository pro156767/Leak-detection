import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import io
import zipfile

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Multi-Zone Pipeline Leak Detection System",
    page_icon="🚰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #0d6efd;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .status-normal { border-left-color: #198754 !important; }
    .status-warning { border-left-color: #ffc107 !important; }
    .status-transient { border-left-color: #fd7e14 !important; }
    .status-leak { border-left-color: #dc3545 !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DEMO DATA GENERATOR
# ==========================================
@st.cache_data
def generate_demo_data(n_stations=5, n_steps=300, dt_sec=5, L=100.0, D_mm=150.0, e_mm=0.045):
    """
    สร้างข้อมูลจำลองสำหรับ Series Pipeline ที่มี n_stations (สร้าง n_stations - 1 โซน)
    - Step 0 - 119: สภาวะปกติ (Baseline)
    - Step 120 - 124: Transient Spike (ค้อนน้ำ / ปั๊มกระชากสั้นๆ 25 วินาที)
    - Step 160 - 210: รอยรั่วขนาดเล็กที่ค่อยๆ โตขึ้น (Incipient Leak) ใน Zone 2
    - Step 211 - 299: รอยรั่วชัดเจน (Sustained Leak) ใน Zone 2
    """
    np.random.seed(42)
    timestamps = pd.date_range("2026-11-10 08:00:00", periods=n_steps, freq=f"{dt_sec}s")
    data = {"timestamp": timestamps}

    D = D_mm / 1000.0
    e = e_mm / 1000.0
    A = np.pi * (D / 2.0) ** 2

    base_Q = 0.035      # m^3/s (~35 L/s)
    base_P = 550000.0   # Pa (5.5 bar)
    base_T = 28.0       # Celsius

    Q_inlet = base_Q + np.random.normal(0, 0.00015, n_steps)
    P_inlet = base_P + np.random.normal(0, 350.0, n_steps)
    T_inlet = base_T + np.sin(np.linspace(0, 2 * np.pi, n_steps)) * 0.4 + np.random.normal(0, 0.05, n_steps)

    # Inject Transient Anomaly (Step 120 to 124 -> 5 steps = 25 seconds)
    P_inlet[120:125] += 18000.0
    Q_inlet[120:125] += 0.0025

    P_stations = np.zeros((n_steps, n_stations))
    Q_stations = np.zeros((n_steps, n_stations))
    T_stations = np.zeros((n_steps, n_stations))

    P_stations[:, 0] = P_inlet
    Q_stations[:, 0] = Q_inlet
    T_stations[:, 0] = T_inlet

    leak_zone_idx = min(1, n_stations - 2)  # Default to Zone 2 (index 1)

    for t in range(n_steps):
        for z in range(n_stations - 1):
            T_curr = T_stations[t, z] - 0.03 * (z + 1) + np.random.normal(0, 0.02)
            T_stations[t, z + 1] = T_curr

            rho = 1000.0 * (1.0 - ((T_curr - 4.0) ** 2) / (119000.0 + 1365.0 * T_curr - 4.0 * (T_curr ** 2)))
            mu = 2.414e-5 * (10.0 ** (247.8 / (T_curr + 273.15 - 140.0)))

            q_in = Q_stations[t, z]
            leak_q = 0.0
            transient_dp_spike = 0.0

            # Transient pressure distortion in Zone 1 during water hammer
            if 120 <= t < 125 and z == 0:
                transient_dp_spike = 4500.0

            # Inject Leak in target zone
            if z == leak_zone_idx:
                if 160 <= t <= 210:
                    leak_q = base_Q * 0.045 * ((t - 160) / 50.0)
                elif t > 210:
                    leak_q = base_Q * 0.055

            q_out = q_in - leak_q + np.random.normal(0, 0.00008)
            Q_stations[t, z + 1] = max(q_out, 0.0001)

            q_eff = q_in - 0.65 * leak_q
            v = q_eff / A
            Re = (rho * v * D) / mu
            if Re < 2300:
                f = 64.0 / max(Re, 1.0)
            else:
                f = 0.25 / (np.log10((e / D) / 3.7 + 5.74 / (Re ** 0.9))) ** 2

            dp_pred = f * (L / D) * (0.5 * rho * (v ** 2))
            leak_dp_distortion = 3200.0 * (leak_q / (base_Q * 0.055)) if leak_q > 0 else 0.0

            P_stations[t, z + 1] = (
                P_stations[t, z]
                - dp_pred
                - leak_dp_distortion
                - transient_dp_spike
                + np.random.normal(0, 120.0)
            )

    for s in range(n_stations):
        idx = s + 1
        data[f"P{idx}"] = np.round(P_stations[:, s], 2)
        data[f"Q{idx}"] = np.round(Q_stations[:, s], 6)
        data[f"T{idx}"] = np.round(T_stations[:, s], 2)

    return pd.DataFrame(data)


# ==========================================
# 2. PHYSICS-BASED MODEL ENGINE
# ==========================================
def water_density(T_celsius):
    """คำนวณความหนาแน่นของน้ำ (kg/m^3) จากอุณหภูมิ (°C)"""
    return 1000.0 * (1.0 - ((T_celsius - 4.0) ** 2) / (119000.0 + 1365.0 * T_celsius - 4.0 * (T_celsius ** 2)))

def water_viscosity(T_celsius):
    """คำนวณความหนืดไดนามิกของน้ำ (Pa·s) จากอุณหภูมิ (°C)"""
    return 2.414e-5 * (10.0 ** (247.8 / (T_celsius + 273.15 - 140.0)))

def compute_physics_zone(df, zone_idx, L, D_mm, e_mm):
    """
    คำนวณพารามิเตอร์ทางฟิสิกส์สำหรับ Zone ที่ระบุ (เชื่อมต่อระหว่าง Station i และ i+1)
    - เก็บ dP_meas และ dP_pred ไว้สำหรับ Monitoring & Diagnosis
    - คำนวณ EP (Pressure Residual = dP_meas - dP_pred) สำหรับใช้เป็นตัวหลักใน Control Chart
    """
    s_in = zone_idx
    s_out = zone_idx + 1

    P_in = df[f"P{s_in}"].astype(float)
    P_out = df[f"P{s_out}"].astype(float)
    Q_in = df[f"Q{s_in}"].astype(float)
    Q_out = df[f"Q{s_out}"].astype(float)
    T_in = df[f"T{s_in}"].astype(float)
    T_out = df[f"T{s_out}"].astype(float)

    D = D_mm / 1000.0
    e = e_mm / 1000.0
    Area = np.pi * (D / 2.0) ** 2

    T_avg = 0.5 * (T_in + T_out)
    rho_in = water_density(T_in)
    rho_out = water_density(T_out)
    rho_avg = water_density(T_avg)
    mu_avg = water_viscosity(T_avg)

    # Mass Flow Difference (kg/s)
    m_dot_in = rho_in * Q_in
    m_dot_out = rho_out * Q_out
    mass_flow_diff = m_dot_in - m_dot_out

    # Mean Velocity & Reynolds Number
    Q_avg = 0.5 * (Q_in + Q_out)
    velocity = np.clip(Q_avg / Area, 1e-6, None)
    Re = (rho_avg * velocity * D) / mu_avg

    # Friction Factor (Laminar vs Swamee-Jain Turbulent)
    f_laminar = 64.0 / np.clip(Re, 1.0, None)
    f_turbulent = 0.25 / (np.log10((e / D) / 3.7 + 5.74 / (np.clip(Re, 1.0, None) ** 0.9))) ** 2
    f = np.where(Re < 2300.0, f_laminar, f_turbulent)

    # Pressure Drop (Monitoring/Diagnosis) & Pressure Residual EP (Primary SPC Variable)
    dp_measured = P_in - P_out
    dp_predicted = f * (L / D) * (0.5 * rho_avg * (velocity ** 2))
    ep_residual = dp_measured - dp_predicted

    res = pd.DataFrame({
        "timestamp": df["timestamp"],
        f"Z{zone_idx}_T_avg": T_avg,
        f"Z{zone_idx}_rho": rho_avg,
        f"Z{zone_idx}_mu": mu_avg,
        f"Z{zone_idx}_Velocity": velocity,
        f"Z{zone_idx}_Re": Re,
        f"Z{zone_idx}_f": f,
        f"Z{zone_idx}_dP_meas": dp_measured,
        f"Z{zone_idx}_dP_pred": dp_predicted,
        f"Z{zone_idx}_EP": ep_residual,
        f"Z{zone_idx}_MassFlowDiff": mass_flow_diff
    })
    return res


# ==========================================
# 3. SPC & TIME-BASED DURATION ENGINE
# ==========================================
def compute_spc_and_duration(series, timestamps, baseline_ratio=0.35, k_sigma=3.0, min_leak_duration_sec=30.0, trend_window=7):
    """
    คำนวณ UCL, CL, LCL, วิเคราะห์ Trend และจำแนกสถานะด้วย Time-based Duration Analysis:
    - Normal
    - Warning (เกิน 2 Sigma หรือเกิด Trend ต่อเนื่อง)
    - Transient Anomaly (เกิน UCL/LCL แต่ระยะเวลาสั้นกว่า min_leak_duration_sec)
    - Suspected Leak (เกิน UCL/LCL ต่อเนื่องยาวนานกว่าหรือเท่ากับ min_leak_duration_sec)
    """
    n_base = max(15, int(len(series) * baseline_ratio))
    base_data = series.iloc[:n_base]

    cl = float(base_data.mean())
    std = float(base_data.std())
    if std < 1e-9:
        std = 1e-6

    ucl = cl + k_sigma * std
    lcl = cl - k_sigma * std
    u_warn = cl + (k_sigma * 0.67) * std
    l_warn = cl - (k_sigma * 0.67) * std

    if len(timestamps) > 1:
        dt_sec = pd.to_datetime(timestamps).diff().median().total_seconds()
        if pd.isna(dt_sec) or dt_sec <= 0:
            dt_sec = 5.0
    else:
        dt_sec = 5.0

    vals = series.values
    n = len(vals)
    beyond_limit = (vals > ucl) | (vals < lcl)
    beyond_warn = (vals > u_warn) | (vals < l_warn)

    diffs = np.diff(vals, prepend=vals[0])
    inc_run = np.zeros(n, dtype=int)
    dec_run = np.zeros(n, dtype=int)
    for i in range(1, n):
        if diffs[i] > 0:
            inc_run[i] = inc_run[i - 1] + 1
        elif diffs[i] < 0:
            dec_run[i] = dec_run[i - 1] + 1

    trend_flag = (inc_run >= trend_window) | (dec_run >= trend_window)

    consec_steps = np.zeros(n, dtype=int)
    duration_sec = np.zeros(n, dtype=float)
    status = ["Normal"] * n

    i = 0
    while i < n:
        if beyond_limit[i]:
            j = i
            while j < n and beyond_limit[j]:
                j += 1
            for k in range(i, j):
                steps_so_far = k - i + 1
                consec_steps[k] = steps_so_far
                dur_so_far = steps_so_far * dt_sec
                duration_sec[k] = dur_so_far
                if dur_so_far >= min_leak_duration_sec:
                    status[k] = "Suspected Leak"
                else:
                    status[k] = "Transient Anomaly"
            i = j
        else:
            if beyond_warn[i] or trend_flag[i]:
                status[i] = "Warning"
            else:
                status[i] = "Normal"
            i += 1

    return {
        "CL": cl,
        "UCL": ucl,
        "LCL": lcl,
        "U_WARN": u_warn,
        "L_WARN": l_warn,
        "std": std,
        "dt_sec": dt_sec,
        "duration_sec": duration_sec,
        "trend_flag": trend_flag,
        "status": status
    }


def combine_zone_spc_status(ep_status_list, mf_status_list):
    """รวมสถานะ SPC ของ Pressure Residual (EP) และ Flow Difference ในแต่ละโซนตามลำดับความรุนแรง"""
    priority = {"Normal": 0, "Warning": 1, "Transient Anomaly": 2, "Suspected Leak": 3}
    inv_priority = {v: k for k, v in priority.items()}
    combined = []
    for s1, s2 in zip(ep_status_list, mf_status_list):
        c = max(priority[s1], priority[s2])
        combined.append(inv_priority[c])
    return combined


# ==========================================
# 4. MULTI-ZONE MACHINE LEARNING ENGINE
# ==========================================
def run_multizone_ml(phys_df, n_zones, spc_results, baseline_ratio=0.35, contamination=0.08, forecast_horizon=15):
    """
    วิเคราะห์ข้อมูลทุกโซนร่วมกันด้วย Multi-Zone Isolation Forest
    - แยกผลลัพธ์ออกจาก SPC อย่างชัดเจน
    - ใช้หลักฐานจาก Pressure Residual (EP) และ MassFlowDiff ร่วมกับ SPC Duration เพื่อยืนยันการรั่ว
    """
    feature_cols = []
    for z in range(1, n_zones + 1):
        feature_cols.extend([f"Z{z}_EP", f"Z{z}_MassFlowDiff", f"Z{z}_dP_meas"])

    X_raw = phys_df[feature_cols].copy().fillna(0.0)
    n_samples = len(X_raw)
    n_train = max(20, int(n_samples * baseline_ratio))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_raw.iloc[:n_train])
    X_all_scaled = scaler.transform(X_raw)

    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42
    )
    iso_forest.fit(X_train_scaled)

    raw_preds = iso_forest.predict(X_all_scaled)  # -1 = Anomaly, 1 = Normal
    decision_scores = -iso_forest.decision_function(X_all_scaled)

    s_min, s_max = decision_scores.min(), decision_scores.max()
    if s_max - s_min > 1e-9:
        anomaly_intensity = np.clip((decision_scores - s_min) / (s_max - s_min) * 100.0, 0.0, 100.0)
    else:
        anomaly_intensity = np.zeros(n_samples)

    zone_contrib = np.zeros((n_samples, n_zones))
    for z in range(1, n_zones + 1):
        idx_ep = feature_cols.index(f"Z{z}_EP")
        idx_mf = feature_cols.index(f"Z{z}_MassFlowDiff")
        z_score_sum = np.abs(X_all_scaled[:, idx_ep]) + np.abs(X_all_scaled[:, idx_mf])
        zone_contrib[:, z - 1] = z_score_sum

    ml_status = []
    localized_zone = []
    cross_zone_pattern = []
    severity_level = []

    for i in range(n_samples):
        is_anom = (raw_preds[i] == -1)
        top_z_idx = int(np.argmax(zone_contrib[i])) + 1

        elevated_zones = [f"Zone {z+1}" for z in range(n_zones) if zone_contrib[i, z] > 3.5]

        # Physical & SPC corroboration check using EP (Pressure Residual) and MassFlowDiff
        spc_z_stat = spc_results[top_z_idx]["combined_status"][i]
        mf_val = phys_df[f"Z{top_z_idx}_MassFlowDiff"].iloc[i]
        mf_ucl = spc_results[top_z_idx]["mf"]["UCL"]

        ep_val = phys_df[f"Z{top_z_idx}_EP"].iloc[i]
        ep_ucl = spc_results[top_z_idx]["ep"]["UCL"]
        ep_lcl = spc_results[top_z_idx]["ep"]["LCL"]
        ep_uwarn = spc_results[top_z_idx]["ep"]["U_WARN"]
        ep_lwarn = spc_results[top_z_idx]["ep"]["L_WARN"]

        ep_exceeded_warn = (ep_val > ep_uwarn) or (ep_val < ep_lwarn)
        ep_exceeded_limit = (ep_val > ep_ucl) or (ep_val < ep_lcl)

        has_physical_evidence = (mf_val > mf_ucl) and ep_exceeded_warn
        has_spc_duration_evidence = (spc_z_stat == "Suspected Leak")

        if not is_anom:
            ml_status.append("ML Normal")
            localized_zone.append("None")
            cross_zone_pattern.append("Stable Multi-Zone Profile")
            severity_level.append("Low (0-25%)")
        else:
            if has_physical_evidence and has_spc_duration_evidence:
                ml_status.append("ML Leak Candidate (Corroborated)")
                localized_zone.append(f"Zone {top_z_idx}")
            elif has_physical_evidence or ep_exceeded_limit or spc_z_stat in ["Warning", "Transient Anomaly"]:
                ml_status.append("ML Anomaly (Pending Duration/Evidence)")
                localized_zone.append(f"Zone {top_z_idx} (Unconfirmed)")
            else:
                ml_status.append("ML Unconfirmed Anomaly (Insufficient Evidence)")
                localized_zone.append("Uncertain (Noise/Shift)")

            if len(elevated_zones) > 1:
                cross_zone_pattern.append(f"Propagated Across {', '.join(elevated_zones)}")
            elif len(elevated_zones) == 1:
                cross_zone_pattern.append(f"Isolated in {elevated_zones[0]}")
            else:
                cross_zone_pattern.append("Minor Multivariate Shift")

            score_val = anomaly_intensity[i]
            if score_val >= 75 and has_physical_evidence:
                severity_level.append(f"Critical ({score_val:.0f}%)")
            elif score_val >= 50:
                severity_level.append(f"High ({score_val:.0f}%)")
            elif score_val >= 30:
                severity_level.append(f"Moderate ({score_val:.0f}%)")
            else:
                severity_level.append(f"Low ({score_val:.0f}%)")

    # Trend Prediction per zone (Linear Regression on recent MassFlowDiff & EP)
    forecasts = {}
    recent_w = min(40, n_samples)
    x_reg = np.arange(recent_w).reshape(-1, 1)
    x_fut = np.arange(recent_w, recent_w + forecast_horizon).reshape(-1, 1)

    for z in range(1, n_zones + 1):
        y_mf = phys_df[f"Z{z}_MassFlowDiff"].iloc[-recent_w:].values
        y_ep = phys_df[f"Z{z}_EP"].iloc[-recent_w:].values

        reg_mf = LinearRegression().fit(x_reg, y_mf)
        pred_mf = reg_mf.predict(x_fut)

        reg_ep = LinearRegression().fit(x_reg, y_ep)
        pred_ep = reg_ep.predict(x_fut)

        forecasts[z] = {
            "mf_slope": float(reg_mf.coef_[0]),
            "mf_pred": pred_mf,
            "ep_slope": float(reg_ep.coef_[0]),
            "ep_pred": pred_ep
        }

    ml_df = pd.DataFrame({
        "timestamp": phys_df["timestamp"],
        "ML_Prediction": raw_preds,
        "ML_Anomaly_Score": np.round(anomaly_intensity, 2),
        "ML_Status": ml_status,
        "ML_Localized_Zone": localized_zone,
        "ML_Cross_Zone_Pattern": cross_zone_pattern,
        "ML_Severity": severity_level
    })

    for z in range(1, n_zones + 1):
        ml_df[f"Z{z}_ML_Contrib"] = np.round(zone_contrib[:, z - 1], 2)

    return ml_df, forecasts


# ==========================================
# 5. PACKAGE BUILDER (IN-APP ZIP EXPORT)
# ==========================================
def create_project_zip(sample_df):
    """สร้างไฟล์ ZIP ที่บรรจุ app.py, requirements.txt, sample_sensor_data.csv, และ README.md"""
    zip_buffer = io.BytesIO()

    reqs_content = """streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.3.0
"""

    readme_content = """# Multi-Zone Pipeline Leak Detection System

ระบบตรวจจับการรั่วไหลของท่อน้ำในโรงงานอุตสาหกรรมแบบหลายโซน (Series Pipeline) ด้วยการผสาน:
1. Physics-based Model (Darcy-Weisbach)
2. Statistical Process Control (SPC) โดยใช้ Pressure Residual (EP) เป็นตัวหลักใน Pressure Drop Control Chart และเก็บค่า Pressure Drop จริงไว้สำหรับ Monitoring & Diagnosis
3. Multi-Zone Machine Learning (Isolation Forest)

## วิธีติดตั้งและใช้งาน
pip install -r requirements.txt
streamlit run app.py
"""
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            app_code = f.read()
    except Exception:
        app_code = "# Source code loaded in memory"

    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("app.py", app_code)
        zf.writestr("requirements.txt", reqs_content)
        zf.writestr("sample_sensor_data.csv", csv_bytes)
        zf.writestr("README.md", readme_content)

    zip_buffer.seek(0)
    return zip_buffer


# ==========================================
# 6. SIDEBAR & GLOBAL PARAMETERS
# ==========================================
st.sidebar.title("🚰 Pipeline Leak System")
st.sidebar.caption("Physics + Residual SPC + Multi-Zone ML")

data_source = st.sidebar.radio(
    "📂 เลือกแหล่งข้อมูล (Data Source):",
    ["Demo Data (จำลองสถานการณ์จริง)", "Upload CSV File"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Shared Pipeline Parameters")
st.sidebar.caption("พารามิเตอร์กายภาพที่ใช้ร่วมกันในทุก Monitoring Zone")

param_L = st.sidebar.number_input("Sensor Spacing - L (m)", min_value=1.0, max_value=5000.0, value=100.0, step=10.0)
param_D = st.sidebar.number_input("Pipe Diameter - D (mm)", min_value=10.0, max_value=2000.0, value=150.0, step=5.0)
param_e = st.sidebar.number_input("Pipe Roughness - e (mm)", min_value=0.001, max_value=10.0, value=0.045, step=0.005, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 SPC & ML Threshold Settings")
baseline_pct = st.sidebar.slider("Baseline Window (% ของข้อมูลแรก)", min_value=15, max_value=70, value=35, step=5) / 100.0
spc_sigma = st.sidebar.slider("SPC Control Limit (k-Sigma)", min_value=2.0, max_value=4.5, value=3.0, step=0.25)
min_leak_dur = st.sidebar.number_input("Min Leak Duration (วินาที)", min_value=5.0, max_value=600.0, value=30.0, step=5.0, help="ระยะเวลาขั้นต่ำที่ค่าต้องเกิน UCL/LCL ติดต่อกันเพื่อเปลี่ยนจาก Transient Anomaly เป็น Suspected Leak")
ml_contam = st.sidebar.slider("ML Contamination Rate", min_value=0.01, max_value=0.25, value=0.08, step=0.01)

# Load or Generate Data
if data_source == "Demo Data (จำลองสถานการณ์จริง)":
    num_stations_demo = st.sidebar.slider("จำนวน Sensor Stations (N)", min_value=3, max_value=8, value=5, help="สร้าง N-1 Monitoring Zones")
    raw_df = generate_demo_data(n_stations=num_stations_demo, L=param_L, D_mm=param_D, e_mm=param_e)
else:
    uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ CSV (ต้องมี timestamp, P1..Pn, Q1..Qn, T1..Tn)", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        if "timestamp" in raw_df.columns:
            raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
        else:
            raw_df["timestamp"] = pd.date_range("2026-11-10 08:00:00", periods=len(raw_df), freq="5s")
    else:
        st.sidebar.warning("ยังไม่ได้อัปโหลดไฟล์ ระบบแสดง Demo Data ชั่วคราว")
        raw_df = generate_demo_data(n_stations=5, L=param_L, D_mm=param_D, e_mm=param_e)

# Detect available stations dynamically
p_cols = [c for c in raw_df.columns if c.startswith("P") and c[1:].isdigit()]
n_stations = len(p_cols)
n_zones = n_stations - 1

if n_zones < 1:
    st.error("ข้อมูลต้องมีอย่างน้อย 2 Sensor Stations (P1, P2, Q1, Q2, T1, T2) เพื่อสร้าง 1 Monitoring Zone")
    st.stop()

# ==========================================
# EXECUTE PIPELINE CALCULATIONS
# ==========================================
phys_dfs = []
for z in range(1, n_zones + 1):
    z_df = compute_physics_zone(raw_df, z, param_L, param_D, param_e)
    if z > 1:
        z_df = z_df.drop(columns=["timestamp"])
    phys_dfs.append(z_df)

phys_all_df = pd.concat(phys_dfs, axis=1)

# SPC uses Pressure Residual (EP) as the primary metric for Pressure Drop Control Chart
spc_results = {}
for z in range(1, n_zones + 1):
    ep_spc = compute_spc_and_duration(
        phys_all_df[f"Z{z}_EP"],
        phys_all_df["timestamp"],
        baseline_ratio=baseline_pct,
        k_sigma=spc_sigma,
        min_leak_duration_sec=min_leak_dur
    )
    mf_spc = compute_spc_and_duration(
        phys_all_df[f"Z{z}_MassFlowDiff"],
        phys_all_df["timestamp"],
        baseline_ratio=baseline_pct,
        k_sigma=spc_sigma,
        min_leak_duration_sec=min_leak_dur
    )
    combined_stat = combine_zone_spc_status(ep_spc["status"], mf_spc["status"])
    spc_results[z] = {
        "ep": ep_spc,
        "mf": mf_spc,
        "combined_status": combined_stat
    }
    phys_all_df[f"Z{z}_SPC_EP_Status"] = ep_spc["status"]
    phys_all_df[f"Z{z}_SPC_MF_Status"] = mf_spc["status"]
    phys_all_df[f"Z{z}_SPC_Status"] = combined_stat
    phys_all_df[f"Z{z}_Violation_Duration_s"] = np.maximum(ep_spc["duration_sec"], mf_spc["duration_sec"])

ml_df, ml_forecasts = run_multizone_ml(
    phys_all_df,
    n_zones,
    spc_results,
    baseline_ratio=baseline_pct,
    contamination=ml_contam
)

full_export_df = pd.concat([raw_df, phys_all_df.drop(columns=["timestamp"]), ml_df.drop(columns=["timestamp"])], axis=1)

# Sidebar Download Package Button
st.sidebar.markdown("---")
st.sidebar.subheader("📦 ส่งมอบโปรแกรม (Deliverables)")
zip_pkg = create_project_zip(generate_demo_data(n_stations=5))
st.sidebar.download_button(
    label="⬇️ Download Full ZIP Package",
    data=zip_pkg,
    file_name="MultiZone_Leak_Detection_System.zip",
    mime="application/zip",
    use_container_width=True
)

# ==========================================
# 7. MAIN DASHBOARD TABS
# ==========================================
st.title("🚰 Multi-Zone Pipeline Leak Detection System")
st.markdown("ระบบตรวจจับการรั่วไหลของท่อน้ำในโรงงานด้วย **Physics-Based Hydraulics (Darcy-Weisbach)** + **Model-Based SPC (Pressure Residual $E_P$)** + **Multi-Zone Machine Learning**")

tabs = st.tabs([
    "1️⃣ System Overview",
    "2️⃣ Sensor & Zone Configuration",
    "3️⃣ Zone Monitoring (Physics & SPC)",
    "4️⃣ Machine Learning (Multi-Zone)",
    "5️⃣ Event History & Report"
])

# ------------------------------------------
# TAB 1: SYSTEM OVERVIEW
# ------------------------------------------
with tabs[0]:
    st.subheader("🌐 ภาพรวมสถานะระบบท่อแบบอนุกรม (Series Pipeline Status)")

    inspect_mode = st.radio(
        "โหมดการแสดงผลสถานะ:",
        ["แสดงสถานะจุดเวลาล่าสุด (Latest Timestamp)", "เลือกจุดเวลาเอง (Time Step Inspector)"],
        horizontal=True
    )
    if inspect_mode == "เลือกจุดเวลาเอง (Time Step Inspector)":
        step_idx = st.slider("เลือก Time Step:", 0, len(full_export_df) - 1, len(full_export_df) - 1)
    else:
        step_idx = len(full_export_df) - 1

    curr_time = full_export_df["timestamp"].iloc[step_idx]
    st.info(f"🕒 **กำลังแสดงผลข้อมูลที่เวลา:** `{curr_time}` (Time Step: {step_idx}) | จำนวนสถานี: **{n_stations} Stations** | จำนวนโซน: **{n_zones} Zones**")

    kpi_cols = st.columns(4)
    active_leaks_spc = sum(1 for z in range(1, n_zones + 1) if spc_results[z]["combined_status"][step_idx] == "Suspected Leak")
    active_transients_spc = sum(1 for z in range(1, n_zones + 1) if spc_results[z]["combined_status"][step_idx] == "Transient Anomaly")
    curr_ml_status = ml_df["ML_Status"].iloc[step_idx]
    curr_ml_loc = ml_df["ML_Localized_Zone"].iloc[step_idx]

    kpi_cols[0].metric("SPC Suspected Leaks", f"{active_leaks_spc} โซน", delta="Critical" if active_leaks_spc > 0 else "Normal", delta_color="inverse")
    kpi_cols[1].metric("SPC Transient Anomalies", f"{active_transients_spc} โซน", delta="Short Spike" if active_transients_spc > 0 else "None", delta_color="off")
    kpi_cols[2].metric("Multi-Zone ML Status", curr_ml_status)
    kpi_cols[3].metric("ML Suspected Location", curr_ml_loc, f"Severity: {ml_df['ML_Severity'].iloc[step_idx]}")

    st.markdown("#### 🗺️ แผนผังการวางตัวของสถานีวัดและโซนตรวจจับ (Series Pipeline Topology)")
    fig_pipe = go.Figure()
    station_x = list(range(1, n_stations + 1))
    station_y = [1] * n_stations

    color_map = {
        "Normal": "#198754",
        "Warning": "#ffc107",
        "Transient Anomaly": "#fd7e14",
        "Suspected Leak": "#dc3545"
    }

    for z in range(1, n_zones + 1):
        z_stat = spc_results[z]["combined_status"][step_idx]
        seg_color = color_map.get(z_stat, "#6c757d")
        dp_val = phys_all_df[f"Z{z}_dP_meas"].iloc[step_idx] / 1000.0
        ep_val = phys_all_df[f"Z{z}_EP"].iloc[step_idx]
        mf_val = phys_all_df[f"Z{z}_MassFlowDiff"].iloc[step_idx]

        fig_pipe.add_trace(go.Scatter(
            x=[z, z + 1],
            y=[1, 1],
            mode="lines",
            line=dict(color=seg_color, width=16),
            name=f"Zone {z} ({z_stat})",
            hoverinfo="text",
            text=(
                f"<b>Zone {z}</b><br>"
                f"SPC Status: {z_stat}<br>"
                f"EP (Residual): {ep_val:.1f} Pa<br>"
                f"Measured ΔP (Monitoring): {dp_val:.2f} kPa<br>"
                f"Δm: {mf_val:.3f} kg/s"
            )
        ))
        fig_pipe.add_annotation(
            x=z + 0.5, y=1.18,
            text=(
                f"<b>Zone {z}</b><br>"
                f"<span style='font-size:11px;'>SPC: {z_stat}</span><br>"
                f"<span style='font-size:11px;'>EP: {ep_val:.0f} Pa | Δm: {mf_val:.2f} kg/s</span>"
            ),
            showarrow=False
        )

    station_texts = []
    for s in range(1, n_stations + 1):
        p_bar = raw_df[f"P{s}"].iloc[step_idx] / 100000.0
        q_ls = raw_df[f"Q{s}"].iloc[step_idx] * 1000.0
        t_c = raw_df[f"T{s}"].iloc[step_idx]
        station_texts.append(f"<b>Station {s}</b><br>P{s}: {p_bar:.2f} bar<br>Q{s}: {q_ls:.2f} L/s<br>T{s}: {t_c:.1f} °C")

    fig_pipe.add_trace(go.Scatter(
        x=station_x,
        y=station_y,
        mode="markers+text",
        marker=dict(size=28, color="#0d6efd", line=dict(width=3, color="white")),
        text=[f"S{s}" for s in station_x],
        textposition="middle center",
        textfont=dict(color="white", size=12),
        hoverinfo="text",
        hovertext=station_texts,
        name="Sensor Stations"
    ))

    fig_pipe.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False, range=[0.5, n_stations + 0.5]),
        yaxis=dict(visible=False, range=[0.6, 1.45]),
        plot_bgcolor="white",
        showlegend=False
    )
    st.plotly_chart(fig_pipe, use_container_width=True)

    st.markdown("#### 📋 ตารางเปรียบเทียบผลลัพธ์แยกตามโซน (SPC Control Metrics vs. Hydraulic Diagnosis Variables)")
    overview_rows = []
    for z in range(1, n_zones + 1):
        overview_rows.append({
            "Monitoring Zone": f"Zone {z} (S{z} → S{z+1})",
            "Pressure Residual EP [SPC Primary] (Pa)": round(phys_all_df[f"Z{z}_EP"].iloc[step_idx], 1),
            "Mass Flow Diff [SPC Primary] (kg/s)": round(phys_all_df[f"Z{z}_MassFlowDiff"].iloc[step_idx], 4),
            "Measured ΔP [Diagnosis] (Pa)": round(phys_all_df[f"Z{z}_dP_meas"].iloc[step_idx], 1),
            "Darcy Predicted ΔP [Diagnosis] (Pa)": round(phys_all_df[f"Z{z}_dP_pred"].iloc[step_idx], 1),
            "Reynolds Number (Re)": f"{phys_all_df[f'Z{z}_Re'].iloc[step_idx]:,.0f}",
            "Friction Factor (f)": round(phys_all_df[f"Z{z}_f"].iloc[step_idx], 5),
            "SPC Status (EP & Δm)": spc_results[z]["combined_status"][step_idx],
            "Violation Duration (s)": round(phys_all_df[f"Z{z}_Violation_Duration_s"].iloc[step_idx], 1),
            "ML Zone Anomaly Contrib.": round(ml_df[f"Z{z}_ML_Contrib"].iloc[step_idx], 2)
        })
    st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

# ------------------------------------------
# TAB 2: SENSOR & ZONE CONFIGURATION
# ------------------------------------------
with tabs[1]:
    st.subheader("⚙️ การตั้งค่าสถานีวัด พารามิเตอร์ท่อ และจัดการข้อมูล (Sensor & Zone Configuration)")
    col_cfg1, col_cfg2 = st.columns([1, 1])

    with col_cfg1:
        st.markdown("##### 📐 พารามิเตอร์กายภาพของทุกโซน (Shared Zone Parameters)")
        st.table(pd.DataFrame({
            "Parameter": [
                "Sensor Spacing (L)",
                "Pipe Internal Diameter (D)",
                "Pipe Cross-Sectional Area (A)",
                "Pipe Roughness (e)",
                "Relative Roughness (e/D)",
                "Number of Sensor Stations",
                "Number of Monitoring Zones"
            ],
            "Value": [
                f"{param_L:.2f} m",
                f"{param_D:.2f} mm ({param_D/1000:.4f} m)",
                f"{np.pi * ((param_D/2000)**2):.6f} m²",
                f"{param_e:.4f} mm",
                f"{(param_e / param_D):.6f}",
                f"{n_stations} สถานี",
                f"{n_zones} โซน"
            ]
        }))

    with col_cfg2:
        st.markdown("##### 🌡️ คุณสมบัติของน้ำเฉลี่ยตามอุณหภูมิในแต่ละโซน (Water Properties)")
        prop_rows = []
        for z in range(1, n_zones + 1):
            prop_rows.append({
                "Zone": f"Zone {z} (S{z}-S{z+1})",
                "Mean Temp (°C)": round(phys_all_df[f"Z{z}_T_avg"].mean(), 2),
                "Water Density ρ (kg/m³)": round(phys_all_df[f"Z{z}_rho"].mean(), 2),
                "Dynamic Viscosity μ (Pa·s)": f"{phys_all_df[f'Z{z}_mu'].mean():.6e}",
                "Mean Velocity (m/s)": round(phys_all_df[f"Z{z}_Velocity"].mean(), 3),
                "Mean Reynolds (Re)": f"{phys_all_df[f'Z{z}_Re'].mean():,.0f}"
            })
        st.dataframe(pd.DataFrame(prop_rows), use_container_width=True, hide_index=True)

    st.markdown("##### 📥 ดาวน์โหลดไฟล์ตัวอย่างและส่งออกข้อมูลทั้งหมด (CSV Data Management)")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        sample_csv_bytes = generate_demo_data(n_stations=5, L=param_L, D_mm=param_D, e_mm=param_e).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ ดาวน์โหลด sample_sensor_data.csv (สำหรับทดสอบ Upload)",
            data=sample_csv_bytes,
            file_name="sample_sensor_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    with dl_col2:
        full_csv_bytes = full_export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export ผลการคำนวณทั้งหมด (Physics + SPC + ML) เป็น CSV",
            data=full_csv_bytes,
            file_name="multizone_leak_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("##### 🔍 ตัวอย่างข้อมูลดิบจากเซนเซอร์ (Raw Sensor Data Preview)")
    st.dataframe(raw_df.head(20), use_container_width=True)

# ------------------------------------------
# TAB 3: ZONE MONITORING (PHYSICS & SPC)
# ------------------------------------------
with tabs[2]:
    st.subheader("📈 การตรวจสอบรายโซนด้วย Model-Based SPC และตัวแปรวินิจฉัยทางชลศาสตร์")

    selected_zone = st.selectbox(
        "🎯 เลือก Monitoring Zone ที่ต้องการวิเคราะห์เชิงลึก:",
        options=list(range(1, n_zones + 1)),
        format_func=lambda z: f"Zone {z} (ระหว่าง Station {z} และ Station {z+1})"
    )

    z = selected_zone
    ep_info = spc_results[z]["ep"]
    mf_info = spc_results[z]["mf"]

    st.markdown("##### 📌 ตัวแปรควบคุมหลัก (SPC Primary Variables) และตัวแปรสำหรับ Monitoring & Diagnosis")
    m_cols = st.columns(6)
    m_cols[0].metric("Mean Residual EP (SPC)", f"{phys_all_df[f'Z{z}_EP'].mean():,.1f} Pa")
    m_cols[1].metric("Mean Mass Flow Diff (SPC)", f"{phys_all_df[f'Z{z}_MassFlowDiff'].mean():.4f} kg/s")
    m_cols[2].metric("Measured ΔP (Diagnosis)", f"{phys_all_df[f'Z{z}_dP_meas'].mean():,.1f} Pa")
    m_cols[3].metric("Predicted ΔP (Diagnosis)", f"{phys_all_df[f'Z{z}_dP_pred'].mean():,.1f} Pa")
    m_cols[4].metric("Mean Reynolds (Re)", f"{phys_all_df[f'Z{z}_Re'].mean():,.0f}")
    m_cols[5].metric("Mean Friction Factor (f)", f"{phys_all_df[f'Z{z}_f'].mean():.5f}")

    def plot_spc_chart(df_time, values, spc_dict, title, y_label, line_color="#1f77b4", series_name="Measured Value"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_time, y=values, mode="lines", name=series_name,
            line=dict(color=line_color, width=1.8)
        ))

        fig.add_hline(y=spc_dict["UCL"], line_dash="dash", line_color="#dc3545", annotation_text=f"UCL ({spc_dict['UCL']:.2f})")
        fig.add_hline(y=spc_dict["CL"], line_dash="solid", line_color="#198754", annotation_text=f"CL ({spc_dict['CL']:.2f})")
        fig.add_hline(y=spc_dict["LCL"], line_dash="dash", line_color="#dc3545", annotation_text=f"LCL ({spc_dict['LCL']:.2f})")

        status_arr = np.array(spc_dict["status"])
        for st_label, color, sym, sz in [
            ("Warning", "#ffc107", "circle", 6),
            ("Transient Anomaly", "#fd7e14", "diamond", 8),
            ("Suspected Leak", "#dc3545", "x", 9)
        ]:
            mask = (status_arr == st_label)
            if np.any(mask):
                fig.add_trace(go.Scatter(
                    x=df_time[mask], y=values[mask],
                    mode="markers", name=st_label,
                    marker=dict(color=color, symbol=sym, size=sz)
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Timestamp",
            yaxis_title=y_label,
            height=370,
            margin=dict(l=20, r=20, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    # Control Chart 1: Pressure Drop Control Chart using Pressure Residual (EP) as Primary Metric
    fig_ep_spc = plot_spc_chart(
        phys_all_df["timestamp"],
        phys_all_df[f"Z{z}_EP"],
        ep_info,
        title=f"Zone {z} — Control Chart 1: Pressure Drop Control Chart (ใช้ Pressure Residual EP = ΔP_meas - ΔP_pred เป็นตัวหลัก)",
        y_label="Pressure Residual EP (Pa)",
        line_color="#6f42c1",
        series_name="Pressure Residual (EP)"
    )
    st.plotly_chart(fig_ep_spc, use_container_width=True)

    # Control Chart 2: Mass Flow Difference Control Chart
    fig_mf_spc = plot_spc_chart(
        phys_all_df["timestamp"],
        phys_all_df[f"Z{z}_MassFlowDiff"],
        mf_info,
        title=f"Zone {z} — Control Chart 2: Flow Difference Control Chart (Δm = m_in - m_out)",
        y_label="Mass Flow Difference (kg/s)",
        line_color="#1f77b4",
        series_name="Mass Flow Diff (Δm)"
    )
    st.plotly_chart(fig_mf_spc, use_container_width=True)

    # Bottom Section: Duration Analysis & Actual Pressure Drop Monitoring/Diagnosis
    st.markdown("##### 🩺 กราฟวิเคราะห์ระยะเวลาต่อเนื่อง (Duration Analysis) และกราฟติดตามแรงดันตกคร่อมจริง (Monitoring & Diagnosis)")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_dur = go.Figure()
        fig_dur.add_trace(go.Scatter(
            x=phys_all_df["timestamp"],
            y=phys_all_df[f"Z{z}_Violation_Duration_s"],
            fill="tozeroy",
            mode="lines",
            name="Violation Duration (s)",
            line=dict(color="#dc3545")
        ))
        fig_dur.add_hline(
            y=min_leak_dur, line_dash="dash", line_color="black",
            annotation_text=f"Leak Threshold ({min_leak_dur}s)"
        )
        fig_dur.update_layout(
            title=f"Zone {z} — Time-based Duration Analysis (วินาทีที่ EP หรือ Δm เกิน Control Limit ต่อเนื่อง)",
            xaxis_title="Timestamp",
            yaxis_title="Duration (Seconds)",
            height=320,
            margin=dict(l=20, r=20, t=45, b=20)
        )
        st.plotly_chart(fig_dur, use_container_width=True)

    with col_d2:
        fig_diag = go.Figure()
        fig_diag.add_trace(go.Scatter(
            x=phys_all_df["timestamp"],
            y=phys_all_df[f"Z{z}_dP_meas"],
            mode="lines",
            name="Actual Measured ΔP (Pa)",
            line=dict(color="#0d6efd", width=1.8)
        ))
        fig_diag.add_trace(go.Scatter(
            x=phys_all_df["timestamp"],
            y=phys_all_df[f"Z{z}_dP_pred"],
            mode="lines",
            name="Darcy-Weisbach Predicted ΔP (Pa)",
            line=dict(color="#198754", width=1.6, dash="dot")
        ))
        fig_diag.update_layout(
            title=f"Zone {z} — Monitoring & Diagnosis: Actual Pressure Drop (ΔP_meas) vs. Predicted (ΔP_pred)",
            xaxis_title="Timestamp",
            yaxis_title="Pressure Drop (Pa)",
            height=320,
            margin=dict(l=20, r=20, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_diag, use_container_width=True)

# ------------------------------------------
# TAB 4: MACHINE LEARNING (MULTI-ZONE)
# ------------------------------------------
with tabs[3]:
    st.subheader("🤖 Multi-Zone Machine Learning (Isolation Forest & Cross-Zone Analytics)")
    st.markdown("""
    > **หลักการแยกผลวิเคราะห์ (Separation of Concerns & Guardrail):**
    > โมเดล **Multi-Zone Isolation Forest** เรียนรู้ความสัมพันธ์ของฟีเจอร์ทุกโซนพร้อมกัน ($E_P, \Delta \dot{m}, \Delta P_{\text{meas}}$) เพื่อตรวจจับความผิดปกติและรูปแบบการแพร่กระจายข้ามโซน (Cross-zone Pattern) อย่างไรก็ตาม **ML จะไม่ยืนยันการรั่วไหลโดยลำพัง** หากไม่มีหลักฐานทางกายภาพจากค่า $E_P$ และ $\Delta \dot{m}$ ร่วมกับระยะเวลาจาก SPC รองรับ
    """)

    fig_ml_score = go.Figure()
    fig_ml_score.add_trace(go.Scatter(
        x=ml_df["timestamp"],
        y=ml_df["ML_Anomaly_Score"],
        mode="lines",
        name="Multi-Zone Anomaly Score (0-100)",
        line=dict(color="#343a40", width=1.8)
    ))

    for st_name, col_hex in [
        ("ML Leak Candidate (Corroborated)", "#dc3545"),
        ("ML Anomaly (Pending Duration/Evidence)", "#fd7e14"),
        ("ML Unconfirmed Anomaly (Insufficient Evidence)", "#ffc107")
    ]:
        mask = (ml_df["ML_Status"] == st_name)
        if np.any(mask):
            fig_ml_score.add_trace(go.Scatter(
                x=ml_df["timestamp"][mask],
                y=ml_df["ML_Anomaly_Score"][mask],
                mode="markers",
                name=st_name,
                marker=dict(color=col_hex, size=8)
            ))

    fig_ml_score.update_layout(
        title="Multi-Zone Isolation Forest — Anomaly Intensity Score & Evidence-Gated Classification",
        xaxis_title="Timestamp",
        yaxis_title="Anomaly Score (0 - 100)",
        height=350,
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_ml_score, use_container_width=True)

    st.markdown("##### 🔥 Cross-Zone Anomaly Contribution Heatmap")
    contrib_cols = [f"Z{z}_ML_Contrib" for z in range(1, n_zones + 1)]
    z_matrix = ml_df[contrib_cols].values.T

    fig_heat = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=ml_df["timestamp"],
        y=[f"Zone {z}" for z in range(1, n_zones + 1)],
        colorscale="YlOrRd",
        colorbar=dict(title="Z-Contribution")
    ))
    fig_heat.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=25, b=20),
        xaxis_title="Timestamp",
        yaxis_title="Monitoring Zone"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("##### 🔮 ML Trend Prediction (การพยากรณ์แนวโน้มล่วงหน้า 15 Steps)")
    pred_zone = st.selectbox(
        "เลือกโซนที่ต้องการดูแนวโน้มการพยากรณ์ (Trend Forecast):",
        options=list(range(1, n_zones + 1)),
        format_func=lambda z: f"Zone {z}",
        key="ml_forecast_zone"
    )

    f_data = ml_forecasts[pred_zone]
    last_ts = pd.to_datetime(phys_all_df["timestamp"].iloc[-1])
    dt_s = spc_results[pred_zone]["mf"]["dt_sec"]
    future_ts = pd.date_range(last_ts + pd.Timedelta(seconds=dt_s), periods=len(f_data["mf_pred"]), freq=f"{int(dt_s)}s")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        fig_f_mf = go.Figure()
        fig_f_mf.add_trace(go.Scatter(
            x=phys_all_df["timestamp"].iloc[-60:],
            y=phys_all_df[f"Z{pred_zone}_MassFlowDiff"].iloc[-60:],
            mode="lines", name="Historical Δm"
        ))
        fig_f_mf.add_trace(go.Scatter(
            x=future_ts,
            y=f_data["mf_pred"],
            mode="lines+markers", name="Linear Trend Forecast",
            line=dict(color="#dc3545", dash="dash")
        ))
        fig_f_mf.add_hline(y=spc_results[pred_zone]["mf"]["UCL"], line_dash="dot", line_color="red", annotation_text="UCL")
        fig_f_mf.update_layout(
            title=f"Zone {pred_zone} — Mass Flow Diff Trend Forecast (Slope: {f_data['mf_slope']:.5f} kg/s per step)",
            height=300, margin=dict(l=20, r=20, t=45, b=20)
        )
        st.plotly_chart(fig_f_mf, use_container_width=True)

    with col_f2:
        fig_f_ep = go.Figure()
        fig_f_ep.add_trace(go.Scatter(
            x=phys_all_df["timestamp"].iloc[-60:],
            y=phys_all_df[f"Z{pred_zone}_EP"].iloc[-60:],
            mode="lines", name="Historical EP", line=dict(color="#6f42c1")
        ))
        fig_f_ep.add_trace(go.Scatter(
            x=future_ts,
            y=f_data["ep_pred"],
            mode="lines+markers", name="Linear Trend Forecast",
            line=dict(color="#fd7e14", dash="dash")
        ))
        fig_f_ep.add_hline(y=spc_results[pred_zone]["ep"]["UCL"], line_dash="dot", line_color="red", annotation_text="UCL (EP)")
        fig_f_ep.update_layout(
            title=f"Zone {pred_zone} — Pressure Residual (EP) Trend Forecast (Slope: {f_data['ep_slope']:.2f} Pa/step)",
            height=300, margin=dict(l=20, r=20, t=45, b=20)
        )
        st.plotly_chart(fig_f_ep, use_container_width=True)

# ------------------------------------------
# TAB 5: EVENT HISTORY & REPORT
# ------------------------------------------
with tabs[4]:
    st.subheader("📑 ประวัติเหตุการณ์ผิดปกติและรายงานสรุปแยกตามระบบ (Event History & Comparison Report)")

    event_records = []
    for i in range(len(full_export_df)):
        ts = full_export_df["timestamp"].iloc[i]
        ml_st = ml_df["ML_Status"].iloc[i]
        ml_loc = ml_df["ML_Localized_Zone"].iloc[i]
        ml_pat = ml_df["ML_Cross_Zone_Pattern"].iloc[i]
        ml_sev = ml_df["ML_Severity"].iloc[i]

        spc_zone_alerts = []
        max_dur = 0.0
        for z in range(1, n_zones + 1):
            st_z = spc_results[z]["combined_status"][i]
            dur_z = phys_all_df[f"Z{z}_Violation_Duration_s"].iloc[i]
            if st_z != "Normal":
                spc_zone_alerts.append(f"Z{z}: {st_z} ({dur_z:.0f}s)")
                max_dur = max(max_dur, dur_z)

        if spc_zone_alerts or ml_st != "ML Normal":
            event_records.append({
                "Timestamp": ts,
                "SPC Status by Zone (EP & Δm)": ", ".join(spc_zone_alerts) if spc_zone_alerts else "All Zones Normal",
                "Max SPC Duration (s)": round(max_dur, 1),
                "ML Status (Independent)": ml_st,
                "ML Localized Zone": ml_loc,
                "ML Cross-Zone Pattern": ml_pat,
                "ML Severity": ml_sev
            })

    events_df = pd.DataFrame(event_records)

    col_f_ev1, col_f_ev2 = st.columns(2)
    with col_f_ev1:
        filter_spc = st.selectbox("กรองตามสถานะ SPC:", ["ทั้งหมด", "Suspected Leak", "Transient Anomaly", "Warning"])
    with col_f_ev2:
        filter_ml = st.selectbox("กรองตามสถานะ ML:", ["ทั้งหมด", "ML Leak Candidate (Corroborated)", "ML Anomaly (Pending Duration/Evidence)", "ML Unconfirmed Anomaly (Insufficient Evidence)"])

    filtered_events = events_df.copy()
    if not filtered_events.empty:
        if filter_spc != "ทั้งหมด":
            filtered_events = filtered_events[filtered_events["SPC Status by Zone (EP & Δm)"].str.contains(filter_spc)]
        if filter_ml != "ทั้งหมด":
            filtered_events = filtered_events[filtered_events["ML Status (Independent)"] == filter_ml]

    st.dataframe(filtered_events, use_container_width=True, hide_index=True)

    if not filtered_events.empty:
        ev_csv = filtered_events.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ ดาวน์โหลดรายงานเหตุการณ์ผิดปกติ (Event History CSV)",
            data=ev_csv,
            file_name="pipeline_event_history_report.csv",
            mime="text/csv"
        )