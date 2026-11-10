# Water Pipeline Leak Detection System

## Run
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install packages: `pip install -r requirements.txt`
4. Start: `streamlit run app.py`
5. Open the local URL shown in the terminal (usually http://localhost:8501).

## Input CSV
### Pre-calculated mode
Required columns: `timestamp, EP, m_diff`
- EP: absolute pressure residual in Pa.
- m_diff: absolute mass-flow difference in kg/s.

### Raw sensor mode
Required columns: `timestamp, P_in, P_out, Q_in, Q_out, Temperature`
- P_in / P_out: Pa(g)
- Q_in / Q_out: L/min
- Temperature: deg C

Enter geometry in the sidebar: L = sensor spacing (m), D = internal pipe diameter (mm), e = absolute pipe roughness (mm).

## Engineering notes
- Raw-mode EP = abs(measured pressure drop - Darcy-Weisbach predicted pressure drop), in Pa.
- Pressure Drop and predicted pressure drop are displayed in Pa.
- Water properties use approximate temperature correlations; pressure dependence is neglected.
- Swamee-Jain is used for turbulent friction factor; laminar uses 64/Re; transitional flow is flagged.
- SPC limits use baseline mean +/- 3 sample standard deviations. Baseline must be stable and leak-free.
- Isolation Forest is trained on the selected baseline. It is an anomaly indicator, not proof of a leak.
- Synthetic demo data is for software demonstration only.
