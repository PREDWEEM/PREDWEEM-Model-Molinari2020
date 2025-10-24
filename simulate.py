# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 WeedCropSystem — Modo Original (robusto)
# - Ciec dinámico por días calendario (logístico)
# - Supresión del cultivo S1–S4
# - Controles herbicidas: letal + residual
# - Sin efectos subletales ni TT modificado
# - Con logs y chequeos para depurar
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as go

# ---------- CONFIG ----------
st.set_page_config(page_title="🌾 WeedCropSystem — OK", layout="wide")
st.title("🌾 WeedCropSystem — Modo original (con chequeos)")

# ---------- UTIL ----------
def _as_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def _safe_date(x):
    if isinstance(x, (dt.date, dt.datetime, pd.Timestamp, np.datetime64)):
        return pd.to_datetime(x).date()
    return dt.date(2025, 6, 1)

# ---------- SIDEBAR ----------
st.sidebar.header("Escenario")
nyears = st.sidebar.slider("Años a simular", 1, 10, 3)
seed_bank0 = st.sidebar.number_input("Banco inicial (semillas·m⁻²)", 0, 20000, 4500)
Tb = st.sidebar.number_input("Temp. base Tb (°C)", 0.0, 15.0, 0.0, 0.5)
K = st.sidebar.number_input("Capacidad de carga K (pl·m⁻²)", 50, 2000, 250)
sim_seed = st.sidebar.number_input("Semilla aleatoria clima", 0, 999999, 42)

st.sidebar.divider()
st.sidebar.subheader("📅 Fechas")
sow_date = _safe_date(st.sidebar.date_input("Fecha de siembra", dt.date(2025, 6, 1)))

st.sidebar.divider()
st.sidebar.subheader("🌿 Canopia (LAI logístico)")
LAI_max = st.sidebar.slider("LAI_max", 2.0, 10.0, 6.0, 0.1)
t_lag = st.sidebar.slider("t_lag (d desde siembra)", 0, 60, 10)
t_close = st.sidebar.slider("t_close (d desde siembra, LAI≈0.5·max)", 10, 100, 35)
LAI_hc = st.sidebar.slider("LAI_hc", 2.0, 10.0, 6.0, 0.1)
Cs = st.sidebar.number_input("Cs (estándar, pl·m⁻²)", 50, 800, 200)
Ca = st.sidebar.number_input("Ca (real, pl·m⁻²)", 30, 800, 200)

st.sidebar.divider()
st.sidebar.subheader("⚖️ Supresión por estadio (exp de (1−Ciec))")
p_S1 = st.sidebar.slider("S1", 0.0, 2.0, 1.0, 0.1)
p_S2 = st.sidebar.slider("S2", 0.0, 2.0, 0.6, 0.1)
p_S3 = st.sidebar.slider("S3", 0.0, 2.0, 0.4, 0.1)
p_S4 = st.sidebar.slider("S4", 0.0, 2.0, 0.2, 0.1)

st.sidebar.divider()
st.sidebar.subheader("🧪 Controles")
preR_days_before = st.sidebar.number_input("PreR: días antes siembra", 0, 45, 14)
preR_eff_S1S2 = st.sidebar.slider("PreR eficacia S1–S2 (%)", 0, 100, 90)
preR_residual = st.sidebar.slider("PreR residual (d)", 0, 90, 30)

postR_days_after = st.sidebar.number_input("PostR: días después siembra", 0, 90, 25)
postR_eff_S1S4 = st.sidebar.slider("PostR eficacia S1–S4 (%)", 0, 100, 85)
postR_residual = st.sidebar.slider("PostR residual (d)", 0, 90, 10)

gram_days_after = st.sidebar.number_input("Graminicida: días después siembra", 0, 90, 10)
gram_eff_S1S3 = st.sidebar.slider("Gram eficacia S1–S3 (%)", 0, 100, 80)
gram_residual = st.sidebar.slider("Gram residual (d)", 0, 90, 7)

st.sidebar.divider()
debug = st.sidebar.checkbox("🔎 Modo depuración (logs)", value=True)
run_btn = st.sidebar.button("▶ Ejecutar simulación")

# ---------- CLIMA SINTÉTICO ----------
def synthetic_meteo(start: dt.date, end: dt.date, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    doy = dates.dayofyear.to_numpy()
    tmean = 12 + 8*np.sin(2*np.pi*(doy-170)/365.0) + rng.normal(0,1.5,n)
    tmin = tmean - (3 + rng.normal(0,0.8,n))
    tmax = tmean + (6 + rng.normal(0,0.8,n))
    prec = rng.choice([0,0,0,0,3,8,15], size=n,
                      p=[0.55,0.15,0.10,0.05,0.07,0.05,0.03])
    df = pd.DataFrame({"date":dates, "tmin":tmin, "tmax":tmax, "prec":prec})
    return df.astype({"tmin":"float64", "tmax":"float64", "prec":"float64"})

# ---------- EMERGENCIA (placeholder) ----------
def emergence_simple(TT: float, prec: float) -> float:
    base = 1.0/(1.0 + np.exp(-(TT-300)/40.0))
    pulse = 0.002 if _as_float(prec) >= 5.0 else 0.0
    return float(min(base*0.003 + pulse, 0.02))

# ---------- LAI/CIEC ----------
def lai_logistic_by_day(days_since_sow: int, LAI_max: float, t_lag: int, t_close: int):
    t50 = max(int(t_close), 1)
    eps = 0.05
    denom = (int(t_lag) - t50)
    if abs(denom) < 1e-6:
        denom = -1.0
    k = -np.log(1/eps - 1) / denom
    LAI = LAI_max / (1.0 + np.exp(-k*(float(days_since_sow) - t50)))
    LAI = float(max(0.0, min(LAI, LAI_max)))
    return LAI, float(k)

def ciec_calendar(days_since_sow: int, LAI_max: float, t_lag: int, t_close: int,
                  LAI_hc: float, Cs: float, Ca: float):
    LAI, k = lai_logistic_by_day(days_since_sow, LAI_max, t_lag, t_close)
    ratio = (float(Cs) / max(float(Ca), 1e-6))
    Ciec = min((LAI / max(float(LAI_hc), 1e-6)) * ratio, 1.0)
    return float(Ciec), LAI, k

# ---------- SIMULADOR (MODO ORIGINAL) ----------
def simulate(
    nyears=3, seed_bank0=4500, K=250, Tb=0.0, seed=42,
    preR_days_before=14, preR_eff=90, preR_residual=30,
    postR_days_after=25, postR_eff=85, postR_residual=10,
    gram_days_after=10, gram_eff=80, gram_residual=7,
    sow_date=dt.date(2025,6,1),
    LAI_max=6.0, t_lag=10, t_close=35, LAI_hc=6.0, Cs=200, Ca=200,
    p_S1=1.0, p_S2=0.6, p_S3=0.4, p_S4=0.2,
    debug=False
) -> pd.DataFrame:

    sow = _safe_date(sow_date)
    start = sow - dt.timedelta(days=int(preR_days_before))
    end = dt.date(sow.year + int(nyears) - 1, 12, 1)
    meteo = synthetic_meteo(start, end, seed)

    Sq = float(seed_bank0)
    TTw = 0.0
    W = [0.0, 0.0, 0.0, 0.0, 0.0]
    out = []

    preR_date = sow - dt.timedelta(days=int(preR_days_before))
    postR_date = sow + dt.timedelta(days=int(postR_days_after))
    gram_date = sow + dt.timedelta(days=int(gram_days_after))
    preR_window = { preR_date + dt.timedelta(days=i) for i in range(int(preR_residual)) }
    postR_window = { postR_date + dt.timedelta(days=i) for i in range(int(postR_residual)) }
    gram_window = { gram_date + dt.timedelta(days=i) for i in range(int(gram_residual)) }

    Th = [70.0, 280.0, 400.0, 300.0]  # °Cd

    for _, row in meteo.iterrows():
        date = pd.to_datetime(row["date"]).date()
        dss = (date - sow).days
        Tmean = (_as_float(row["tmin"]) + _as_float(row["tmax"])) / 2.0
        TTw += max(Tmean - float(Tb), 0.0)

        Ciec_t, LAI_t, _ = ciec_calendar(dss, float(LAI_max), int(t_lag), int(t_close),
                                         float(LAI_hc), float(Cs), float(Ca))
        E_t = emergence_simple(TTw, _as_float(row["prec"]))

        Wk = sum(np.array(W) * np.array([0.15, 0.30, 0.60, 1.0, 0.0]))
        surv_intra = 1.0 - min(Wk / float(K), 1.0)

        sup_S1 = (1.0 - Ciec_t)**float(p_S1)
        I1_t = max(0.0, Sq * E_t * surv_intra * sup_S1)

        Ct_post = [0.0, 0.0, 0.0, 0.0, 0.0]
        if date in preR_window:
            Ct_post[0] = max(Ct_post[0], float(preR_eff)/100.0)
        if date in postR_window:
            for i in range(4):
                Ct_post[i] = max(Ct_post[i], float(postR_eff)/100.0)
        if date in gram_window:
            for i in range(3):
                Ct_post[i] = max(Ct_post[i], float(gram_eff)/100.0)

        W_ctrl = [max(0.0, w * (1.0 - c)) for w, c in zip(W, Ct_post)]

        sup_S2 = (1.0 - Ciec_t)**float(p_S2)
        sup_S3 = (1.0 - Ciec_t)**float(p_S3)
        sup_S4 = (1.0 - Ciec_t)**float(p_S4)
        W_ctrl[1] *= sup_S2
        W_ctrl[2] *= sup_S3
        W_ctrl[3] *= sup_S4

        O1 = I1_t if TTw >= Th[0] else 0.0
        O2 = W_ctrl[1] if TTw >= (Th[0]+Th[1]) else 0.0
        O3 = W_ctrl[2] if TTw >= (Th[0]+Th[1]+Th[2]) else 0.0
        O4 = W_ctrl[3] if TTw >= (Th[0]+Th[1]+Th[2]+Th[3]) else 0.0

        W1 = max(0.0, W_ctrl[0] + I1_t - O1)
        W2 = max(0.0, W_ctrl[1] + O1 - O2)
        W3 = max(0.0, W_ctrl[2] + O2 - O3)
        W4 = max(0.0, W_ctrl[3] + O3 - O4)
        W5 = max(0.0, W_ctrl[4] + O4)
        W = [W1, W2, W3, W4, W5]

        out.append({
            "date": date, "days_since_sow": int(dss),
            "TTw": float(TTw),
            "LAI": float(LAI_t), "Ciec": float(Ciec_t),
            "E_t": float(E_t), "I1_t": float(I1_t),
            "W1": float(W1), "W2": float(W2), "W3": float(W3), "W4": float(W4), "W5": float(W5),
            "sup_S1": float(sup_S1), "sup_S2": float(sup_S2), "sup_S3": float(sup_S3), "sup_S4": float(sup_S4),
            "Ct_post_sum": float(sum(Ct_post)),
        })

    df = pd.DataFrame(out)
    # garantizar tipos correctos
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])
    return df

# ---------- RUN ----------
if run_btn:
    try:
        df = simulate(
            nyears, seed_bank0, K, Tb, sim_seed,
            preR_days_before, preR_eff_S1S2, preR_residual,
            postR_days_after, postR_eff_S1S4, postR_residual,
            gram_days_after, gram_eff_S1S3, gram_residual,
            sow_date,
            LAI_max, t_lag, t_close, LAI_hc, Cs, Ca,
            p_S1, p_S2, p_S3, p_S4,
            debug=debug
        )
        st.success(f"Listo: {len(df)} días simulados desde {sow_date}.")

        # Tabs
        t1, t2, t3 = st.tabs(["Densidades S1–S5", "Ciec & LAI & Emergencia", "Datos / Descargar"])

        with t1:
            fig = go.Figure()
            for s in ["W1","W2","W3","W4","W5"]:
                fig.add_trace(go.Scatter(x=df["date"], y=df[s], mode="lines", name=s))
            fig.update_layout(
                title="Densidad por estadio (S1–S5)",
                xaxis_title="Fecha", yaxis_title="pl·m⁻²",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df["date"], y=df["Ciec"], name="Ciec"))
            fig2.add_trace(go.Scatter(x=df["date"], y=df["LAI"], name="LAI", yaxis="y2"))
            fig2.add_trace(go.Scatter(x=df["date"], y=df["E_t"]*100.0, name="Emergencia diaria (%)", yaxis="y2"))
            fig2.update_layout(
                title="Ciec(t), LAI(t) y Emergencia (%)",
                xaxis=dict(title="Fecha"),
                yaxis=dict(title="Ciec (0–1)"),
                yaxis2=dict(title="LAI / Emergencia (%)", overlaying="y", side="right", showgrid=False),
                template="plotly_white"
            )
            st.plotly_chart(fig2, use_container_width=True)

        with t3:
            st.dataframe(df.tail(80), use_container_width=True)
            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Descargar CSV", csv, "weedcrop_OK.csv", "text/csv")

        if debug:
            st.divider()
            st.caption("🔎 Debug info")
            st.code(df.head(5).to_string(index=False))

    except Exception as e:
        st.error("❌ Ocurrió un error en la simulación. Abajo van detalles útiles.")
        st.exception(e)
        st.stop()
else:
    st.info("Configura parámetros y presioná ▶ Ejecutar simulación.")

