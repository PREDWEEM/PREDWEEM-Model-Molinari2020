# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 WeedCropSystem — Streamlit (Molinari et al. 2020)
# ---------------------------------------------------------------
# Ciec dinámico por días calendario + supresión multietapa
# - Fecha de siembra seleccionable
# - Supresión del cultivo S1–S4
# - Controles herbicidas simples
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as go

# ---------- CONFIGURACIÓN ----------
st.set_page_config(page_title="🌾 WeedCropSystem — Ciec dinámico", layout="wide")
st.title("🌾 WeedCropSystem — Ciec dinámico (calendario) + supresión S1–S4")

# ---------- PANEL LATERAL ----------
st.sidebar.header("Configuración del escenario")
nyears = st.sidebar.slider("Años a simular", 1, 10, 3)
seed_bank0 = st.sidebar.number_input("Banco inicial (semillas · m⁻²)", 0, 10000, 4500)
Tb = st.sidebar.number_input("Temp. base Tb (°C)", 0.0, 10.0, 0.0)
K = st.sidebar.number_input("Capacidad de carga K (pl · m⁻²)", 50, 1000, 250)
sim_seed = st.sidebar.number_input("Semilla aleatoria clima", 0, 9999, 42)

# ---------- FECHA DE SIEMBRA ----------
st.sidebar.divider()
st.sidebar.subheader("📅 Fechas del cultivo")
sow_date = st.sidebar.date_input("Fecha de siembra", dt.date(2025, 6, 1))

# ---------- PARÁMETROS DEL CULTIVO ----------
st.sidebar.divider()
st.sidebar.subheader("🌿 Canopia (LAI logístico por días calendario)")
LAI_max = st.sidebar.slider("LAI_max", 2.0, 10.0, 6.0, 0.1)
t_lag = st.sidebar.slider("t_lag (días desde siembra)", 0, 60, 10)
t_close = st.sidebar.slider("t_close (días desde siembra, LAI≈0.5·max)", 10, 90, 35)
LAI_hc = st.sidebar.slider("LAI_hc (referencia competitiva)", 3.0, 10.0, 6.0, 0.1)
Cs = st.sidebar.number_input("Cs (estándar, pl·m⁻²)", 50, 600, 200)
Ca = st.sidebar.number_input("Ca (real, pl·m⁻²)", 30, 600, 200)

# ---------- SUPRESIÓN POR ESTADIO ----------
st.sidebar.divider()
st.sidebar.subheader("⚖️ Supresión por estadio (exponente de (1−Ciec))")
p_S1 = st.sidebar.slider("Exponente S1", 0.0, 1.5, 1.0, 0.1)
p_S2 = st.sidebar.slider("Exponente S2", 0.0, 1.5, 0.6, 0.1)
p_S3 = st.sidebar.slider("Exponente S3", 0.0, 1.5, 0.4, 0.1)
p_S4 = st.sidebar.slider("Exponente S4", 0.0, 1.5, 0.2, 0.1)

# ---------- CONTROLES HERBICIDAS ----------
st.sidebar.divider()
st.sidebar.subheader("🧪 Controles herbicidas")

st.sidebar.markdown("**Presiembra residual (preR)**")
preR_days_before = st.sidebar.number_input("Días antes de siembra", 0, 30, 14)
preR_eff_S1S2 = st.sidebar.slider("Eficacia S1–S2 (%)", 0, 100, 90)
preR_residual = st.sidebar.slider("Duración residual (días)", 0, 60, 30)

st.sidebar.markdown("**Postemergente (postR)**")
postR_days_after = st.sidebar.number_input("Días después de siembra", 0, 60, 25)
postR_eff_S1S4 = st.sidebar.slider("Eficacia S1–S4 (%)", 0, 100, 85)

st.sidebar.markdown("**Graminicida (gram)**")
gram_days_after = st.sidebar.number_input("Días después de siembra", 0, 60, 10)
gram_eff_S1S3 = st.sidebar.slider("Eficacia S1–S3 (%)", 0, 100, 80)

run_btn = st.sidebar.button("▶ Ejecutar simulación")

# ---------- FUNCIONES AUXILIARES ----------
def synthetic_meteo(start, end, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.to_numpy()
    tmean = 12 + 8*np.sin(2*np.pi*(doy-170)/365.0) + rng.normal(0,1.5,len(dates))
    tmin = tmean - (3 + rng.normal(0,0.8,len(dates)))
    tmax = tmean + (6 + rng.normal(0,0.8,len(dates)))
    prec = rng.choice([0,0,0,0,3,8,15], size=len(dates),
                      p=[0.55,0.15,0.10,0.05,0.07,0.05,0.03])
    return pd.DataFrame({"date":dates,"tmin":tmin,"tmax":tmax,"prec":prec})

def emergence_simple(TT, prec):
    base = 1/(1+np.exp(-(TT-300)/40))
    pulse = 0.002 if prec>=5 else 0
    return min(base*0.003 + pulse, 0.02)

def lai_logistic_by_day(days_since_sow, LAI_max, t_lag, t_close):
    t50 = max(t_close, 1)
    eps = 0.05
    denom = (t_lag - t50)
    if abs(denom) < 1e-6:
        denom = -1.0
    k = -np.log(1/eps - 1) / denom
    LAI = LAI_max / (1.0 + np.exp(-k*(days_since_sow - t50)))
    return float(max(0.0, min(LAI, LAI_max))), k

def ciec_calendar(days_since_sow, LAI_max, t_lag, t_close, LAI_hc, Cs, Ca):
    LAI, k = lai_logistic_by_day(days_since_sow, LAI_max, t_lag, t_close)
    ratio = (Cs / max(Ca, 1e-6))
    Ciec = min((LAI / max(LAI_hc, 1e-6)) * ratio, 1.0)
    return Ciec, LAI, k

# ---------- SIMULADOR ----------
def simulate(nyears=3, seed_bank0=4500, K=250, Tb=0.0, seed=42,
             preR_days_before=14, preR_eff=90, preR_residual=30,
             postR_days_after=25, postR_eff=85,
             gram_days_after=10, gram_eff=80,
             sow_date=dt.date(2025,6,1),
             LAI_max=6.0, t_lag=10, t_close=35, LAI_hc=6.0, Cs=200, Ca=200,
             p_S1=1.0, p_S2=0.6, p_S3=0.4, p_S4=0.2):

    sow = sow_date
    start = sow - dt.timedelta(days=preR_days_before)
    end = dt.date(sow.year+nyears-1, 12, 1)
    meteo = synthetic_meteo(start, end, seed)

    Sq = seed_bank0
    TTw = 0.0
    W = [0,0,0,0,0]
    out = []

    preR_date = sow - dt.timedelta(days=preR_days_before)
    postR_date = sow + dt.timedelta(days=postR_days_after)
    gram_date = sow + dt.timedelta(days=gram_days_after)
    preR_window = [preR_date + dt.timedelta(days=i) for i in range(preR_residual)]

    for _, row in meteo.iterrows():
        date = row["date"].date()
        dss = (date - sow).days
        Tmean = (row["tmin"] + row["tmax"]) / 2
        TTw += max(Tmean - Tb, 0)

        Ciec_t, LAI_t, k_lai = ciec_calendar(dss, LAI_max, t_lag, t_close, LAI_hc, Cs, Ca)
        E_t = emergence_simple(TTw, row["prec"])
        Wk = sum(np.array(W)*np.array([0.15,0.30,0.60,1.0,0.0]))
        surv_intra = 1 - min(Wk/K, 1.0)

        Ct01 = preR_eff/100 if date in preR_window else 0
        sup_S1 = (1.0 - Ciec_t)**p_S1
        I1_t = Sq * E_t * surv_intra * sup_S1 * (1 - Ct01)

        Th = [70,280,400,300]
        O1 = I1_t if TTw>=Th[0] else 0
        O2 = W[1] if TTw>=sum(Th[:2]) else 0
        O3 = W[2] if TTw>=sum(Th[:3]) else 0
        O4 = W[3] if TTw>=sum(Th[:4]) else 0

        Ct_post = [0,0,0,0,0]
        if date == postR_date:
            Ct_post = [postR_eff/100]*4 + [0]
        if date == gram_date:
            Ct_post = [gram_eff/100]*3 + [0,0]
        W_ctrl = [w*(1-c) for w,c in zip(W,Ct_post)]

        sup_S2 = (1.0 - Ciec_t)**p_S2
        sup_S3 = (1.0 - Ciec_t)**p_S3
        sup_S4 = (1.0 - Ciec_t)**p_S4
        W_ctrl[1] *= sup_S2
        W_ctrl[2] *= sup_S3
        W_ctrl[3] *= sup_S4

        W1 = max(0, W_ctrl[0] + I1_t - O1)
        W2 = max(0, W_ctrl[1] + O1 - O2)
        W3 = max(0, W_ctrl[2] + O2 - O3)
        W4 = max(0, W_ctrl[3] + O3 - O4)
        W5 = max(0, W_ctrl[4] + O4)
        W = [W1,W2,W3,W4,W5]

        out.append({
            "date": date, "days_since_sow": dss, "TTw": TTw,
            "LAI": LAI_t, "Ciec": Ciec_t,
            "E_t": E_t, "I1_t": I1_t,
            "W1": W1, "W2": W2, "W3": W3, "W4": W4, "W5": W5,
            "Ct01": Ct01, "Ct_post_sum": sum(Ct_post),
            "sup_S1": sup_S1, "sup_S2": sup_S2, "sup_S3": sup_S3, "sup_S4": sup_S4
        })

    return pd.DataFrame(out)

# ---------- EJECUCIÓN ----------
if run_btn:
    df = simulate(nyears, seed_bank0, K, Tb, sim_seed,
                  preR_days_before, preR_eff_S1S2, preR_residual,
                  postR_days_after, postR_eff_S1S4,
                  gram_days_after, gram_eff_S1S3,
                  sow_date,
                  LAI_max, t_lag, t_close, LAI_hc, Cs, Ca,
                  p_S1, p_S2, p_S3, p_S4)

    st.success(f"Simulación completada — {len(df)} días desde {sow_date}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Densidades S1–S5", "Ciec & LAI & Emergencia",
        "Controles & Supresión", "Datos / Descargar"
    ])

    with tab1:
        fig = go.Figure()
        for s in ["W1","W2","W3","W4","W5"]:
            fig.add_trace(go.Scatter(x=df["date"], y=df[s], mode="lines", name=s))
        fig.update_layout(title="Densidad por estadio (S1–S5)",
                          xaxis_title="Fecha", yaxis_title="pl · m⁻²",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # ✅ Bloque corregido (dos ejes)
    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["date"], y=df["Ciec"], name="Ciec", line=dict(color="green", width=2)))
        fig2.add_trace(go.Scatter(x=df["date"], y=df["LAI"], name="LAI", yaxis="y2", line=dict(color="blue", width=2, dash="dot")))
        fig2.add_trace(go.Scatter(x=df["date"], y=df["E_t"]*100, name="Emergencia diaria (%)", yaxis="y2", line=dict(color="orange", width=1.5)))
        fig2.update_layout(
            title="Ciec(t), LAI(t) y Emergencia diaria",
            xaxis=dict(title="Fecha"),
            yaxis=dict(title="Ciec (0–1)", color="green"),
            yaxis2=dict(title="LAI / Emergencia (%)", overlaying="y", side="right", showgrid=False, color="blue"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            template="plotly_white",
            margin=dict(l=60, r=80, t=60, b=40)
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown(f"- **Presiembra residual:** {preR_eff_S1S2}% desde {(sow_date - dt.timedelta(days=preR_days_before)).isoformat()} (duración {preR_residual} d)")
        st.markdown(f"- **Postemergente:** {postR_eff_S1S4}% el {(sow_date + dt.timedelta(days=postR_days_after)).isoformat()}")
        st.markdown(f"- **Graminicida:** {gram_eff_S1S3}% el {(sow_date + dt.timedelta(days=gram_days_after)).isoformat()}")

        fig3 = go.Figure()
        for s, col in zip(["S1","S2","S3","S4"], ["sup_S1","sup_S2","sup_S3","sup_S4"]):
            fig3.add_trace(go.Scatter(x=df["date"], y=df[col], name=f"(1−Ciec)^{s}"))
        fig3.update_layout(title="Factores de supresión diarios por estadio",
                           xaxis_title="Fecha", yaxis_title="Factor de supervivencia",
                           template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.dataframe(df.tail(80), use_container_width=True)
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Descargar CSV", csv, "weedcrop_ciec_v3_1.csv", "text/csv")

else:
    st.info("Configura los parámetros y presiona ▶ Ejecutar simulación.")
