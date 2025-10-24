# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 WeedCropSystem — Streamlit (Molinari et al. 2020)
# ---------------------------------------------------------------
# Simulación diaria con controles herbicidas (modo simple)
# - Banco de semillas (Eq.1)
# - Emergencia (placeholder ANN)
# - Fenología / TT (Eq.4–6)
# - Competencia intra e interespecífica (Eq.7–9)
# - Control químico (Eq.14)
# ---------------------------------------------------------------
# Requiere: streamlit, numpy, pandas, plotly
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as go

# ---------- CONFIGURACIÓN ----------
st.set_page_config(page_title="🌾 WeedCropSystem — Controles herbicidas", layout="wide")
st.title("🌾 WeedCropSystem — Simulación con controles herbicidas (modo simple)")

# ---------- SIDEBAR ----------
st.sidebar.header("Configuración del escenario")
nyears = st.sidebar.slider("Años a simular", 1, 10, 3)
seed_bank0 = st.sidebar.number_input("Banco inicial (semillas · m⁻²)", 0, 10000, 4500)
Tb = st.sidebar.number_input("Temp. base Tb (°C)", 0.0, 10.0, 0.0)
K = st.sidebar.number_input("Capacidad de carga K (pl · m⁻²)", 50, 1000, 250)
sim_seed = st.sidebar.number_input("Semilla aleatoria", 0, 9999, 42)

st.sidebar.divider()
st.sidebar.subheader("🧪 Controles herbicidas (modo simple)")

# --- Presiembra residual ---
st.sidebar.markdown("**Presiembra residual (preR)**")
preR_days_before = st.sidebar.number_input("Días antes de siembra", 0, 30, 14)
preR_eff_S1S2 = st.sidebar.slider("Eficacia S1–S2 (%)", 0, 100, 90)
preR_residual = st.sidebar.slider("Duración residual (días)", 0, 60, 30)

# --- Preemergente / post temprana ---
st.sidebar.markdown("**Preemergente / Postemergente (postR)**")
postR_days_after = st.sidebar.number_input("Días después de siembra", 0, 60, 25)
postR_eff_S1S4 = st.sidebar.slider("Eficacia S1–S4 (%)", 0, 100, 85)

# --- Graminicida ---
st.sidebar.markdown("**Graminicida (gram)**")
gram_days_after = st.sidebar.number_input("Días después de siembra (graminicida)", 0, 60, 10)
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

def Ciec_simple(TT_c, Cs=200, Ca=200, LAI_hc=6.0):
    LAI = max(0, 0.008*TT_c - 0.000004*(TT_c**2))
    return min((LAI/LAI_hc)*(Cs/Ca), 1.0), LAI

# ---------- SIMULACIÓN PRINCIPAL ----------
def simulate(nyears=3, seed_bank0=4500, K=250, Tb=0.0, seed=42,
             preR_days_before=14, preR_eff=90, preR_residual=30,
             postR_days_after=25, postR_eff=85,
             gram_days_after=10, gram_eff=80):

    start_year = 2021
    sow = dt.date(start_year,6,1)
    start = sow - dt.timedelta(days=preR_days_before)
    end = dt.date(start_year+nyears-1,12,1)
    meteo = synthetic_meteo(start, end, seed)

    Sq = seed_bank0
    TTw, TTc = 0.0, 0.0
    W = [0,0,0,0,0]
    out = []

    # Fechas de control absolutas
    preR_date = sow - dt.timedelta(days=preR_days_before)
    postR_date = sow + dt.timedelta(days=postR_days_after)
    gram_date = sow + dt.timedelta(days=gram_days_after)
    preR_window = [preR_date + dt.timedelta(days=i) for i in range(preR_residual)]

    for i,row in meteo.iterrows():
        date = row["date"].date()
        Tmean = (row["tmin"]+row["tmax"])/2
        TTw += max(Tmean - Tb, 0)
        TTc += max(Tmean - Tb, 0)

        # Competencia del cultivo
        Ciec_t, LAI_t = Ciec_simple(TTc)
        # Emergencia
        E_t = emergence_simple(TTw, row["prec"])

        # Intraespecífica
        Wk = sum(np.array(W)*np.array([0.15,0.30,0.60,1.0,0.0]))
        surv_intra = 1 - min(Wk/K, 1.0)

        # Presiembra residual activo?
        Ct01 = preR_eff/100 if date in preR_window else 0

        # Ingreso a s1 (Eq.3)
        I1_t = Sq * E_t * surv_intra * (1 - Ciec_t) * (1 - Ct01)

        # Transiciones simplificadas (TT)
        Th = [70,280,400,300]
        O1 = I1_t if TTw>=Th[0] else 0
        O2 = W[1] if TTw>=sum(Th[:2]) else 0
        O3 = W[2] if TTw>=sum(Th[:3]) else 0
        O4 = W[3] if TTw>=sum(Th[:4]) else 0

        # Mortalidad por controles post y graminicida
        Ct_post = [0,0,0,0,0]
        if date == postR_date:
            Ct_post = [postR_eff/100]*4 + [0]  # actúa S1–S4
        if date == gram_date:
            Ct_post = [gram_eff/100]*3 + [0,0] # actúa S1–S3

        # Aplicar mortalidad (Eq.14)
        W_ctrl = [w*(1-c) for w,c in zip(W,Ct_post)]

        # Actualización de estados
        W1 = max(0, W_ctrl[0]+I1_t-O1)
        W2 = max(0, W_ctrl[1]+O1-O2)
        W3 = max(0, W_ctrl[2]+O2-O3)
        W4 = max(0, W_ctrl[3]+O3-O4)
        W5 = max(0, W_ctrl[4]+O4)
        W = [W1,W2,W3,W4,W5]

        out.append({
            "date":date, "TTw":TTw, "TTc":TTc,
            "LAI":LAI_t, "Ciec":Ciec_t, "E_t":E_t, "I1_t":I1_t,
            "W1":W1, "W2":W2, "W3":W3, "W4":W4, "W5":W5,
            "Ct_post_sum":sum(Ct_post), "Ct01":Ct01
        })
    return pd.DataFrame(out)

# ---------- EJECUCIÓN ----------
if run_btn:
    df = simulate(nyears, seed_bank0, K, Tb, sim_seed,
                  preR_days_before, preR_eff_S1S2, preR_residual,
                  postR_days_after, postR_eff_S1S4,
                  gram_days_after, gram_eff_S1S3)

    st.success(f"Simulación completada ({len(df)} días)")

    # --- Gráficos ---
    tab1, tab2, tab3 = st.tabs(["Densidades S1–S5","Supresión y Emergencia","Controles y Datos"])

    with tab1:
        fig = go.Figure()
        for s in ["W1","W2","W3","W4","W5"]:
            fig.add_trace(go.Scatter(x=df["date"], y=df[s], mode="lines", name=s))
        fig.update_layout(title="Densidad por estadio (S1–S5)",
                          xaxis_title="Fecha", yaxis_title="pl · m⁻²",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["date"], y=df["Ciec"], name="Ciec", line=dict(color="green")))
        fig2.add_trace(go.Scatter(x=df["date"], y=df["E_t"]*100, name="Emergencia diaria (%)", yaxis="y2", line=dict(color="orange")))
        fig2.update_layout(title="Supresión del cultivo y emergencia diaria",
                           xaxis_title="Fecha",
                           yaxis=dict(title="Ciec", color="green"),
                           yaxis2=dict(title="Emergencia diaria (%)", overlaying="y", side="right", color="orange"),
                           template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.write("🧪 **Aplicaciones herbicidas**:")
        st.markdown(f"- Presiembra residual: {preR_eff_S1S2}% (duración {preR_residual} d) desde **{(dt.date(2021,6,1)-dt.timedelta(days=preR_days_before)).isoformat()}**")
        st.markdown(f"- Postemergente: {postR_eff_S1S4}% el **{(dt.date(2021,6,1)+dt.timedelta(days=postR_days_after)).isoformat()}**")
        st.markdown(f"- Graminicida: {gram_eff_S1S3}% el **{(dt.date(2021,6,1)+dt.timedelta(days=gram_days_after)).isoformat()}**")

        st.dataframe(df.tail(50), use_container_width=True)
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Descargar CSV", csv, "weedcrop_controls.csv", "text/csv")

else:
    st.info("Configura los parámetros y presiona ▶ Ejecutar simulación.")
