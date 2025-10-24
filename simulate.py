# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 WeedCropSystem — Streamlit demo (Molinari et al. 2020 base)
# ---------------------------------------------------------------
# Simulación diaria (Pasos 0–4):
#  - Banco de semillas (Eq.1)
#  - Emergencia (placeholder ANN)
#  - Fenología / TT (Eq.4–6)
#  - Competencia intra e interespecífica (Eq.7–9)
# ---------------------------------------------------------------
# Requiere: streamlit, numpy, pandas, plotly
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as go

# ---------- PARÁMETROS INICIALES ----------
st.set_page_config(page_title="🌾 WeedCropSystem — Molinari2020 base", layout="wide")
st.title("🌾 WeedCropSystem — Simulación base (Pasos 0–4)")

# --- Panel lateral de configuración ---
st.sidebar.header("Configuración del escenario")
nyears = st.sidebar.slider("Años a simular", 1, 10, 5)
seed_bank0 = st.sidebar.number_input("Banco inicial (semillas · m⁻²)", 0, 10000, 4500)
Tb = st.sidebar.number_input("Temp. base Tb (°C)", 0.0, 10.0, 0.0)
K = st.sidebar.number_input("Capacidad de carga K (pl · m⁻²)", 50, 1000, 250)
sim_seed = st.sidebar.number_input("Semilla aleatoria", 0, 9999, 42)
run_btn = st.sidebar.button("▶ Ejecutar simulación")

# ---------- FUNCIONES DEL MODELO ----------
def synthetic_meteo(start, end, seed=42):
    """Clima sintético: tmax, tmin, precip."""
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
    """Emergencia diaria (placeholder ANN)"""
    base = 1/(1+np.exp(-(TT-300)/40))
    pulse = 0.002 if prec>=5 else 0
    return min(base*0.003 + pulse, 0.02)

def Ciec_simple(TT_c, Cs=200, Ca=200, LAI_hc=6.0):
    """Supresión del cultivo Eq.(9) con LAI parabólico."""
    LAI = max(0, 0.008*TT_c - 0.000004*(TT_c**2))
    return min((LAI/LAI_hc)*(Cs/Ca), 1.0), LAI

# ---------- SIMULACIÓN PRINCIPAL ----------
def simulate(nyears=5, seed_bank0=4500, K=250, Tb=0.0, seed=42):
    start_year = 2021
    sow = dt.date(start_year,6,1)
    harv = dt.date(start_year,12,1)
    start = sow
    end = dt.date(start_year+nyears-1,12,1)
    meteo = synthetic_meteo(start, end, seed)

    Sq = seed_bank0
    TTw, TTc = 0.0, 0.0
    W = [0,0,0,0,0]
    out = []

    for i,row in meteo.iterrows():
        date = row["date"].date()
        Tmean = (row["tmin"]+row["tmax"])/2
        TTw += max(Tmean - Tb, 0)
        TTc += max(Tmean - Tb, 0)

        # Ciec y LAI
        Ciec_t, LAI_t = Ciec_simple(TTc)
        # Emergencia (ANN placeholder)
        E_t = emergence_simple(TTw, row["prec"])

        # Intraespecífica (Eq.7–8 simplificada)
        Wk = sum(np.array(W)*np.array([0.15,0.30,0.60,1.0,0.0]))
        surv_intra = 1 - min(Wk/K, 1.0)

        # Ingreso a s1 (Eq.3 simplificada)
        I1_t = Sq * E_t * surv_intra * (1 - Ciec_t)

        # Transiciones (simplificadas por TT)
        Th = [70,280,400,300]
        O1 = I1_t if TTw>=Th[0] else 0
        O2 = W[1] if TTw>=sum(Th[:2]) else 0
        O3 = W[2] if TTw>=sum(Th[:3]) else 0
        O4 = W[3] if TTw>=sum(Th[:4]) else 0

        W1 = max(0, W[0]+I1_t-O1)
        W2 = max(0, W[1]+O1-O2)
        W3 = max(0, W[2]+O2-O3)
        W4 = max(0, W[3]+O3-O4)
        W5 = max(0, W[4]+O4)
        W = [W1,W2,W3,W4,W5]

        out.append({
            "date":date, "TTw":TTw, "TTc":TTc,
            "LAI":LAI_t, "Ciec":Ciec_t, "E_t":E_t, "I1_t":I1_t,
            "W1":W1, "W2":W2, "W3":W3, "W4":W4, "W5":W5
        })
    return pd.DataFrame(out)

# ---------- EJECUCIÓN ----------
if run_btn:
    df = simulate(nyears, seed_bank0, K, Tb, sim_seed)
    st.success(f"Simulación completada: {len(df)} días ({nyears} años)")

    # --- Gráficos ---
    tab1, tab2, tab3 = st.tabs(["Densidades por estadio","Supresión y emergencia","Datos"])

    with tab1:
        fig = go.Figure()
        for s in ["W1","W2","W3","W4","W5"]:
            fig.add_trace(go.Scatter(x=df["date"], y=df[s],
                                     mode="lines", name=s))
        fig.update_layout(title="Densidad por estadio (s=1–5)",
                          xaxis_title="Fecha", yaxis_title="pl·m⁻²",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["date"], y=df["Ciec"], name="Ciec", line=dict(color="green")))
        fig2.add_trace(go.Scatter(x=df["date"], y=df["E_t"]*100, name="Emergencia diaria (%)", yaxis="y2", line=dict(color="orange")))
        fig2.update_layout(title="Supresión del cultivo y emergencia diaria",
                           xaxis_title="Fecha",
                           yaxis=dict(title="Ciec (0–1)", color="green"),
                           yaxis2=dict(title="Emergencia diaria (%)", overlaying="y", side="right", color="orange"),
                           template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.dataframe(df.tail(50), use_container_width=True)
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Descargar resultados CSV", csv, "weedcrop_daily.csv", "text/csv")

else:
    st.info("Configura los parámetros y presiona ▶ Ejecutar simulación.")
