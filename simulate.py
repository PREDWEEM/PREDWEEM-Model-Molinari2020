# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 WeedCropSystem — v3.17.2-final
# ---------------------------------------------------------------
# ✔ Cruce dentro del Período Crítico (8/oct–4/nov)
# ✔ Pérdida ≤ Lmax (saturada)
# ✔ Línea vertical sin errores (usando add_shape)
# ✔ Slider de sensibilidad pos-PC
# ===============================================================

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date as _date, timedelta

# ===============================================================
# Meteorología sintética
# ===============================================================
def synthetic_meteo(start, end, seed=42):
    rng = np.random.default_rng(int(seed))
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.to_numpy()
    tmean = 12 + 8*np.sin(2*np.pi*(doy-170)/365.0) + rng.normal(0,1.5,len(dates))
    tmin = tmean - (3 + rng.normal(0,0.8,len(dates)))
    tmax = tmean + (6 + rng.normal(0,0.8,len(dates)))
    prec = rng.choice([0,0,0,0,3,8,15], size=len(dates),
                      p=[0.55,0.15,0.10,0.05,0.07,0.05,0.03])
    return pd.DataFrame({"date":dates,"tmin":tmin,"tmax":tmax,"prec":prec})

# ===============================================================
# Auxiliares fisiológicas
# ===============================================================
def emergence_simple(TT, prec):
    base = 1.0/(1.0 + np.exp(-(TT-300)/40.0))
    pulse = 0.002 if float(prec) >= 5.0 else 0.0
    return float(min(base*0.003 + pulse, 0.02))

def lai_logistic_by_day(days_since_sow, LAI_max, t_lag, t_close):
    t50 = max(int(t_close), 1)
    eps = 0.05
    denom = (int(t_lag) - t50)
    if abs(denom) < 1e-6: denom = -1.0
    k = -np.log(1/eps - 1) / denom
    LAI = LAI_max / (1.0 + np.exp(-k*(float(days_since_sow) - t50)))
    return float(max(0.0, min(LAI, LAI_max))), float(k)

def ciec_calendar(days_since_sow, LAI_max, t_lag, t_close, LAI_hc, Cs, Ca):
    LAI, _ = lai_logistic_by_day(days_since_sow, LAI_max, t_lag, t_close)
    ratio = (float(Cs)/max(float(Ca),1e-6))
    Ciec = min((LAI / max(float(LAI_hc),1e-6)) * ratio, 1.0)
    return float(Ciec), LAI

def _date_range(start_date, days):
    return {start_date + dt.timedelta(days=i) for i in range(int(days))}

# ===============================================================
# Simulador principal
# ===============================================================
def simulate_with_controls(
    nyears=1, seed_bank0=4500, K=250, Tb=0.0, seed=42,
    sow_date=dt.date(2025,6,1),
    LAI_max=6.0, t_lag=10, t_close=35, LAI_hc=6.0, Cs=200, Ca=200,
    p_S1=1.0, p_S2=0.6, p_S3=0.4, p_S4=0.2,
    w_S1=0.15, w_S2=0.30, w_S3=0.60, w_S4=1.00,
    alpha=0.9782, Lmax=83.77, GY_pot=6000.0,
    preR_eff=90, preemR_eff=70, postR_eff=85, gram_eff=80,
    preR_residual=30, preemR_residual=30, postR_residual=30,
    gram_residual_forward=11,
    preR_date=None, preemR_date=None, postR_date=None, gram_date=None,
    enforce_rules=True,
    sens_factor_pc=5.0, sens_factor_postpc=0.5
):
    sow = pd.to_datetime(sow_date).date()
    start = sow - dt.timedelta(days=90)
    end = dt.date(sow.year + int(nyears) - 1, 12, 1)
    meteo = synthetic_meteo(start, end, seed)

    # Ventanas fenológicas
    preR_rng  = (sow - dt.timedelta(days=30), sow - dt.timedelta(days=14))
    preem_rng = (sow, sow + dt.timedelta(days=10))
    postR_rng = (sow + dt.timedelta(days=20), sow + dt.timedelta(days=180))
    gram_rng  = (sow + dt.timedelta(days=25), sow + dt.timedelta(days=35))
    if enforce_rules:
        if preR_date and not (preR_rng[0] <= preR_date <= preR_rng[1]): return None
        if preemR_date and not (preem_rng[0] <= preemR_date <= preem_rng[1]): return None
        if postR_date and not (postR_rng[0] <= postR_date <= postR_rng[1]): return None
        if gram_date and not (gram_rng[0] <= gram_date <= gram_rng[1]): return None

    preR_window   = set() if preR_date is None else _date_range(preR_date, preR_residual)
    preemR_window = set() if preemR_date is None else _date_range(preemR_date, preemR_residual)
    postR_window  = set() if postR_date is None else _date_range(postR_date, postR_residual)
    gram_window   = set() if gram_date is None else _date_range(gram_date, gram_residual_forward)

    EPS = 1e-9
    Sq, TTw = float(seed_bank0), 0.0
    W = [0,0,0,0,0]
    Th = [70, 280, 400, 300]
    rows = []

    for _, row in meteo.iterrows():
        date = pd.to_datetime(row["date"]).date()
        if date < sow: continue

        dss = (date - sow).days
        Tmean = (row["tmin"] + row["tmax"]) / 2
        TTw += max(Tmean - Tb, 0)
        Ciec_t, LAI_t = ciec_calendar(dss, LAI_max, t_lag, t_close, LAI_hc, Cs, Ca)
        E_t = 2.0 * emergence_simple(TTw, row["prec"])

        # Supresión y crecimiento
        Wk = sum(np.array(W)*np.array([0.15,0.3,1.0,1.2,0.0]))
        surv_intra = 1 - min(Wk/K,1)
        I1_t = max(0, Sq * E_t * surv_intra * (1-Ciec_t)**p_S1)
        Wc = W.copy()
        Wc[1]*=(1-Ciec_t)**p_S2
        Wc[2]*=(1-Ciec_t)**p_S3
        Wc[3]*=(1-Ciec_t)**p_S4

        # Controles
        def _apply(eff, states):
            if eff<=0: return
            for i,s in enumerate(["S1","S2","S3","S4"]):
                if s in states: Wc[i]*=(1-eff/100.0)
        if date in preR_window:   _apply(preR_eff,["S1","S2"])
        if date in preemR_window: _apply(preemR_eff,["S1","S2"])
        if date in postR_window:  _apply(postR_eff,["S1","S2","S3","S4"])
        if date in gram_window:   _apply(gram_eff,["S1","S2","S3"])

        # Transiciones térmicas
        O1 = I1_t if TTw>=Th[0] else 0
        O2 = Wc[1] if TTw>=sum(Th[:2]) else 0
        O3 = Wc[2] if TTw>=sum(Th[:3]) else 0
        O4 = Wc[3] if TTw>=sum(Th[:4]) else 0
        W1 = max(0,Wc[0]+I1_t-O1)
        W2 = max(0,Wc[1]+O1-O2)
        W3 = max(0,Wc[2]+O2-O3)
        W4 = max(0,Wc[3]+O3-O4)
        W5 = max(0,Wc[4]+O4)
        W=[W1,W2,W3,W4,W5]
        rows.append({"date":date,"days_since_sow":dss,"TTw":TTw,"Ciec":Ciec_t,"LAI":LAI_t,
                    "W1":W1,"W2":W2,"W3":W3,"W4":W4})

    df=pd.DataFrame(rows)
    if df.empty: return df
    df["WC"]=w_S1*df["W1"]+w_S2*df["W2"]+w_S3*df["W3"]+w_S4*df["W4"]

    # --- AUC y sensibilidad
    PC_ini, PC_fin = dt.date(sow.year,10,8), dt.date(sow.year,11,4)
    wc=df["WC"].to_numpy()
    auc_base=np.concatenate(([0.0],np.cumsum(0.5*(wc[1:]+wc[:-1]))))
    df["AUC_base"]=auc_base
    sens=np.ones(len(df))
    sens[(df["date"]>=PC_ini)&(df["date"]<=PC_fin)] = sens_factor_pc
    sens[df["date"]>PC_fin] = sens_factor_postpc
    wcw=wc*sens
    auc_weighted=np.concatenate(([0.0],np.cumsum(0.5*(wcw[1:]+wcw[:-1]))))
    df["AUC_weighted"]=auc_weighted

    def _loss(x,a,L): return (a*x)/(1+(a*x/L))
    df["Loss_running_%"]=[_loss(x,alpha,Lmax) for x in df["AUC_weighted"]]
    df["Yield_relative_%"]=np.clip(100-df["Loss_running_%"],0,100)
    df["Yield_abs_kg_ha"]=GY_pot*(df["Yield_relative_%"]/100)
    dloss=np.gradient(df["Loss_running_%"].to_numpy())
    idx_pc=(df["date"]>=PC_ini)&(df["date"]<=PC_fin)
    i_peak=int(np.argmax(dloss[idx_pc]))+np.where(idx_pc)[0][0]
    peak=df["date"].iloc[i_peak]
    df.attrs.update({"PC_ini":PC_ini,"PC_fin":PC_fin,"peak":peak,
                     "AUCw_final":float(df["AUC_weighted"].iloc[-1]),
                     "Loss_final":float(df["Loss_running_%"].iloc[-1])})
    return df

# ===============================================================
# Streamlit UI
# ===============================================================
st.set_page_config(page_title="WeedCropSystem v3.17.2",layout="wide")
st.title("🌾 WeedCropSystem v3.17.2 — Cruce en PC + Pérdida saturada")

st.sidebar.header("⚙️ Parámetros base")
nyears=st.sidebar.slider("Años a simular",1,3,1)
seed_bank0=st.sidebar.number_input("Banco inicial (semillas·m⁻²)",0,50000,4500)
K=st.sidebar.number_input("Capacidad de carga K",50,2000,250)
Tb=st.sidebar.number_input("Temp. base Tb (°C)",0.0,15.0,0.0,0.5)
sim_seed=st.sidebar.number_input("Semilla clima",0,999999,42)
sow_date=st.sidebar.date_input("Fecha siembra",_date(2025,6,1))

st.sidebar.subheader("🌿 Canopia")
LAI_max=st.sidebar.slider("LAI_max",2.0,10.0,6.0,0.1)
t_lag=st.sidebar.slider("t_lag",0,60,10)
t_close=st.sidebar.slider("t_close",10,100,35)
LAI_hc=st.sidebar.slider("LAI_hc",2.0,10.0,6.0,0.1)
Cs=st.sidebar.number_input("Cs",50,800,200)
Ca=st.sidebar.number_input("Ca",30,800,200)

st.sidebar.subheader("⚖️ Supresión")
p_S1=st.sidebar.slider("S1",0.0,2.0,1.0,0.1)
p_S2=st.sidebar.slider("S2",0.0,2.0,0.6,0.1)
p_S3=st.sidebar.slider("S3",0.0,2.0,0.4,0.1)
p_S4=st.sidebar.slider("S4",0.0,2.0,0.2,0.1)

st.sidebar.subheader("🌾 Pérdida y sensibilidad")
GY_pot=st.sidebar.number_input("Rinde potencial (kg/ha)",1000,15000,6000,100)
alpha=st.sidebar.number_input("α",0.1,3.0,0.9782,0.01)
Lmax=st.sidebar.number_input("Lmax (%)",10.0,100.0,83.77,0.1)
sens_factor_pc=st.sidebar.slider("Sensibilidad PC (×)",1.0,7.0,5.0,0.5)
sens_factor_postpc=st.sidebar.slider("Sensibilidad pos-PC (×)",0.0,2.0,0.5,0.1)

if st.sidebar.button("▶ Ejecutar simulación"):
    df=simulate_with_controls(nyears=nyears,seed_bank0=seed_bank0,K=K,Tb=Tb,seed=sim_seed,
        sow_date=sow_date,LAI_max=LAI_max,t_lag=t_lag,t_close=t_close,LAI_hc=LAI_hc,
        Cs=Cs,Ca=Ca,p_S1=p_S1,p_S2=p_S2,p_S3=p_S3,p_S4=p_S4,
        alpha=alpha,Lmax=Lmax,GY_pot=GY_pot,
        sens_factor_pc=sens_factor_pc,sens_factor_postpc=sens_factor_postpc)
    if df is None or df.empty:
        st.error("❌ Simulación vacía o fuera de reglas.")
    else:
        st.success(f"✅ Simulación exitosa — {len(df)} días")
        pc_ini,pc_fin,peak=df.attrs["PC_ini"],df.attrs["PC_fin"],df.attrs["peak"]

        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df["date"],y=df["Yield_abs_kg_ha"],name="Rinde (kg/ha)"))
        fig.add_trace(go.Scatter(x=df["date"],y=df["Loss_running_%"],name="Pérdida (%)",yaxis="y2"))
        fig.add_vrect(x0=pc_ini,x1=pc_fin,fillcolor="LightSalmon",opacity=0.25,line_width=0)
        # línea segura
        fig.add_shape(type="line",x0=peak,x1=peak,y0=0,y1=1,xref="x",yref="paper",
                      line=dict(color="red",width=2,dash="dash"))
        fig.add_annotation(x=peak,y=1.02,xref="x",yref="paper",
                           text="Máx. dPérdida",showarrow=False,font=dict(color="red",size=10))
        fig.update_layout(title=f"Rinde y Pérdida — pico {peak}",
                          xaxis_title="Fecha",yaxis_title="Rinde (kg/ha)",
                          yaxis2=dict(title="Pérdida (%)",overlaying="y",side="right",range=[0,40]),
                          template="plotly_white")
        st.plotly_chart(fig,use_container_width=True)

        st.metric("💰 Rinde final",f"{df['Yield_abs_kg_ha'].iloc[-1]:.0f} kg/ha")
        st.metric("📉 Pérdida final",f"{df.attrs['Loss_final']:.2f}%")
        st.metric("🧮 AUC ponderado final",f"{df.attrs['AUCw_final']:.2f}")

        st.download_button("📥 Descargar CSV",
            df.to_csv(index=False).encode(),"simulacion_v3172.csv","text/csv")
