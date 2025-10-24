# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 WeedCropSystem — v3.9.1 (efecto de maleza visible)
# ---------------------------------------------------------------
# - Una única especie
# - Emergencia intensificada ×8
# - Índice de competencia WC → pérdida de rinde (α/Lmax fijos)
# - Nuevo tab "WC vs. Pérdida de rinde"
# ===============================================================

import sys, datetime as dt
import numpy as np
import pandas as pd

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

def simulate(
    nyears=3, seed_bank0=4500, K=250, Tb=0.0, seed=42,
    preR_days_before=14, preR_eff=90, preR_residual=30,
    postR_days_after=25, postR_eff=85, postR_residual=10,
    gram_days_after=10, gram_eff=80, gram_residual=7,
    sow_date=dt.date(2025,6,1),
    LAI_max=6.0, t_lag=10, t_close=35, LAI_hc=6.0, Cs=200, Ca=200,
    p_S1=1.0, p_S2=0.6, p_S3=0.4, p_S4=0.2,
    w_S1=0.15, w_S2=0.30, w_S3=0.60, w_S4=1.00,
    alpha=0.9782, Lmax=83.77, GY_pot=6000.0
):
    sow = pd.to_datetime(sow_date).date()
    start = sow - dt.timedelta(days=int(preR_days_before))
    end = dt.date(sow.year + int(nyears) - 1, 12, 1)
    meteo = synthetic_meteo(start, end, seed)

    preR_window = { (sow - dt.timedelta(days=int(preR_days_before))) + dt.timedelta(days=i)
                    for i in range(int(preR_residual)) }
    postR_window = { (sow + dt.timedelta(days=int(postR_days_after))) + dt.timedelta(days=i)
                     for i in range(int(postR_residual)) }
    gram_window = { (sow + dt.timedelta(days=int(gram_days_after))) + dt.timedelta(days=i)
                    for i in range(int(gram_residual)) }

    Sq = float(seed_bank0)
    TTw = 0.0
    W = [0,0,0,0,0]
    Th = [70, 280, 400, 300]
    out = []

    for _, row in meteo.iterrows():
        date = pd.to_datetime(row["date"]).date()
        dss = (date - sow).days
        Tmean = (float(row["tmin"])+float(row["tmax"])) / 2
        TTw += max(Tmean - float(Tb), 0)
        Ciec_t, LAI_t = ciec_calendar(dss, LAI_max, t_lag, t_close, LAI_hc, Cs, Ca)

        # 🔹 Aumentar emergencia ×8 para que haya efecto real de maleza
        E_t = 8.0 * emergence_simple(TTw, float(row["prec"]))

        Wk = sum(np.array(W)*np.array([0.15,0.3,0.6,1.0,0.0]))
        surv_intra = 1 - min(Wk/K,1)
        sup_S1 = (1-Ciec_t)**p_S1
        I1_t = max(0, Sq * E_t * surv_intra * sup_S1)

        eff_pre=[0]*5; eff_post=[0]*5; eff_gram=[0]*5
        if date in preR_window: eff_pre[0]=eff_pre[1]=preR_eff/100
        if date in postR_window: eff_post[:4]=[postR_eff/100]*4
        if date in gram_window: eff_gram[:3]=[gram_eff/100]*3

        W_ctrl=[w*(1-e) for w,e in zip(W,eff_pre)]
        W_ctrl=[w*(1-e) for w,e in zip(W_ctrl,eff_post)]
        W_ctrl=[w*(1-e) for w,e in zip(W_ctrl,eff_gram)]

        W_ctrl[1]*=(1-Ciec_t)**p_S2
        W_ctrl[2]*=(1-Ciec_t)**p_S3
        W_ctrl[3]*=(1-Ciec_t)**p_S4

        O1 = I1_t if TTw>=Th[0] else 0
        O2 = W_ctrl[1] if TTw>=sum(Th[:2]) else 0
        O3 = W_ctrl[2] if TTw>=sum(Th[:3]) else 0
        O4 = W_ctrl[3] if TTw>=sum(Th[:4]) else 0

        W1 = max(0,W_ctrl[0]+I1_t-O1)
        W2 = max(0,W_ctrl[1]+O1-O2)
        W3 = max(0,W_ctrl[2]+O2-O3)
        W4 = max(0,W_ctrl[3]+O3-O4)
        W5 = max(0,W_ctrl[4]+O4)
        W=[W1,W2,W3,W4,W5]

        out.append({"date":date,"days_since_sow":dss,"TTw":TTw,"Ciec":Ciec_t,"LAI":LAI_t,
                    "W1":W1,"W2":W2,"W3":W3,"W4":W4,"W5":W5})

    df=pd.DataFrame(out)
    df["W_total"]=df[["W1","W2","W3","W4"]].sum(axis=1)
    df["WC"]=w_S1*df["W1"]+w_S2*df["W2"]+w_S3*df["W3"]+w_S4*df["W4"]
    df["Yield_loss_%"]=(alpha*df["WC"])/(1+(alpha*df["WC"]/Lmax))
    df["Yield_relative_%"]=100-df["Yield_loss_%"]
    df["Yield_abs_kg_ha"]=GY_pot*(df["Yield_relative_%"]/100)
    return df

# ------------------- STREAMLIT -------------------
if "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv):
    import streamlit as st
    import plotly.graph_objects as go

    st.set_page_config(page_title="WeedCropSystem v3.9.1", layout="wide")
    st.title("🌾 WeedCropSystem — v3.9.1 (Efecto real de maleza sobre rinde)")

    # --- Panel lateral ---
    nyears = st.sidebar.slider("Años a simular", 1, 10, 3)
    seed_bank0 = st.sidebar.number_input("Banco inicial (semillas·m⁻²)", 0, 20000, 4500)
    K = st.sidebar.number_input("Cap. de carga K (pl·m⁻²)", 50, 2000, 250)
    Tb = st.sidebar.number_input("Temp. base Tb (°C)", 0.0, 15.0, 0.0, 0.5)
    sim_seed = st.sidebar.number_input("Semilla aleatoria clima", 0, 999999, 42)
    sow_date = st.sidebar.date_input("Fecha de siembra", dt.date(2025,6,1))

    st.sidebar.subheader("🌾 Rinde potencial del cultivo (GY_pot)")
    gy_option = st.sidebar.selectbox("Seleccionar cultivo:",
        ["Trigo (6000 kg/ha)", "Cebada (7000 kg/ha)", "Personalizado"])
    if gy_option=="Trigo (6000 kg/ha)": GY_pot=6000.0
    elif gy_option=="Cebada (7000 kg/ha)": GY_pot=7000.0
    else: GY_pot=st.sidebar.number_input("GY_pot personalizado (kg/ha)",1000,15000,6000,100)

    st.sidebar.subheader("🌿 Canopia")
    LAI_max=st.sidebar.slider("LAI_max",2.0,10.0,6.0,0.1)
    t_lag=st.sidebar.slider("t_lag",0,60,10)
    t_close=st.sidebar.slider("t_close",10,100,35)
    LAI_hc=st.sidebar.slider("LAI_hc",2.0,10.0,6.0,0.1)
    Cs=st.sidebar.number_input("Cs",50,800,200)
    Ca=st.sidebar.number_input("Ca",30,800,200)

    st.sidebar.subheader("⚖️ Supresión (exp de (1−Ciec))")
    p_S1=st.sidebar.slider("S1",0.0,2.0,1.0,0.1)
    p_S2=st.sidebar.slider("S2",0.0,2.0,0.6,0.1)
    p_S3=st.sidebar.slider("S3",0.0,2.0,0.4,0.1)
    p_S4=st.sidebar.slider("S4",0.0,2.0,0.2,0.1)

    st.sidebar.subheader("🧪 Herbicidas")
    preR_days_before=st.sidebar.number_input("PreR: días antes",0,180,14)
    preR_eff=st.sidebar.slider("PreR eficacia (%)",0,100,90)
    preR_residual=st.sidebar.slider("PreR residual (d)",0,180,30)
    postR_days_after=st.sidebar.number_input("PostR: días después",0,180,25)
    postR_eff=st.sidebar.slider("PostR eficacia (%)",0,100,85)
    postR_residual=st.sidebar.slider("PostR residual (d)",0,180,10)
    gram_days_after=st.sidebar.number_input("Gram: días después",0,180,10)
    gram_eff=st.sidebar.slider("Gram eficacia (%)",0,100,80)
    gram_residual=st.sidebar.slider("Gram residual (d)",0,180,7)

    if st.sidebar.button("▶ Ejecutar simulación"):
        df = simulate(nyears, seed_bank0, K, Tb, sim_seed,
                      preR_days_before, preR_eff, preR_residual,
                      postR_days_after, postR_eff, postR_residual,
                      gram_days_after, gram_eff, gram_residual,
                      sow_date, LAI_max, t_lag, t_close, LAI_hc, Cs, Ca,
                      p_S1, p_S2, p_S3, p_S4, 0.15,0.3,0.6,1.0,0.9782,83.77, GY_pot)

        st.success(f"✅ Simulación completada — {len(df)} días")

        # --- Tab 1: Evolución temporal ---
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df["date"], y=df["Yield_abs_kg_ha"], name="Rinde (kg/ha)", line=dict(color="green")))
        fig1.add_trace(go.Scatter(x=df["date"], y=df["Yield_loss_%"], name="Pérdida (%)", yaxis="y2", line=dict(color="red")))
        fig1.update_layout(title="Rinde absoluto y pérdida de rinde",
            xaxis=dict(title="Fecha"), yaxis=dict(title="Rinde (kg/ha)", rangemode="tozero"),
            yaxis2=dict(title="Pérdida (%)", overlaying="y", side="right"),
            template="plotly_white", legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig1, use_container_width=True)

        # --- Tab 2: Relación WC vs Pérdida ---
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["WC"], y=df["Yield_loss_%"],
                                  mode="markers+lines", name="WC vs Pérdida", line=dict(color="darkorange")))
        fig2.update_layout(title="Índice de competencia (WC) vs Pérdida de rinde",
                           xaxis_title="WC (ponderado)", yaxis_title="Pérdida (%)",
                           template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

        st.metric("💰 Rinde final (kg/ha)", f"{df['Yield_abs_kg_ha'].iloc[-1]:.0f}")
        st.metric("🌾 Rinde relativo (%)", f"{df['Yield_relative_%'].iloc[-1]:.1f}")
        st.metric("🧮 Pérdida final (%)", f"{df['Yield_loss_%'].iloc[-1]:.1f}")

        st.download_button("📥 Descargar CSV", df.to_csv(index=False).encode(),
                           "weedcrop_v391_yield_effective.csv","text/csv")
    else:
        st.info("Configura parámetros y presiona ▶ Ejecutar simulación.")
