# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 WeedCropSystem — Híbrido + Ventanas activas de control
# ---------------------------------------------------------------
# Modo Streamlit → interfaz visual
# Modo Consola → imprime resumen + ventanas activas + exporta CSV
# ===============================================================

import sys, datetime as dt
import numpy as np
import pandas as pd

# ---------- FUNCIONES BÁSICAS ----------
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
    LAI = float(max(0.0, min(LAI, LAI_max)))
    return LAI, float(k)

def ciec_calendar(days_since_sow, LAI_max, t_lag, t_close, LAI_hc, Cs, Ca):
    LAI, k = lai_logistic_by_day(days_since_sow, LAI_max, t_lag, t_close)
    ratio = (float(Cs) / max(float(Ca), 1e-6))
    Ciec = min((LAI / max(float(LAI_hc), 1e-6)) * ratio, 1.0)
    return float(Ciec), LAI, k


# ---------- SIMULADOR ----------
def simulate(
    nyears=1, seed_bank0=4500, K=250, Tb=0.0, seed=42,
    preR_days_before=14, preR_eff=90, preR_residual=30,
    postR_days_after=25, postR_eff=85, postR_residual=10,
    gram_days_after=10, gram_eff=80, gram_residual=7,
    sow_date=dt.date(2025,6,1),
    LAI_max=6.0, t_lag=10, t_close=35, LAI_hc=6.0, Cs=200, Ca=200,
    p_S1=1.0, p_S2=0.6, p_S3=0.4, p_S4=0.2
) -> tuple[pd.DataFrame, dict]:
    sow = pd.to_datetime(sow_date).date()
    start = sow - dt.timedelta(days=int(preR_days_before))
    end = dt.date(sow.year + int(nyears) - 1, 12, 1)
    meteo = synthetic_meteo(start, end, seed)

    # Ventanas
    preR_date = sow - dt.timedelta(days=int(preR_days_before))
    postR_date = sow + dt.timedelta(days=int(postR_days_after))
    gram_date = sow + dt.timedelta(days=int(gram_days_after))
    preR_window = [preR_date + dt.timedelta(days=i) for i in range(int(preR_residual))]
    postR_window = [postR_date + dt.timedelta(days=i) for i in range(int(postR_residual))]
    gram_window = [gram_date + dt.timedelta(days=i) for i in range(int(gram_residual))]
    windows = {
        "preR": (preR_window[0], preR_window[-1], len(preR_window)),
        "postR": (postR_window[0], postR_window[-1], len(postR_window)),
        "gram": (gram_window[0], gram_window[-1], len(gram_window)),
    }

    Sq = float(seed_bank0)
    TTw = 0.0
    W = [0,0,0,0,0]
    Th = [70, 280, 400, 300]
    out = []

    for _, row in meteo.iterrows():
        date = pd.to_datetime(row["date"]).date()
        dss = (date - sow).days
        Tmean = (float(row["tmin"]) + float(row["tmax"])) / 2
        TTw += max(Tmean - Tb, 0)
        Ciec_t, LAI_t, _ = ciec_calendar(dss, LAI_max, t_lag, t_close, LAI_hc, Cs, Ca)
        E_t = emergence_simple(TTw, float(row["prec"]))

        Wk = sum(np.array(W) * np.array([0.15,0.3,0.6,1.0,0.0]))
        surv_intra = 1 - min(Wk/K, 1)
        sup_S1 = (1 - Ciec_t)**p_S1
        I1_t = max(0, Sq * E_t * surv_intra * sup_S1)

        Ct = [0,0,0,0,0]
        if date in preR_window: Ct[0] = preR_eff/100
        if date in postR_window: 
            for i in range(4): Ct[i] = postR_eff/100
        if date in gram_window: 
            for i in range(3): Ct[i] = gram_eff/100

        W_ctrl = [w*(1-c) for w,c in zip(W,Ct)]
        sup_S2 = (1 - Ciec_t)**p_S2
        sup_S3 = (1 - Ciec_t)**p_S3
        sup_S4 = (1 - Ciec_t)**p_S4
        W_ctrl[1]*=sup_S2; W_ctrl[2]*=sup_S3; W_ctrl[3]*=sup_S4

        O1 = I1_t if TTw>=Th[0] else 0
        O2 = W_ctrl[1] if TTw>=sum(Th[:2]) else 0
        O3 = W_ctrl[2] if TTw>=sum(Th[:3]) else 0
        O4 = W_ctrl[3] if TTw>=sum(Th[:4]) else 0

        W1 = max(0, W_ctrl[0] + I1_t - O1)
        W2 = max(0, W_ctrl[1] + O1 - O2)
        W3 = max(0, W_ctrl[2] + O2 - O3)
        W4 = max(0, W_ctrl[3] + O3 - O4)
        W5 = max(0, W_ctrl[4] + O4)
        W = [W1,W2,W3,W4,W5]

        out.append({
            "date":date,"TTw":TTw,"Ciec":Ciec_t,"LAI":LAI_t,"E_t":E_t,
            "W1":W1,"W2":W2,"W3":W3,"W4":W4,"W5":W5,
            "control_preR":Ct[0],"control_postR":max(Ct[:4]),"control_gram":max(Ct[:3])
        })
    df = pd.DataFrame(out)
    df["W_total"] = df[["W1","W2","W3","W4"]].sum(axis=1)
    return df, windows


# ---------- DETECCIÓN MODO ----------
if "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv):
    import streamlit as st, plotly.graph_objects as go
    st.set_page_config(page_title="WeedCropSystem Híbrido", layout="wide")
    st.title("🌾 WeedCropSystem — Híbrido con ventanas activas")
    sow_date = st.sidebar.date_input("Fecha de siembra", dt.date(2025,6,1))
    run_btn = st.sidebar.button("▶ Ejecutar simulación")
    if run_btn:
        df, windows = simulate(sow_date=sow_date)
        st.success(f"Simulación completada ({len(df)} días).")

        st.write("### 🧪 Ventanas activas")
        for k,(ini,fin,dias) in windows.items():
            st.write(f"**{k.upper()}**: {ini} → {fin} ({dias} días)")

        fig = go.Figure()
        for s in ["W1","W2","W3","W4"]:
            fig.add_trace(go.Scatter(x=df["date"], y=df[s], mode="lines", name=s))
        fig.update_layout(title="Densidades por estado", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("📥 Descargar CSV", df.to_csv(index=False).encode(), "weedcrop_hybrid.csv")
    else:
        st.info("Configura parámetros y presioná ▶ Ejecutar simulación.")
else:
    print("🌾 WeedCropSystem — modo consola")
    df, windows = simulate()
    print("\nVentanas activas:")
    for k,(ini,fin,dias) in windows.items():
        print(f" - {k.capitalize():5s}: {ini} → {fin} ({dias} días)")
    print(f"\nSimulación completada ({len(df)} días).")
    print(df[["date","W1","W2","W3","W4","W_total"]].head(10))
    df.to_csv("weedcrop_hybrid.csv", index=False)
    print("✅ Resultados guardados en weedcrop_hybrid.csv")
