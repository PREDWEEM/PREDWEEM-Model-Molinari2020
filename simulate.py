# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 WeedCropSystem — v3.15 (Supuestos agronómicos + sensibilidad PC)
# ---------------------------------------------------------------
# - Supuesto 1: lote limpio a la siembra (sin malezas previas)
# - Supuesto 2: el cultivo es 5× más sensible al AUC de competencia
#                durante el período crítico (8/oct–4/nov)
# - Incluye reglas agronómicas para preR, preemR, postR y gram
# - Base: simulación diaria + AUC de competencia + gateo jerárquico
# ===============================================================

import sys, datetime as dt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# 🌦️ Generador meteorológico sintético
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# 🌱 Funciones auxiliares fisiológicas
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# 🧠 Simulador diario con gateo jerárquico y AUC de competencia
# ---------------------------------------------------------------
def simulate_with_controls(
    nyears=1, seed_bank0=4500, K=250, Tb=0.0, seed=42,
    sow_date=dt.date(2025,6,1),
    LAI_max=3.0, t_lag=10, t_close=35, LAI_hc=6.0, Cs=200, Ca=200,
    p_S1=1.0, p_S2=0.6, p_S3=0.4, p_S4=0.2,
    w_S1=0.15, w_S2=0.30, w_S3=0.60, w_S4=1.00,
    alpha=0.9782, Lmax=83.77, GY_pot=6000.0,
    preR_eff=90, preemR_eff=70, postR_eff=85, gram_eff=80,
    preR_residual=30, preemR_residual=30, postR_residual=30, gram_residual_forward=11,
    preR_date=None, preemR_date=None, postR_date=None, gram_date=None,
    enforce_rules=True
):
    sow = pd.to_datetime(sow_date).date()
    start = sow - dt.timedelta(days=90)
    end = dt.date(sow.year + int(nyears) - 1, 12, 1)
    meteo = synthetic_meteo(start, end, seed)

    # 🧭 Ventanas fenológicas agronómicas
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
    out = []

    for _, row in meteo.iterrows():
        date = pd.to_datetime(row["date"]).date()

        # =======================================================
        # 🌱 Supuesto 1: lote limpio a la siembra
        # No se acumulan emergencias antes del día 0 (siembra)
        # =======================================================
        if date < sow:
            continue

        dss = (date - sow).days
        Tmean = (row["tmin"] + row["tmax"]) / 2
        TTw += max(Tmean - Tb, 0)
        Ciec_t, LAI_t = ciec_calendar(dss, LAI_max, t_lag, t_close, LAI_hc, Cs, Ca)
        E_t = 2.0 * emergence_simple(TTw, row["prec"])

        Wk = sum(np.array(W)*np.array([0.15,0.3,0.6,1.0,0.0]))
        surv_intra = 1 - min(Wk/K,1)
        sup_S1 = (1-Ciec_t)**p_S1
        I1_t = max(0, Sq * E_t * surv_intra * sup_S1)
        Wc = W.copy()
        Wc[1] *= (1-Ciec_t)**p_S2
        Wc[2] *= (1-Ciec_t)**p_S3
        Wc[3] *= (1-Ciec_t)**p_S4

        eff_accum = 0.0
        def _rem(states): return sum(Wc[i] for i,s in enumerate(["S1","S2","S3","S4"]) if s in states)
        def _apply(eff, states):
            nonlocal eff_accum
            if eff<=0 or eff_accum>=0.99: return
            if _rem(states)<=EPS: return
            f = 1 - eff/100.0
            for i,s in enumerate(["S1","S2","S3","S4"]):
                if s in states: Wc[i]*=f
            eff_accum = 1-(1-eff_accum)*(1-eff/100.0)

        if date in preR_window:   _apply(preR_eff,   ["S1","S2"])
        if date in preemR_window: _apply(preemR_eff, ["S1","S2"])
        if date in postR_window:  _apply(postR_eff,  ["S1","S2","S3","S4"])
        if date in gram_window:   _apply(gram_eff,   ["S1","S2","S3"])

        O1 = I1_t if TTw>=Th[0] else 0
        O2 = Wc[1] if TTw>=sum(Th[:2]) else 0
        O3 = Wc[2] if TTw>=sum(Th[:3]) else 0
        O4 = Wc[3] if TTw>=sum(Th[:4]) else 0
        W1 = max(0, Wc[0]+I1_t-O1)
        W2 = max(0, Wc[1]+O1-O2)
        W3 = max(0, Wc[2]+O2-O3)
        W4 = max(0, Wc[3]+O3-O4)
        W5 = max(0, Wc[4]+O4)
        W = [W1,W2,W3,W4,W5]

        out.append({
            "date":date,"days_since_sow":dss,"TTw":TTw,"Ciec":Ciec_t,"LAI":LAI_t,
            "W1":W1,"W2":W2,"W3":W3,"W4":W4
        })

    df = pd.DataFrame(out)
    df["W_total"] = df[["W1","W2","W3","W4"]].sum(axis=1)
    df["WC"] = w_S1*df["W1"]+w_S2*df["W2"]+w_S3*df["W3"]+w_S4*df["W4"]

    # 💡 AUC acumulado de competencia
    df["WC_acum"] = df["WC"].cumsum() / len(df)

    # ==============================================================
    # 🌾 Supuesto 2: Período crítico (8/oct–4/nov) sensibilidad ×5
    # ==============================================================
    PC_ini = dt.date(sow.year, 10, 8)
    PC_fin = dt.date(sow.year, 11, 4)
    df["Sens_factor"] = np.where(
        (df["date"] >= PC_ini) & (df["date"] <= PC_fin),
        5.0, 1.0
    )

    # 📉 Pérdida de rinde con sensibilidad dinámica
    df["Yield_loss_%"] = (
        (alpha * df["WC_acum"] * df["Sens_factor"]) /
        (1 + (alpha * df["WC_acum"] * df["Sens_factor"] / Lmax))
    )
    df["Yield_relative_%"] = 100 - df["Yield_loss_%"]
    df["Yield_abs_kg_ha"] = GY_pot * (df["Yield_relative_%"]/100)
    return df

    # ===============================================================
# BLOQUE 2 — Streamlit + Optimización de fechas (v3.15)
# ===============================================================
import streamlit as st
import plotly.graph_objects as go
import itertools, random, datetime as dt
import numpy as np
import pandas as pd

def objective_loss(params, base_kwargs):
    sim = simulate_with_controls(**base_kwargs, **params)
    if sim is None or sim.empty:
        return np.inf
    return float(sim["Yield_loss_%"].iloc[-1])

st.set_page_config(page_title="WeedCropSystem v3.15", layout="wide")
st.title("🌾 WeedCropSystem v3.15 — Supuestos + Sensibilidad PC + Optimización")

# ----- Parámetros base -----
st.sidebar.header("⚙️ Parámetros base")
nyears = st.sidebar.slider("Años a simular", 1, 3, 1)
seed_bank0 = st.sidebar.number_input("Banco inicial (semillas·m⁻²)", 0, 50000, 4500)
K = st.sidebar.number_input("Capacidad de carga K (pl·m⁻²)", 50, 2000, 250)
Tb = st.sidebar.number_input("Temp. base Tb (°C)", 0.0, 15.0, 0.0, 0.5)
sim_seed = st.sidebar.number_input("Semilla aleatoria clima", 0, 999999, 42)
sow_date = st.sidebar.date_input("Fecha de siembra", dt.date(2025,6,1))

st.sidebar.subheader("🌿 Canopia / Competencia")
LAI_max = st.sidebar.slider("LAI_max", 2.0, 10.0, 6.0, 0.1)
t_lag   = st.sidebar.slider("t_lag", 0, 60, 10)
t_close = st.sidebar.slider("t_close", 10, 100, 35)
LAI_hc  = st.sidebar.slider("LAI_hc", 2.0, 10.0, 6.0, 0.1)
Cs      = st.sidebar.number_input("Cs (pl/m²)", 50, 800, 200)
Ca      = st.sidebar.number_input("Ca (pl/m²)", 30, 800, 200)

st.sidebar.subheader("⚖️ Supresión (1−Ciec)^p")
p_S1 = st.sidebar.slider("S1", 0.0, 2.0, 1.0, 0.1)
p_S2 = st.sidebar.slider("S2", 0.0, 2.0, 0.6, 0.1)
p_S3 = st.sidebar.slider("S3", 0.0, 2.0, 0.4, 0.1)
p_S4 = st.sidebar.slider("S4", 0.0, 2.0, 0.2, 0.1)

st.sidebar.subheader("🌾 Rinde potencial y pérdida")
GY_pot = st.sidebar.number_input("Rinde potencial (kg/ha)", 1000, 15000, 6000, 100)
alpha, Lmax = 0.9782, 83.77

st.sidebar.header("Eficacias fijas (%)")
preR_eff = st.sidebar.slider("Presiembra (S1–S2)", 0, 100, 90, 1)
preemR_eff = st.sidebar.slider("Preemergente (S1–S2)", 0, 100, 70, 1)
postR_eff = st.sidebar.slider("Post residual (S1–S4)", 0, 100, 85, 1)
gram_eff = st.sidebar.slider("Graminicida (S1–S3)", 0, 100, 80, 1)

st.sidebar.header("Residualidades (días)")
preR_residual = st.sidebar.slider("preR", 10, 180, 30, 1)
preemR_residual = st.sidebar.slider("preemR", 10, 180, 30, 1)
postR_residual = st.sidebar.slider("postR", 10, 180, 30, 1)
gram_residual_forward = 11

base_kwargs = dict(
  nyears=nyears, seed_bank0=seed_bank0, K=K, Tb=Tb, seed=sim_seed,
  sow_date=sow_date, LAI_max=LAI_max, t_lag=t_lag, t_close=t_close,
  LAI_hc=LAI_hc, Cs=Cs, Ca=Ca,
  p_S1=p_S1, p_S2=p_S2, p_S3=p_S3, p_S4=p_S4,
  w_S1=0.15, w_S2=0.30, w_S3=0.60, w_S4=1.00,
  alpha=alpha, Lmax=Lmax, GY_pot=GY_pot,
  preR_eff=preR_eff, preemR_eff=preemR_eff, postR_eff=postR_eff, gram_eff=gram_eff,
  preR_residual=preR_residual, preemR_residual=preemR_residual, postR_residual=postR_residual,
  gram_residual_forward=gram_residual_forward, enforce_rules=True
)

# ----- Modo -----
mode = st.sidebar.selectbox("Modo:", ["Simulación única", "Optimización de fechas"])

# ---------------------------------------------------------------
# 🧩 SIMULACIÓN ÚNICA
# ---------------------------------------------------------------
if mode == "Simulación única":
    st.sidebar.subheader("Fechas de aplicación (respetan reglas)")
    preR_date = st.sidebar.date_input("Presiembra (−30 a −14)", sow_date - dt.timedelta(days=20))
    preemR_date = st.sidebar.date_input("Preemergente (0 a +10)", sow_date)
    postR_date = st.sidebar.date_input("Postresidual (+20 a +180)", sow_date + dt.timedelta(days=30))
    gram_date = st.sidebar.date_input("Graminicida (+25 a +35)", sow_date + dt.timedelta(days=30))

    if st.sidebar.button("▶ Ejecutar simulación"):
        df = simulate_with_controls(**base_kwargs,
            preR_date=preR_date, preemR_date=preemR_date,
            postR_date=postR_date, gram_date=gram_date)
        if df is None or df.empty:
            st.error("❌ Fechas fuera de las reglas fenológicas.")
        else:
            st.success(f"✅ Simulación exitosa — {len(df)} días")

            # Gráfico de rinde y pérdida
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["date"], y=df["Yield_abs_kg_ha"], name="Rinde (kg/ha)"))
            fig.add_trace(go.Scatter(x=df["date"], y=df["Yield_loss_%"], name="Pérdida (%)", yaxis="y2"))
            fig.update_layout(
                title="Rinde y Pérdida de rinde (%)",
                xaxis_title="Fecha",
                yaxis_title="Rinde (kg/ha)",
                yaxis2=dict(title="Pérdida (%)", overlaying="y", side="right"),
                template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Gráfico de competencia y AUC
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df["date"], y=df["WC"], name="Competencia instantánea (WC)"))
            fig2.add_trace(go.Scatter(x=df["date"], y=df["WC_acum"], name="AUC de competencia"))
            fig2.update_layout(title="Dinámica de competencia", template="plotly_white",
                               xaxis_title="Fecha", yaxis_title="Índice")
            st.plotly_chart(fig2, use_container_width=True)

            st.metric("💰 Rinde final", f"{df['Yield_abs_kg_ha'].iloc[-1]:.0f} kg/ha")
            st.metric("🧮 Pérdida final", f"{df['Yield_loss_%'].iloc[-1]:.2f}%")
            st.download_button("📥 Descargar CSV", df.to_csv(index=False).encode(),
                               "simulacion_v315.csv","text/csv")

# ---------------------------------------------------------------
# 🧠 OPTIMIZACIÓN DE FECHAS
# ---------------------------------------------------------------
else:
    st.sidebar.header("Espacio de búsqueda (respetando reglas)")
    preR_enable  = st.sidebar.checkbox("Optimizar presiembra (−30..−14)", True)
    preem_enable = st.sidebar.checkbox("Optimizar preemergente (0..+10)", True)
    postR_enable = st.sidebar.checkbox("Optimizar post residual (+20..+180)", True)
    gram_enable  = st.sidebar.checkbox("Optimizar graminicida (+25..+35)", True)

    step_preR  = st.sidebar.number_input("Paso preR (días)", 1, 15, 7)
    step_preem = st.sidebar.number_input("Paso preemR (días)", 1, 5, 3)
    step_postR = st.sidebar.number_input("Paso postR (días)", 1, 30, 7)
    step_gram  = st.sidebar.number_input("Paso gram (días)", 1, 5, 3)

    max_evals   = st.sidebar.number_input("Máx. escenarios a evaluar", 10, 5000, 500)
    search_mode = st.sidebar.selectbox("Estrategia", ["Grid (completo)", "Muestreo aleatorio"])
    run_opt = st.sidebar.button("🚀 Ejecutar optimización")

    def _cand_preR():
        # desde -30 hasta -14 inclusive
        days = list(range(30, 13, -int(step_preR)))
        return [sow_date - dt.timedelta(days=d) for d in days if 14 <= d <= 30]
    def _cand_preem():
        return [sow_date + dt.timedelta(days=d) for d in range(0, 11, int(step_preem))]
    def _cand_postR():
        return [sow_date + dt.timedelta(days=d) for d in range(20, 181, int(step_postR))]
    def _cand_gram():
        return [sow_date + dt.timedelta(days=d) for d in range(25, 36, int(step_gram))]

    if run_opt:
        preR_list   = _cand_preR()  if preR_enable  else [None]
        preem_list  = _cand_preem() if preem_enable else [None]
        postR_list  = _cand_postR() if postR_enable else [None]
        gram_list   = _cand_gram()  if gram_enable  else [None]

        combos = list(itertools.product(preR_list, preem_list, postR_list, gram_list))
        if search_mode.startswith("Muestreo"):
            random.seed(123)
            if len(combos) > max_evals:
                combos = random.sample(combos, int(max_evals))
        else:
            if len(combos) > max_evals:
                st.warning(f"Grid muy grande ({len(combos):,}) → recortado a {int(max_evals):,}")
                combos = combos[:int(max_evals)]

        results = []
        prog = st.progress(0.0)
        for i,(d_preR, d_preem, d_postR, d_gram) in enumerate(combos, 1):
            params = dict(preR_date=d_preR, preemR_date=d_preem, postR_date=d_postR, gram_date=d_gram)
            loss = objective_loss(params, base_kwargs)
            if np.isfinite(loss):
                results.append({
                    "preR": d_preR, "preemR": d_preem, "postR": d_postR, "gram": d_gram,
                    "loss_pct": loss
                })
            if i % max(1, len(combos)//100) == 0:
                prog.progress(min(1.0, i/len(combos)))
        prog.progress(1.0)

        if not results:
            st.error("No se encontraron combinaciones válidas con las reglas.")
        else:
            res_df = pd.DataFrame(results).sort_values("loss_pct").reset_index(drop=True)
            best = res_df.iloc[0]
            st.success(f"🏆 Mejor pérdida de rinde: {best['loss_pct']:.2f}%")
            st.dataframe(res_df.head(15), use_container_width=True)
            st.download_button("📥 Descargar resultados CSV",
                               res_df.to_csv(index=False).encode(),
                               "opt_fechas_resultados_v315.csv","text/csv")

            # Re-simular el mejor escenario
            df_best = simulate_with_controls(**base_kwargs,
                preR_date=best["preR"], preemR_date=best["preemR"],
                postR_date=best["postR"], gram_date=best["gram"])
            if df_best is not None and not df_best.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_best["date"], y=df_best["Yield_abs_kg_ha"], name="Rinde (kg/ha)"))
                fig.add_trace(go.Scatter(x=df_best["date"], y=df_best["Yield_loss_%"], name="Pérdida (%)", yaxis="y2"))
                fig.update_layout(title="📈 Mejor escenario — Rinde y Pérdida", template="plotly_white",
                                  xaxis_title="Fecha", yaxis_title="Rinde (kg/ha)",
                                  yaxis2=dict(title="Pérdida (%)", overlaying="y", side="right"))
                st.plotly_chart(fig, use_container_width=True)
                st.metric("💰 Rinde final", f"{df_best['Yield_abs_kg_ha'].iloc[-1]:.0f} kg/ha")
                st.metric("🧮 Pérdida final", f"{df_best['Yield_loss_%'].iloc[-1]:.2f}%")
    else:
        st.info("Ajustá el espacio de búsqueda y presioná “Ejecutar optimización”.")





















