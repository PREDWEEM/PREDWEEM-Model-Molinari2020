# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 WeedCropSystem — v3.12 (Reglas fenológicas reales + Optimización)
# ---------------------------------------------------------------
# - Una única especie
# - Emergencia intensificada ×8
# - Pérdida de rinde: α/Lmax fijos
# - Gateo por remanente + exclusión jerárquica (99%) por DÍA
# - Ventanas agronómicas:
#   · preR: entre 30 y 14 días antes de siembra
#   · preemR: entre siembra y emergencia del cultivo (~+10 d)
#   · postR: desde 2 hojas (~+20 d) hasta fin del ciclo
#   · gram: entre 3 y 4 hojas (~+25 a +35 d)
# ===============================================================

import sys, datetime as dt
import numpy as np
import pandas as pd

# ===============================================================
# 🌦️ Núcleo de simulación
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
# 🧩 Simulación con controles y reglas fenológicas
# ===============================================================
def simulate_with_controls(
    nyears=1, seed_bank0=4500, K=250, Tb=0.0, seed=42,
    sow_date=dt.date(2025,6,1),
    LAI_max=6.0, t_lag=10, t_close=35, LAI_hc=6.0, Cs=200, Ca=200,
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

    # ===============================================================
    # 🌱 Ventanas fenológicas reales de aplicación
    # ===============================================================
    preR_start  = sow - dt.timedelta(days=30)
    preR_end    = sow - dt.timedelta(days=14)
    preemR_start = sow
    preemR_end   = sow + dt.timedelta(days=10)
    postR_start  = sow + dt.timedelta(days=20)
    postR_end    = sow + dt.timedelta(days=180)
    gram_start   = sow + dt.timedelta(days=25)
    gram_end     = sow + dt.timedelta(days=35)

    if enforce_rules:
        if preR_date is not None and not (preR_start <= preR_date <= preR_end): return None
        if preemR_date is not None and not (preemR_start <= preemR_date <= preemR_end): return None
        if postR_date is not None and not (postR_start <= postR_date <= postR_end): return None
        if gram_date is not None and not (gram_start <= gram_date <= gram_end): return None

    preR_window   = set() if preR_date is None else _date_range(preR_date, preR_residual)
    preemR_window = set() if preemR_date is None else _date_range(preemR_date, preemR_residual)
    postR_window  = set() if postR_date is None else _date_range(postR_date, postR_residual)
    gram_window   = set() if gram_date is None else _date_range(gram_date, gram_residual_forward)

    # ===============================================================
    # 🔄 Dinámica diaria
    # ===============================================================
    EPS_REMAIN = 1e-9
    EPS_EXCLUDE = 0.99
    Sq = float(seed_bank0)
    TTw = 0.0
    W = [0,0,0,0,0]  # S1..S5
    Th = [70, 280, 400, 300]  # TT para transición
    out = []

    for _, row in meteo.iterrows():
        date = pd.to_datetime(row["date"]).date()
        dss = (date - sow).days
        Tmean = (float(row["tmin"])+float(row["tmax"])) / 2
        TTw += max(Tmean - float(Tb), 0)

        Ciec_t, LAI_t = ciec_calendar(dss, LAI_max, t_lag, t_close, LAI_hc, Cs, Ca)
        E_t = 8.0 * emergence_simple(TTw, float(row["prec"]))
        Wk = sum(np.array(W)*np.array([0.15,0.3,0.6,1.0,0.0]))
        surv_intra = 1 - min(Wk/K,1)
        sup_S1 = (1-Ciec_t)**p_S1
        I1_t = max(0, Sq * E_t * surv_intra * sup_S1)

        Wc = W.copy()
        Wc[1] *= (1-Ciec_t)**p_S2
        Wc[2] *= (1-Ciec_t)**p_S3
        Wc[3] *= (1-Ciec_t)**p_S4

        eff_accum = 0.0
        def _rem_in(states):
            return sum(Wc[i] for i,s in enumerate(["S1","S2","S3","S4"]) if s in states)
        def _apply_eff(eff_pct, states):
            nonlocal eff_accum
            if eff_pct <= 0 or eff_accum >= EPS_EXCLUDE: return
            rem = _rem_in(states)
            if rem <= EPS_REMAIN: return
            f = max(0.0, 1.0 - (eff_pct/100.0))
            for i,s in enumerate(["S1","S2","S3","S4"]):
                if s in states: Wc[i] *= f
            eff_accum = 1.0 - (1.0 - eff_accum)*(1.0 - eff_pct/100.0)

        if date in preR_window:   _apply_eff(preR_eff,   ["S1","S2"])
        if date in preemR_window: _apply_eff(preemR_eff, ["S1","S2"])
        if date in postR_window:  _apply_eff(postR_eff,  ["S1","S2","S3","S4"])
        if date in gram_window:   _apply_eff(gram_eff,   ["S1","S2","S3"])

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
        out.append({"date":date,"days_since_sow":dss,"TTw":TTw,"Ciec":Ciec_t,"LAI":LAI_t,
                    "W1":W1,"W2":W2,"W3":W3,"W4":W4,"W5":W5})

    df = pd.DataFrame(out)
    df["W_total"]=df[["W1","W2","W3","W4"]].sum(axis=1)
    df["WC"]=w_S1*df["W1"]+w_S2*df["W2"]+w_S3*df["W3"]+w_S4*df["W4"]
    df["Yield_loss_%"]=(alpha*df["WC"])/(1+(alpha*df["WC"]/Lmax))
    df["Yield_relative_%"]=100-df["Yield_loss_%"]
    df["Yield_abs_kg_ha"]=GY_pot*(df["Yield_relative_%"]/100)
    return df

# ===============================================================
# 🎯 Función objetivo para optimización
# ===============================================================
def objective_loss(params, base_kwargs):
    sim = simulate_with_controls(**base_kwargs, **params)
    if sim is None or sim.empty:
        return np.inf
    return float(sim["Yield_loss_%"].iloc[-1])

# ===============================================================
# 🖥️ STREAMLIT APP
# ===============================================================
if "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv):
    import streamlit as st
    import plotly.graph_objects as go
    import itertools, random

    st.set_page_config(page_title="WeedCropSystem v3.12", layout="wide")
    st.title("🌾 WeedCropSystem — v3.12 (Reglas fenológicas reales + Optimización)")

    # -------------------------------------------------------------
    # Panel lateral
    # -------------------------------------------------------------
    st.sidebar.header("Escenario base")
    nyears = st.sidebar.slider("Años a simular", 1, 3, 1)
    seed_bank0 = st.sidebar.number_input("Banco inicial (semillas·m⁻²)", 0, 50000, 4500)
    K = st.sidebar.number_input("Cap. de carga K (pl·m⁻²)", 50, 2000, 250)
    Tb = st.sidebar.number_input("Temp. base Tb (°C)", 0.0, 15.0, 0.0, 0.5)
    sim_seed = st.sidebar.number_input("Semilla aleatoria clima", 0, 999999, 42)
    sow_date = st.sidebar.date_input("Fecha de siembra", dt.date(2025,6,1))

    st.sidebar.subheader("🌿 Canopia / Competencia")
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

    st.sidebar.subheader("🌾 Rinde potencial")
    GY_pot = 6000.0
    alpha, Lmax = 0.9782, 83.77

    st.sidebar.divider()
    st.sidebar.header("Eficacias fijas (%)")
    preR_eff = st.sidebar.slider("preR (S1–S2)", 0, 100, 90)
    preemR_eff = st.sidebar.slider("preemR (S1–S2)", 0, 100, 70)
    postR_eff = st.sidebar.slider("postR (S1–S4)", 0, 100, 85)
    gram_eff = st.sidebar.slider("gram (S1–S3)", 0, 100, 80)

    st.sidebar.header("Residualidades (días)")
    preR_residual = st.sidebar.slider("preR residual", 10, 180, 30)
    preemR_residual = st.sidebar.slider("preemR residual", 10, 180, 30)
    postR_residual = st.sidebar.slider("postR residual", 10, 180, 30)
    gram_residual_forward = 11

    mode = st.sidebar.selectbox("Modo", ["Simulación única", "Optimización (fechas)"])

    base_kwargs = dict(
        nyears=nyears, seed_bank0=seed_bank0, K=K, Tb=Tb, seed=sim_seed,
        sow_date=sow_date, LAI_max=LAI_max, t_lag=t_lag, t_close=t_close,
        LAI_hc=LAI_hc, Cs=Cs, Ca=Ca, p_S1=p_S1, p_S2=p_S2, p_S3=p_S3, p_S4=p_S4,
        alpha=alpha, Lmax=Lmax, GY_pot=GY_pot,
        preR_eff=preR_eff, preemR_eff=preemR_eff, postR_eff=postR_eff, gram_eff=gram_eff,
        preR_residual=preR_residual, preemR_residual=preemR_residual, postR_residual=postR_residual,
        gram_residual_forward=gram_residual_forward, enforce_rules=True
    )

    # -------------------------------------------------------------
    # 🔹 Simulación única
    # -------------------------------------------------------------
    if mode == "Simulación única":
        st.sidebar.subheader("Fechas (opcionales, cumplen reglas)")
        use_preR   = st.sidebar.checkbox("Usar preR", False)
        use_preemR = st.sidebar.checkbox("Usar preemR", False)
        use_postR  = st.sidebar.checkbox("Usar postR", False)
        use_gram   = st.sidebar.checkbox("Usar gram", False)

        preR_date = (st.sidebar.date_input("preR (−30 a −14)",
                                           min_value=sow_date - dt.timedelta(days=60),
                                           max_value=sow_date,
                                           value=sow_date - dt.timedelta(days=21))
                     if use_preR else None)
        preemR_date = (st.sidebar.date_input("preemR (0 a +10)",
                                             min_value=sow_date,
                                             max_value=sow_date + dt.timedelta(days=10),
                                             value=sow_date)
                       if use_preemR else None)
        postR_date = (st.sidebar.date_input("postR (≥ +20)",
                                            min_value=sow_date + dt.timedelta(days=20),
                                            max_value=sow_date + dt.timedelta(days=180),
                                            value=sow_date + dt.timedelta(days=30))
                      if use_postR else None)
        gram_date = (st.sidebar.date_input("gram (+25 a +35)",
                                           min_value=sow_date + dt.timedelta(days=25),
                                           max_value=sow_date + dt.timedelta(days=35),
                                           value=sow_date + dt.timedelta(days=28))
                     if use_gram else None)

        if st.sidebar.button("▶ Ejecutar simulación"):
            df = simulate_with_controls(**base_kwargs, preR_date=preR_date,
                                        preemR_date=preemR_date, postR_date=postR_date,
                                        gram_date=gram_date)
            if df is None or df.empty:
                st.error("Configuración inválida según reglas fenológicas.")
            else:
                st.success(f"Simulación OK — {len(df)} días")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["date"], y=df["Yield_abs_kg_ha"], name="Rinde (kg/ha)"))
                fig.add_trace(go.Scatter(x=df["date"], y=df["Yield_loss_%"], name="Pérdida (%)", yaxis="y2"))
                fig.update_layout(title="Rinde y Pérdida (%)",
                                  xaxis_title="Fecha",
                                  yaxis_title="Rinde (kg/ha)",
                                  yaxis2=dict(title="Pérdida (%)", overlaying="y", side="right"),
                                  template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                st.metric("💰 Rinde final (kg/ha)", f"{df['Yield_abs_kg_ha'].iloc[-1]:.0f}")
                st.metric("🧮 Pérdida final (%)", f"{df['Yield_loss_%'].iloc[-1]:.1f}")
                st.download_button("📥 CSV", df.to_csv(index=False).encode(), "simulacion_v312.csv","text/csv")

    # -------------------------------------------------------------
    # 🔸 Optimización de fechas
    # -------------------------------------------------------------
    else:
        st.sidebar.header("Espacio de búsqueda (fechas)")
        preR_enable   = st.sidebar.checkbox("Optimizar preR (−30 a −14)", True)
        preemR_enable = st.sidebar.checkbox("Optimizar preemR (0 a +10)", True)
        postR_enable  = st.sidebar.checkbox("Optimizar postR (≥ +20)", True)
        gram_enable   = st.sidebar.checkbox("Optimizar gram (+25 a +35)", True)

        resol = st.sidebar.selectbox("Resolución", ["Fina","Media","Gruesa"], index=1)
        search_mode = st.sidebar.selectbox("Estrategia", ["Grid (completo)", "Muestreo aleatorio"], index=0)
        max_evals = st.sidebar.number_input("Máx. escenarios a evaluar", 10, 20000, 2000, 10)

        # Helpers de candidatos con reglas fenológicas reales
        def _candidates_preR(step):
            dates = []
            cur = sow_date - dt.timedelta(days=30)
            end = sow_date - dt.timedelta(days=14)
            while cur <= end:
                dates.append(cur)
                cur += dt.timedelta(days=int(step))
            return dates

        def _candidates_preemR(step):
            return [sow_date + dt.timedelta(days=d) for d in range(0, 11, int(step)) if 0 <= d <= 10]

        def _candidates_postR(step):
            dates=[]
            cur = sow_date + dt.timedelta(days=20)
            end = sow_date + dt.timedelta(days=180)
            while cur <= end:
                dates.append(cur)
                cur += dt.timedelta(days=int(step))
            return dates

        def _candidates_gram(step):
            dates=[]
            cur = sow_date + dt.timedelta(days=25)
            end = sow_date + dt.timedelta(days=35)
            while cur <= end:
                dates.append(cur)
                cur += dt.timedelta(days=int(step))
            return dates

        run_opt = st.sidebar.button("🚀 Ejecutar optimización")

        if run_opt:
            # Paso según resolución (solo al ejecutar)
            if resol == "Fina":
                step_preR, step_preem, step_postR, step_gram = 3, 2, 5, 2
            elif resol == "Media":
                step_preR, step_preem, step_postR, step_gram = 7, 4, 10, 5
            else:
                step_preR, step_preem, step_postR, step_gram = 14, 6, 20, 10

            preR_list   = _candidates_preR(step_preR)    if preR_enable   else [None]
            preem_list  = _candidates_preemR(step_preem) if preemR_enable else [None]
            postR_list  = _candidates_postR(step_postR)  if postR_enable  else [None]
            gram_list   = _candidates_gram(step_gram)    if gram_enable   else [None]

            # Estimación de combinaciones
            n_preR  = len(preR_list)   if preR_enable   else 1
            n_preem = len(preem_list)  if preemR_enable else 1
            n_postR = len(postR_list)  if postR_enable  else 1
            n_gram  = len(gram_list)   if gram_enable   else 1
            n_total = n_preR * n_preem * n_postR * n_gram
            st.info(f"🧮 Se evaluarán hasta **{min(n_total, max_evals):,}** combinaciones (de {n_total:,} posibles).")

            combos = list(itertools.product(preR_list, preem_list, postR_list, gram_list))
            if search_mode.startswith("Muestreo"):
                random.seed(123)
                if len(combos) > max_evals:
                    combos = random.sample(combos, int(max_evals))
            else:
                if len(combos) > max_evals:
                    st.warning(f"Grid muy grande: {len(combos):,} → se recorta a {int(max_evals):,}")
                    combos = combos[:int(max_evals)]

            results = []
            prog = st.progress(0.0)
            for i,(d_preR, d_preem, d_postR, d_gram) in enumerate(combos, 1):
                params = dict(preR_date=d_preR, preemR_date=d_preem, postR_date=d_postR, gram_date=d_gram)
                loss = objective_loss(params, base_kwargs)
                if np.isfinite(loss):
                    results.append({"preR":d_preR, "preemR":d_preem, "postR":d_postR, "gram":d_gram, "loss_pct":loss})
                if i % max(1, len(combos)//100) == 0 or i == len(combos):
                    prog.progress(min(1.0, i/len(combos)))
            prog.progress(1.0)

            if not results:
                st.error("No se encontraron combinaciones válidas con las reglas.")
            else:
                res_df = pd.DataFrame(results).sort_values("loss_pct").reset_index(drop=True)
                best = res_df.iloc[0]
                st.success(f"Mejor pérdida: {best['loss_pct']:.2f}%")
                c1,c2 = st.columns(2)
                with c1:
                    st.dataframe(res_df.head(15), use_container_width=True)
                with c2:
                    st.download_button("📥 Descargar tabla completa (CSV)",
                                       res_df.to_csv(index=False).encode(),
                                       "opt_fechas_resultados_v312.csv","text/csv")

                # Re-simular mejor
                df_best = simulate_with_controls(**base_kwargs,
                                                 preR_date=best["preR"] if pd.notna(best["preR"]) else None,
                                                 preemR_date=best["preemR"] if pd.notna(best["preemR"]) else None,
                                                 postR_date=best["postR"] if pd.notna(best["postR"]) else None,
                                                 gram_date=best["gram"] if pd.notna(best["gram"]) else None)
                if df_best is not None and not df_best.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_best["date"], y=df_best["Yield_abs_kg_ha"], name="Rinde (kg/ha)"))
                    fig.add_trace(go.Scatter(x=df_best["date"], y=df_best["Yield_loss_%"], name="Pérdida (%)", yaxis="y2"))
                    fig.update_layout(title="Mejor escenario — Rinde y Pérdida",
                                      xaxis_title="Fecha", yaxis_title="Rinde (kg/ha)",
                                      yaxis2=dict(title="Pérdida (%)", overlaying="y", side="right"),
                                      template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    st.metric("💰 Rinde final (kg/ha)", f"{df_best['Yield_abs_kg_ha'].iloc[-1]:.0f}")
                    st.metric("🧮 Pérdida final (%)", f"{df_best['Yield_loss_%'].iloc[-1]:.1f}")
                    st.download_button("📥 CSV (mejor escenario)",
                                       df_best.to_csv(index=False).encode(),
                                       "simulacion_mejor_escenario_v312.csv","text/csv")
        else:
            st.info("Ajustá el espacio de búsqueda y presioná **Ejecutar optimización**.")
