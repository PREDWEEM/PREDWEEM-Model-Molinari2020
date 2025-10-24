# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 WeedCropSystem — v3.10 (Optimización de fechas con reglas)
# ---------------------------------------------------------------
# - Base: v3.9.1 (una especie, emergencia ×8, curva de pérdida α/Lmax)
# - NUEVO: Optimizador de fechas de aplicación (grid/aleatorio)
# - Reglas exactas (como PREDWEEM):
#   · preR: SOLO ≤ siembra−14 (actúa S1–S2; residual)
#   · preemR: [siembra, siembra+10] (S1–S2; residual)
#   · postR: ≥ siembra+20 (S1–S4; residual)
#   · gram: [siembra, siembra+10] (S1–S3; ventana de 11 días: día de app + 10)
# - Gateo por remanente + exclusión jerárquica (99%) por DÍA:
#   preR → preemR → postR → gram
# ===============================================================

import sys, datetime as dt
import numpy as np
import pandas as pd

# =========================
# Núcleo de simulación base
# =========================
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

# Helpers de ventanas (incluye preemR y regla gram con +10 días)
def _date_range(start_date, days):
    return {start_date + dt.timedelta(days=i) for i in range(int(days))}

def _one_day(date):
    return {date}

# ================================
# Simulador con reglas de control
# ================================
def simulate_with_controls(
    # agronomía/ambiente
    nyears=1, seed_bank0=4500, K=250, Tb=0.0, seed=42,
    sow_date=dt.date(2025,6,1),
    # canopia / competencia
    LAI_max=6.0, t_lag=10, t_close=35, LAI_hc=6.0, Cs=200, Ca=200,
    p_S1=1.0, p_S2=0.6, p_S3=0.4, p_S4=0.2,
    # pesos de WC
    w_S1=0.15, w_S2=0.30, w_S3=0.60, w_S4=1.00,
    # pérdida de rinde
    alpha=0.9782, Lmax=83.77, GY_pot=6000.0,
    # eficacia fija (usuario puede cambiarlas en UI si desea)
    preR_eff=90, preemR_eff=70, postR_eff=85, gram_eff=80,
    # residualidades (días)
    preR_residual=30, preemR_residual=30, postR_residual=30, gram_residual_forward=11,
    # FECHAS (pueden ser None si no se aplican)
    preR_date=None, preemR_date=None, postR_date=None, gram_date=None,
    # validación de reglas (apagar para escenarios libres)
    enforce_rules=True
):
    EPS_REMAIN = 1e-9
    EPS_EXCLUDE = 0.99

    sow = pd.to_datetime(sow_date).date()
    start = sow - dt.timedelta(days=90)
    end = dt.date(sow.year + int(nyears) - 1, 12, 1)
    meteo = synthetic_meteo(start, end, seed)

       # ===============================================================
    # 🔍 Validación de reglas de aplicación (v3.12 — reglas fenológicas reales)
    # ===============================================================
    # Definición de ventanas según la siembra (sow_date) y desarrollo térmico estimado:
    # - PreR: entre 30 y 14 días antes de la siembra
    # - PreemR: entre siembra y emergencia del cultivo (≈ +200 °Cd ≈ 7–10 días)
    # - PostR: desde 2 hojas del cultivo (≈ +300 °Cd ≈ +15–20 días)
    # - Gram: entre 3 y 4 hojas (≈ +400 °Cd ≈ +20–25 días)
    #
    # Nota: los límites fenológicos se aproximan en días calendario a partir de siembra.

    # Límite inferior y superior de cada ventana (en días relativos)
    preR_start  = sow - dt.timedelta(days=30)
    preR_end    = sow - dt.timedelta(days=14)
    preemR_start = sow
    preemR_end   = sow + dt.timedelta(days=10)      # hasta emergencia del cultivo
    postR_start  = sow + dt.timedelta(days=20)      # ≈ 2 hojas
    postR_end    = sow + dt.timedelta(days=180)     # hasta fin de ciclo
    gram_start   = sow + dt.timedelta(days=25)      # 3 hojas
    gram_end     = sow + dt.timedelta(days=35)      # 4 hojas

    # ====== Validación de reglas ======
    if enforce_rules:
        if preR_date is not None and not (preR_start <= preR_date <= preR_end):
            return None
        if preemR_date is not None and not (preemR_start <= preemR_date <= preemR_end):
            return None
        if postR_date is not None and not (postR_start <= postR_date <= postR_end):
            return None
        if gram_date is not None and not (gram_start <= gram_date <= gram_end):
            return None

    # Ventanas (sets de fechas) según duración residual
    preR_window   = set() if preR_date is None else _date_range(preR_date, preR_residual)
    preemR_window = set() if preemR_date is None else _date_range(preemR_date, preemR_residual)
    postR_window  = set() if postR_date is None else _date_range(postR_date, postR_residual)
    gram_window   = set() if gram_date is None else _date_range(gram_date, gram_residual_forward)


    # Ventanas (sets de fechas)
    preR_window   = set() if preR_date is None else _date_range(preR_date, preR_residual)
    preemR_window = set() if preemR_date is None else _date_range(preemR_date, preemR_residual)
    postR_window  = set() if postR_date is None else _date_range(postR_date, postR_residual)
    gram_window   = set() if gram_date is None else _date_range(gram_date, gram_residual_forward)

    Sq = float(seed_bank0)
    TTw = 0.0
    W = [0,0,0,0,0]  # S1..S5
    Th = [70, 280, 400, 300]  # umbrales de pasaje por TT
    out = []

    for _, row in meteo.iterrows():
        date = pd.to_datetime(row["date"]).date()
        dss = (date - sow).days
        Tmean = (float(row["tmin"])+float(row["tmax"])) / 2
        TTw += max(Tmean - float(Tb), 0)

        Ciec_t, LAI_t = ciec_calendar(dss, LAI_max, t_lag, t_close, LAI_hc, Cs, Ca)

        # Emergencia ×8 para efecto visible
        E_t = 8.0 * emergence_simple(TTw, float(row["prec"]))

        # Población efectiva (intra-específica) + supresión
        Wk = sum(np.array(W)*np.array([0.15,0.3,0.6,1.0,0.0]))
        surv_intra = 1 - min(Wk/K,1)

        # Ingreso a S1 (afectado por supresión S1)
        sup_S1 = (1-Ciec_t)**p_S1
        I1_t = max(0, Sq * E_t * surv_intra * sup_S1)

        # --- Copia controlable de estados (antes de aplicar controles del día) ---
        Wc = W.copy()  # Wc[0..4] → S1..S5 (control del día)
        # Supresión por canopia (S2..S4)
        Wc[1] *= (1-Ciec_t)**p_S2
        Wc[2] *= (1-Ciec_t)**p_S3
        Wc[3] *= (1-Ciec_t)**p_S4

        # ---------- Gateo por remanente + exclusión jerárquica (diario) ----------
        eff_accum = 0.0  # acumulador jerárquico del día (0..1)

        def _rem_in(states):
            tot = 0.0
            if "S1" in states: tot += Wc[0]
            if "S2" in states: tot += Wc[1]
            if "S3" in states: tot += Wc[2]
            if "S4" in states: tot += Wc[3]
            return float(tot)

        def _apply_eff(eff_pct, states):
            nonlocal eff_accum
            if eff_pct <= 0: return
            if eff_accum >= EPS_EXCLUDE: return
            rem = _rem_in(states)
            if rem <= EPS_REMAIN: return
            f = max(0.0, 1.0 - (eff_pct/100.0))
            # aplicar reducción a los estados seleccionados
            if "S1" in states: Wc[0] *= f
            if "S2" in states: Wc[1] *= f
            if "S3" in states: Wc[2] *= f
            if "S4" in states: Wc[3] *= f
            # combinar eficacias como independencia: 1-(1-a)(1-b)...
            eff_accum = 1.0 - (1.0 - eff_accum) * (1.0 - eff_pct/100.0)

        # Aplicación por orden jerárquico SOLO si el día cae en su ventana:
        if date in preR_window:   _apply_eff(preR_eff,   ["S1","S2"])
        if date in preemR_window: _apply_eff(preemR_eff, ["S1","S2"])
        if date in postR_window:  _apply_eff(postR_eff,  ["S1","S2","S3","S4"])
        if date in gram_window:   _apply_eff(gram_eff,   ["S1","S2","S3"])

        # ---------- Transiciones fenológicas (usando W controlado + I1_t) ----------
        O1 = I1_t if TTw>=Th[0] else 0
        O2 = Wc[1] if TTw>=sum(Th[:2]) else 0
        O3 = Wc[2] if TTw>=sum(Th[:3]) else 0
        O4 = Wc[3] if TTw>=sum(Th[:4]) else 0

        W1 = max(0, Wc[0] + I1_t - O1)
        W2 = max(0, Wc[1] + O1   - O2)
        W3 = max(0, Wc[2] + O2   - O3)
        W4 = max(0, Wc[3] + O3   - O4)
        W5 = max(0, Wc[4] + O4)

        W = [W1,W2,W3,W4,W5]

        out.append({
            "date":date,"days_since_sow":dss,"TTw":TTw,"Ciec":Ciec_t,"LAI":LAI_t,
            "W1":W1,"W2":W2,"W3":W3,"W4":W4,"W5":W5
        })

    df = pd.DataFrame(out)
    df["W_total"] = df[["W1","W2","W3","W4"]].sum(axis=1)
    df["WC"] = w_S1*df["W1"] + w_S2*df["W2"] + w_S3*df["W3"] + w_S4*df["W4"]
    df["Yield_loss_%"] = (alpha*df["WC"])/(1+(alpha*df["WC"]/Lmax))
    df["Yield_relative_%"] = 100 - df["Yield_loss_%"]
    df["Yield_abs_kg_ha"] = GY_pot * (df["Yield_relative_%"]/100)
    return df

# ===============================================
# Objetivo para optimización (minimizar pérdida %)
# ===============================================
def objective_loss(params, base_kwargs):
    """ params: dict con fechas (o None) y residualidades (enteros).
        base_kwargs: argumentos fijos (sow_date, LAI, etc.). """
    sim = simulate_with_controls(**base_kwargs, **params)
    if sim is None or sim.empty:
        return np.inf
    return float(sim["Yield_loss_%"].iloc[-1])

# ============================
# --------- STREAMLIT --------
# ============================
if "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv):
    import streamlit as st
    import plotly.graph_objects as go

    st.set_page_config(page_title="WeedCropSystem v3.10", layout="wide")
    st.title("🌾 WeedCropSystem — v3.10 (Optimización de fechas con reglas)")

    # ----- Panel lateral: parámetros base -----
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
    Cs=st.sidebar.number_input("Cs (pl/m²)",50,800,200)
    Ca=st.sidebar.number_input("Ca (pl/m²)",30,800,200)

    st.sidebar.subheader("⚖️ Supresión (exp de (1−Ciec))")
    p_S1=st.sidebar.slider("S1",0.0,2.0,1.0,0.1)
    p_S2=st.sidebar.slider("S2",0.0,2.0,0.6,0.1)
    p_S3=st.sidebar.slider("S3",0.0,2.0,0.4,0.1)
    p_S4=st.sidebar.slider("S4",0.0,2.0,0.2,0.1)

    st.sidebar.subheader("🌾 Rinde potencial y pérdida")
    gy_option = st.sidebar.selectbox("Cultivo:", ["Trigo (6000 kg/ha)","Cebada (7000 kg/ha)","Personalizado"])
    GY_pot = 6000.0 if "Trigo" in gy_option else (7000.0 if "Cebada" in gy_option else st.sidebar.number_input("GY_pot (kg/ha)", 1000, 15000, 6000, 100))
    alpha=0.9782; Lmax=83.77

    st.sidebar.divider()
    st.sidebar.header("Eficacias fijas (%)")
    preR_eff = st.sidebar.slider("preR (S1–S2)", 0, 100, 90, 1)
    preemR_eff = st.sidebar.slider("preemR (S1–S2)", 0, 100, 70, 1)
    postR_eff = st.sidebar.slider("postR (S1–S4)", 0, 100, 85, 1)
    gram_eff = st.sidebar.slider("gram (S1–S3)", 0, 100, 80, 1)

    st.sidebar.header("Residualidades (días)")
    preR_residual = st.sidebar.slider("preR residual", 10, 180, 30, 1)
    preemR_residual = st.sidebar.slider("preemR residual", 10, 180, 30, 1)
    postR_residual = st.sidebar.slider("postR residual", 10, 180, 30, 1)
    gram_residual_forward = 11  # fijo por regla (día de app +10)

    # =======================
    # MODO: Simular u Optimizar
    # =======================
    mode = st.sidebar.selectbox("Modo:", ["Simulación única", "Optimización (fechas)"], index=0)

    base_kwargs = dict(
        nyears=nyears, seed_bank0=seed_bank0, K=K, Tb=Tb, seed=sim_seed,
        sow_date=sow_date, LAI_max=LAI_max, t_lag=t_lag, t_close=t_close,
        LAI_hc=LAI_hc, Cs=Cs, Ca=Ca, p_S1=p_S1, p_S2=p_S2, p_S3=p_S3, p_S4=p_S4,
        w_S1=0.15, w_S2=0.30, w_S3=0.60, w_S4=1.00,
        alpha=alpha, Lmax=Lmax, GY_pot=GY_pot,
        preR_eff=preR_eff, preemR_eff=preemR_eff, postR_eff=postR_eff, gram_eff=gram_eff,
        preR_residual=preR_residual, preemR_residual=preemR_residual, postR_residual=postR_residual,
        gram_residual_forward=gram_residual_forward,
        enforce_rules=True
    )

    # =======================
    # --- SIMULACIÓN ÚNICA ---
    # =======================
    if mode == "Simulación única":
        st.sidebar.subheader("Fechas de aplicación (opcionales)")
        preR_date = st.sidebar.date_input("preR (≤ siembra−14)", value=None, min_value=sow_date-dt.timedelta(days=180), max_value=sow_date, key="preR") if st.sidebar.checkbox("Usar preR", False) else None
        preemR_date = st.sidebar.date_input("preemR [siembra..siembra+10]", value=sow_date, min_value=sow_date, max_value=sow_date+dt.timedelta(days=10), key="preemR") if st.sidebar.checkbox("Usar preemR", False) else None
        postR_date = st.sidebar.date_input("postR (≥ siembra+20)", value=sow_date+dt.timedelta(days=20), min_value=sow_date+dt.timedelta(days=20), max_value=sow_date+dt.timedelta(days=180), key="postR") if st.sidebar.checkbox("Usar postR", False) else None
        gram_date = st.sidebar.date_input("gram [siembra..siembra+10]", value=sow_date, min_value=sow_date, max_value=sow_date+dt.timedelta(days=10), key="gram") if st.sidebar.checkbox("Usar gram", False) else None

        if st.sidebar.button("▶ Ejecutar simulación"):
            df = simulate_with_controls(**base_kwargs,
                                        preR_date=preR_date, preemR_date=preemR_date,
                                        postR_date=postR_date, gram_date=gram_date)
            if df is None or df.empty:
                st.error("Configuración inválida con las reglas de fechas.")
            else:
                st.success(f"Simulación OK — {len(df)} días")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df["date"], y=df["Yield_abs_kg_ha"], name="Rinde (kg/ha)"))
                fig1.add_trace(go.Scatter(x=df["date"], y=df["Yield_loss_%"], name="Pérdida (%)", yaxis="y2"))
                fig1.update_layout(title="Rinde y Pérdida (%)", xaxis_title="Fecha",
                                   yaxis_title="Rinde (kg/ha)",
                                   yaxis2=dict(title="Pérdida (%)", overlaying="y", side="right"),
                                   template="plotly_white")
                st.plotly_chart(fig1, use_container_width=True)
                st.metric("💰 Rinde final (kg/ha)", f"{df['Yield_abs_kg_ha'].iloc[-1]:.0f}")
                st.metric("🧮 Pérdida final (%)", f"{df['Yield_loss_%'].iloc[-1]:.1f}")
                st.download_button("📥 CSV", df.to_csv(index=False).encode(), "simulacion_v310.csv","text/csv")

    # =====================
    # --- OPTIMIZACIÓN  ---
    # =====================
    else:
        st.sidebar.header("Espacio de búsqueda (fechas)")
        preR_enable = st.sidebar.checkbox("Optimizar preR (≤ siembra−14)", True)
        preemR_enable = st.sidebar.checkbox("Optimizar preemR [siembra..siembra+10]", True)
        postR_enable = st.sidebar.checkbox("Optimizar postR (≥ siembra+20)", True)
        gram_enable = st.sidebar.checkbox("Optimizar gram [siembra..siembra+10]", True)

        step_preR = st.sidebar.number_input("Paso preR (días)", 1, 30, 7, 1)
        step_preem = st.sidebar.number_input("Paso preemR (días)", 1, 10, 3, 1)
        step_postR = st.sidebar.number_input("Paso postR (días)", 1, 30, 7, 1)
        step_gram = st.sidebar.number_input("Paso gram (días)", 1, 10, 3, 1)

        max_evals = st.sidebar.number_input("Máx. escenarios a evaluar", 10, 20000, 2000, 10)
        search_mode = st.sidebar.selectbox("Estrategia", ["Grid (completo)", "Muestreo aleatorio"], index=0)
        run_opt = st.sidebar.button("🚀 Ejecutar optimización")

        def _candidates_preR():
            # desde siembra-180 hasta siembra-14, step_preR
            dates = []
            cur = sow_date - dt.timedelta(days=180)
            end = sow_date - dt.timedelta(days=14)
            while cur <= end:
                dates.append(cur)
                cur += dt.timedelta(days=int(step_preR))
            return dates

        def _candidates_preemR():
            return [sow_date + dt.timedelta(days=d) for d in range(0, 11, int(step_preem)) if 0 <= d <= 10]

        def _candidates_postR():
            dates=[]
            cur = sow_date + dt.timedelta(days=20)
            end = sow_date + dt.timedelta(days=180)
            while cur <= end:
                dates.append(cur)
                cur += dt.timedelta(days=int(step_postR))
            return dates

        def _candidates_gram():
            return [sow_date + dt.timedelta(days=d) for d in range(0, 11, int(step_gram)) if 0 <= d <= 10]

        if run_opt:
            import itertools, random
            preR_list   = _candidates_preR()   if preR_enable   else [None]
            preem_list  = _candidates_preemR() if preemR_enable else [None]
            postR_list  = _candidates_postR()  if postR_enable  else [None]
            gram_list   = _candidates_gram()   if gram_enable   else [None]

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
                if loss != np.inf:
                    results.append({"preR":d_preR, "preemR":d_preem, "postR":d_postR, "gram":d_gram, "loss_pct":loss})
                if i % max(1, len(combos)//100) == 0:
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
                                       "opt_fechas_resultados.csv","text/csv")

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
                                       "simulacion_mejor_escenario_v310.csv","text/csv")
        else:
            st.info("Ajustá el espacio de búsqueda y presioná “Ejecutar optimización”.")
