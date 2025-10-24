# ===============================================================
# BLOQUE 2 — Streamlit + Optimización de fechas (v3.15.1)
# ===============================================================
import streamlit as st
import plotly.graph_objects as go
import itertools, random, datetime as dt
import numpy as np
import pandas as pd

from datetime import date as _date

def objective_loss(params, base_kwargs):
    sim = simulate_with_controls(**base_kwargs, **params)
    if sim is None or sim.empty:
        return np.inf, None
    return float(sim.attrs.get("Loss_final_pct", sim["Loss_running_%"].iloc[-1])), sim

st.set_page_config(page_title="WeedCropSystem v3.15.1", layout="wide")
st.title("🌾 WeedCropSystem v3.15.1 — AUC ponderado por sensibilidad + Optimización")

# ----- Parámetros base -----
st.sidebar.header("⚙️ Parámetros base")
nyears = st.sidebar.slider("Años a simular", 1, 3, 1)
seed_bank0 = st.sidebar.number_input("Banco inicial (semillas·m⁻²)", 0, 50000, 4500)
K = st.sidebar.number_input("Capacidad de carga K (pl·m⁻²)", 50, 2000, 250)
Tb = st.sidebar.number_input("Temp. base Tb (°C)", 0.0, 15.0, 0.0, 0.5)
sim_seed = st.sidebar.number_input("Semilla aleatoria clima", 0, 999999, 42)
sow_date = st.sidebar.date_input("Fecha de siembra", _date(2025,6,1))

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
alpha, Lmax = 0.9782, 83.77  # fijos (calibrados)

st.sidebar.header("Eficacias fijas (%)")
preR_eff = st.sidebar.slider("Presiembra (S1–S2)", 0, 100, 90, 1)
preemR_eff = st.sidebar.slider("Preemergente (S1–S2)", 0, 100, 70, 1)
postR_eff = st.sidebar.slider("Post residual (S1–S4)", 0, 100, 85, 1)
gram_eff = st.sidebar.slider("Graminicida (S1–S3)", 0, 100, 80, 1)

st.sidebar.header("Residualidades (días)")
preR_residual = st.sidebar.slider("preR", 10, 180, 30, 1)
preemR_residual = st.sidebar.slider("preemR", 10, 180, 30, 1)
postR_residual = st.sidebar.slider("postR", 10, 180, 30, 1)
gram_residual_forward = 11  # fijo (ventana corta)

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

            # Gráfico 1: rinde y pérdida (running)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["date"], y=df["Yield_abs_kg_ha"], name="Rinde (kg/ha)"))
            fig.add_trace(go.Scatter(x=df["date"], y=df["Loss_running_%"], name="Pérdida (%)", yaxis="y2"))
            fig.update_layout(
                title="Rinde y Pérdida de rinde (running) — AUC ponderado",
                xaxis_title="Fecha",
                yaxis_title="Rinde (kg/ha)",
                yaxis2=dict(title="Pérdida (%)", overlaying="y", side="right"),
                template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Gráfico 2: competencia (WC) y AUC ponderado acumulado
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df["date"], y=df["WC"], name="Competencia instantánea (WC)"))
            fig2.add_trace(go.Scatter(x=df["date"], y=df["AUC_weighted"], name="AUC ponderado (acum)"))
            fig2.update_layout(title="Dinámica de competencia y AUC ponderado (trapezoidal)",
                               template="plotly_white",
                               xaxis_title="Fecha", yaxis_title="Índice / Área")
            st.plotly_chart(fig2, use_container_width=True)

            # Métricas finales
            auc_final = df.attrs.get("AUC_weighted_final", float(df["AUC_weighted"].iloc[-1]))
            loss_final = df.attrs.get("Loss_final_pct", float(df["Loss_running_%"].iloc[-1]))
            st.metric("🧮 AUC ponderado final", f"{auc_final:.2f}")
            st.metric("💰 Rinde final", f"{df['Yield_abs_kg_ha'].iloc[-1]:.0f} kg/ha")
            st.metric("📉 Pérdida final", f"{loss_final:.2f}%")

            st.download_button("📥 Descargar CSV", df.to_csv(index=False).encode(),
                               "simulacion_v3151.csv","text/csv")

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
        days = list(range(30, 13, -int(step_preR)))  # 30..14
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

        rows = []
        prog = st.progress(0.0)
        for i,(d_preR, d_preem, d_postR, d_gram) in enumerate(combos, 1):
            params = dict(preR_date=d_preR, preemR_date=d_preem, postR_date=d_postR, gram_date=d_gram)
            loss, sim = objective_loss(params, base_kwargs)
            if np.isfinite(loss):
                rows.append({
                    "preR": d_preR, "preemR": d_preem, "postR": d_postR, "gram": d_gram,
                    "loss_pct": loss,
                    "AUC_weighted": None if (sim is None or sim.empty) else float(sim.attrs.get("AUC_weighted_final", sim["AUC_weighted"].iloc[-1])),
                    "WC_max": None if (sim is None or sim.empty) else float(sim.attrs.get("WC_max", sim["WC"].max()))
                })
            if i % max(1, len(combos)//100) == 0:
                prog.progress(min(1.0, i/len(combos)))
        prog.progress(1.0)

        if not rows:
            st.error("No se encontraron combinaciones válidas con las reglas.")
        else:
            res_df = pd.DataFrame(rows).sort_values("loss_pct").reset_index(drop=True)
            best = res_df.iloc[0]
            st.success(f"🏆 Mejor pérdida de rinde: {best['loss_pct']:.2f}%  |  AUC*: {best['AUC_weighted']:.2f}  |  WC_max: {best['WC_max']:.2f}")
            st.dataframe(res_df.head(15), use_container_width=True)
            st.download_button("📥 Descargar resultados CSV",
                               res_df.to_csv(index=False).encode(),
                               "opt_fechas_resultados_v3151.csv","text/csv")

            # Re-simular el mejor escenario
            df_best = simulate_with_controls(**base_kwargs,
                preR_date=best["preR"], preemR_date=best["preemR"],
                postR_date=best["postR"], gram_date=best["gram"])
            if df_best is not None and not df_best.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_best["date"], y=df_best["Yield_abs_kg_ha"], name="Rinde (kg/ha)"))
                fig.add_trace(go.Scatter(x=df_best["date"], y=df_best["Loss_running_%"], name="Pérdida (%)", yaxis="y2"))
                fig.update_layout(title="📈 Mejor escenario — Rinde y Pérdida (running)",
                                  template="plotly_white",
                                  xaxis_title="Fecha", yaxis_title="Rinde (kg/ha)",
                                  yaxis2=dict(title="Pérdida (%)", overlaying="y", side="right"))
                st.plotly_chart(fig, use_container_width=True)

                auc_final = df_best.attrs.get("AUC_weighted_final", float(df_best["AUC_weighted"].iloc[-1]))
                st.metric("🧮 AUC ponderado final", f"{auc_final:.2f}")
                st.metric("💰 Rinde final", f"{df_best['Yield_abs_kg_ha'].iloc[-1]:.0f} kg/ha")
                st.metric("🧮 Pérdida final", f"{df_best['Loss_running_%'].iloc[-1]:.2f}%")
    else:
        st.info("Ajustá el espacio de búsqueda y presioná “Ejecutar optimización”.")




