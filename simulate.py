# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 WeedCropSystem — Controles con visualización (modo original)
# ---------------------------------------------------------------
# - Ciec dinámico por días calendario (LAI logístico)
# - Supresión del cultivo S1–S4 (sin TT modificado)
# - Herbicidas: letal + residual (preR, postR, gram)
# - Visualización: bandas de aplicación + mortalidad por control
# - Clima sintético (reproducible)
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as go

# ---------- CONFIG ----------
st.set_page_config(page_title="🌾 WeedCropSystem — Visual", layout="wide")
st.title("🌾 WeedCropSystem — Efecto de herbicidas sobre individuos y estados (S1–S4)")

# ---------- SIDEBAR ----------
st.sidebar.header("Escenario")
nyears = st.sidebar.slider("Años a simular", 1, 10, 3)
seed_bank0 = st.sidebar.number_input("Banco inicial (semillas·m⁻²)", 0, 20000, 4500)
Tb = st.sidebar.number_input("Temp. base Tb (°C)", 0.0, 15.0, 0.0, 0.5)
K = st.sidebar.number_input("Capacidad de carga K (pl·m⁻²)", 50, 2000, 250)
sim_seed = st.sidebar.number_input("Semilla aleatoria clima", 0, 999999, 42)

st.sidebar.divider()
st.sidebar.subheader("📅 Fechas")
sow_date = st.sidebar.date_input("Fecha de siembra", dt.date(2025, 6, 1))

st.sidebar.divider()
st.sidebar.subheader("🌿 Canopia (LAI logístico)")
LAI_max = st.sidebar.slider("LAI_max", 2.0, 10.0, 6.0, 0.1)
t_lag = st.sidebar.slider("t_lag (d desde siembra)", 0, 60, 10)
t_close = st.sidebar.slider("t_close (d desde siembra, LAI≈0.5·max)", 10, 100, 35)
LAI_hc = st.sidebar.slider("LAI_hc", 2.0, 10.0, 6.0, 0.1)
Cs = st.sidebar.number_input("Cs (estándar, pl·m⁻²)", 50, 800, 200)
Ca = st.sidebar.number_input("Ca (real, pl·m⁻²)", 30, 800, 200)

st.sidebar.divider()
st.sidebar.subheader("⚖️ Supresión por estadio (exp de (1−Ciec))")
p_S1 = st.sidebar.slider("S1", 0.0, 2.0, 1.0, 0.1)
p_S2 = st.sidebar.slider("S2", 0.0, 2.0, 0.6, 0.1)
p_S3 = st.sidebar.slider("S3", 0.0, 2.0, 0.4, 0.1)
p_S4 = st.sidebar.slider("S4", 0.0, 2.0, 0.2, 0.1)

st.sidebar.divider()
st.sidebar.subheader("🧪 Controles (modo original)")
# Presiembra residual
preR_days_before = st.sidebar.number_input("PreR: días antes siembra", 0, 90, 14)
preR_eff_S1S2 = st.sidebar.slider("PreR eficacia S1–S2 (%)", 0, 100, 90)
preR_residual = st.sidebar.slider("PreR residual (d)", 0, 120, 30)
# Postemergente
postR_days_after = st.sidebar.number_input("PostR: días después siembra", 0, 120, 25)
postR_eff_S1S4 = st.sidebar.slider("PostR eficacia S1–S4 (%)", 0, 100, 85)
postR_residual = st.sidebar.slider("PostR residual (d)", 0, 120, 10)
# Graminicida
gram_days_after = st.sidebar.number_input("Gram: días después siembra", 0, 120, 10)
gram_eff_S1S3 = st.sidebar.slider("Gram eficacia S1–S3 (%)", 0, 100, 80)
gram_residual = st.sidebar.slider("Gram residual (d)", 0, 120, 7)

run_btn = st.sidebar.button("▶ Ejecutar simulación")

# ---------- CLIMA SINTÉTICO ----------
def synthetic_meteo(start: dt.date, end: dt.date, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    doy = dates.dayofyear.to_numpy()
    tmean = 12 + 8*np.sin(2*np.pi*(doy-170)/365.0) + rng.normal(0,1.5,n)
    tmin = tmean - (3 + rng.normal(0,0.8,n))
    tmax = tmean + (6 + rng.normal(0,0.8,n))
    prec = rng.choice([0,0,0,0,3,8,15], size=n,
                      p=[0.55,0.15,0.10,0.05,0.07,0.05,0.03])
    df = pd.DataFrame({"date":dates, "tmin":tmin, "tmax":tmax, "prec":prec})
    df["date"] = pd.to_datetime(df["date"])
    return df

# ---------- EMERGENCIA (placeholder) ----------
def emergence_simple(TT: float, prec: float) -> float:
    base = 1.0/(1.0 + np.exp(-(TT-300)/40.0))
    pulse = 0.002 if float(prec) >= 5.0 else 0.0
    return float(min(base*0.003 + pulse, 0.02))

# ---------- LAI / Ciec ----------
def lai_logistic_by_day(days_since_sow: int, LAI_max: float, t_lag: int, t_close: int):
    t50 = max(int(t_close), 1)
    eps = 0.05
    denom = (int(t_lag) - t50)
    if abs(denom) < 1e-6:
        denom = -1.0
    k = -np.log(1/eps - 1) / denom
    LAI = LAI_max / (1.0 + np.exp(-k*(float(days_since_sow) - t50)))
    LAI = float(max(0.0, min(LAI, LAI_max)))
    return LAI, float(k)

def ciec_calendar(days_since_sow: int, LAI_max: float, t_lag: int, t_close: int,
                  LAI_hc: float, Cs: float, Ca: float):
    LAI, k = lai_logistic_by_day(days_since_sow, LAI_max, t_lag, t_close)
    ratio = (float(Cs) / max(float(Ca), 1e-6))
    Ciec = min((LAI / max(float(LAI_hc), 1e-6)) * ratio, 1.0)
    return float(Ciec), LAI, k

# ---------- SIMULADOR ----------
def simulate(
    nyears=3, seed_bank0=4500, K=250, Tb=0.0, seed=42,
    preR_days_before=14, preR_eff=90, preR_residual=30,
    postR_days_after=25, postR_eff=85, postR_residual=10,
    gram_days_after=10, gram_eff=80, gram_residual=7,
    sow_date=dt.date(2025,6,1),
    LAI_max=6.0, t_lag=10, t_close=35, LAI_hc=6.0, Cs=200, Ca=200,
    p_S1=1.0, p_S2=0.6, p_S3=0.4, p_S4=0.2
) -> pd.DataFrame:

    sow = pd.to_datetime(sow_date).date()
    start = sow - dt.timedelta(days=int(preR_days_before))
    end = dt.date(sow.year + int(nyears) - 1, 12, 1)
    meteo = synthetic_meteo(start, end, seed)

    Sq = float(seed_bank0)
    TTw = 0.0
    W = [0.0, 0.0, 0.0, 0.0, 0.0]  # S1..S5
    out = []

    # Ventanas de control (como conjuntos, para acceso O(1))
    preR_date = sow - dt.timedelta(days=int(preR_days_before))
    postR_date = sow + dt.timedelta(days=int(postR_days_after))
    gram_date = sow + dt.timedelta(days=int(gram_days_after))
    preR_window = { preR_date + dt.timedelta(days=i) for i in range(int(preR_residual)) }
    postR_window = { postR_date + dt.timedelta(days=i) for i in range(int(postR_residual)) }
    gram_window = { gram_date + dt.timedelta(days=i) for i in range(int(gram_residual)) }

    # Umbrales térmicos (°Cd) para transiciones (modo original)
    Th = [70.0, 280.0, 400.0, 300.0]

    for _, row in meteo.iterrows():
        date = pd.to_datetime(row["date"]).date()
        dss = (date - sow).days
        Tmean = (float(row["tmin"]) + float(row["tmax"])) / 2.0
        TTw += max(Tmean - float(Tb), 0.0)

        # Ciec y supresión cultivo
        Ciec_t, LAI_t, _ = ciec_calendar(dss, float(LAI_max), int(t_lag), int(t_close),
                                         float(LAI_hc), float(Cs), float(Ca))

        # Emergencia (placeholder, aún sin ANN)
        E_t = emergence_simple(TTw, float(row["prec"]))

        # Competencia intraespecífica
        Wk = sum(np.array(W) * np.array([0.15,0.30,0.60,1.0,0.0]))
        surv_intra = 1.0 - min(Wk / float(K), 1.0)

        # Ingreso a S1 (supresión cultivo en S1)
        sup_S1 = (1.0 - Ciec_t)**float(p_S1)
        I1_t = max(0.0, Sq * E_t * surv_intra * sup_S1)

        # --------- Mortalidad por control (desglosada por tipo y estado) ----------
        # Implementación determinística, multiplicativa y secuencial (preR -> postR -> gram)
        # Nota: el paper usa efecto letal; aquí distribuimos la contribución por tipo para visualizar.
        eff_pre = [0.0]*5
        eff_post = [0.0]*5
        eff_gram = [0.0]*5

        if date in preR_window:
            eff_pre[0] = float(preR_eff)/100.0  # S1; podrías extender a S2 si corresponde
            eff_pre[1] = float(preR_eff)/100.0  # (opcional S1–S2 como en tu app previa)

        if date in postR_window:
            for i in range(4):  # S1–S4
                eff_post[i] = max(eff_post[i], float(postR_eff)/100.0)

        if date in gram_window:
            for i in range(3):  # S1–S3
                eff_gram[i] = max(eff_gram[i], float(gram_eff)/100.0)

        # Aplicamos mortalidad separando la contribución de cada control: preR → postR → gram
        W_before = W.copy()
        Mort_pre = [0.0]*5
        Mort_post = [0.0]*5
        Mort_gram = [0.0]*5

        # preR
        W_after_pre = [w * (1.0 - e) for w, e in zip(W_before, eff_pre)]
        Mort_pre = [max(0.0, wb - wa) for wb, wa in zip(W_before, W_after_pre)]

        # postR
        W_after_post = [w * (1.0 - e) for w, e in zip(W_after_pre, eff_post)]
        Mort_post = [max(0.0, wp - wa) for wp, wa in zip(W_after_pre, W_after_post)]

        # gram
        W_after_gram = [w * (1.0 - e) for w, e in zip(W_after_post, eff_gram)]
        Mort_gram = [max(0.0, wpo - wa) for wpo, wa in zip(W_after_post, W_after_gram)]

        # Estados tras mortalidad del día
        W_ctrl = W_after_gram

        # Supresión cultivo sobre S2–S4
        sup_S2 = (1.0 - Ciec_t)**float(p_S2)
        sup_S3 = (1.0 - Ciec_t)**float(p_S3)
        sup_S4 = (1.0 - Ciec_t)**float(p_S4)
        W_ctrl[1] *= sup_S2
        W_ctrl[2] *= sup_S3
        W_ctrl[3] *= sup_S4

        # Transiciones fenológicas dependientes solo de TT
        O1 = I1_t if TTw >= Th[0] else 0.0
        O2 = W_ctrl[1] if TTw >= (Th[0]+Th[1]) else 0.0
        O3 = W_ctrl[2] if TTw >= (Th[0]+Th[1]+Th[2]) else 0.0
        O4 = W_ctrl[3] if TTw >= (Th[0]+Th[1]+Th[2]+Th[3]) else 0.0

        # Actualización de estados
        W1 = max(0.0, W_ctrl[0] + I1_t - O1)
        W2 = max(0.0, W_ctrl[1] + O1 - O2)
        W3 = max(0.0, W_ctrl[2] + O2 - O3)
        W4 = max(0.0, W_ctrl[3] + O3 - O4)
        W5 = max(0.0, W_ctrl[4] + O4)
        W = [W1, W2, W3, W4, W5]

        out.append({
            "date": pd.to_datetime(date),
            "days_since_sow": int(dss),
            "TTw": float(TTw),
            "LAI": float(LAI_t), "Ciec": float(Ciec_t),
            "E_t": float(E_t), "I1_t": float(I1_t),
            "W1": float(W1), "W2": float(W2), "W3": float(W3), "W4": float(W4), "W5": float(W5),
            # mortalidad diaria por control (sumada S1–S4 y también por estado)
            "Mort_preR": float(sum(Mort_pre[:4])),
            "Mort_postR": float(sum(Mort_post[:4])),
            "Mort_gram": float(sum(Mort_gram[:4])),
            "Mort_preR_S1": float(Mort_pre[0]), "Mort_preR_S2": float(Mort_pre[1]),
            "Mort_postR_S1": float(Mort_post[0]), "Mort_postR_S2": float(Mort_post[1]),
            "Mort_postR_S3": float(Mort_post[2]), "Mort_postR_S4": float(Mort_post[3]),
            "Mort_gram_S1": float(Mort_gram[0]), "Mort_gram_S2": float(Mort_gram[1]), "Mort_gram_S3": float(Mort_gram[2]),
            # factores de supresión por estadio
            "sup_S1": float(sup_S1), "sup_S2": float(sup_S2), "sup_S3": float(sup_S3), "sup_S4": float(sup_S4),
        })

    df = pd.DataFrame(out).sort_values("date").reset_index(drop=True)

    # Agregados útiles para gráficos
    df["W_total_S1S4"] = df[["W1","W2","W3","W4"]].sum(axis=1)
    df["Mort_total"] = df[["Mort_preR","Mort_postR","Mort_gram"]].sum(axis=1)
    cum_cols = ["Mort_preR","Mort_postR","Mort_gram","Mort_total"]
    for c in cum_cols:
        df[c+"_cum"] = df[c].cumsum()

    # Proporciones por estado (evitar /0)
    eps = 1e-9
    for s in ["W1","W2","W3","W4"]:
        df[s+"_prop"] = df[s] / (df["W_total_S1S4"] + eps)

    # Ventanas de control (para shapes)
    return df, sow, preR_window, postR_window, gram_window

# ---------- RUN ----------
if run_btn:
    df, sow, preR_window, postR_window, gram_window = simulate(
        nyears, seed_bank0, K, Tb, sim_seed,
        preR_days_before, preR_eff_S1S2, preR_residual,
        postR_days_after, postR_eff_S1S4, postR_residual,
        gram_days_after, gram_eff_S1S3, gram_residual,
        sow,
        LAI_max, t_lag, t_close, LAI_hc, Cs, Ca,
        p_S1, p_S2, p_S3, p_S4
    )
    st.success(f"Listo: {len(df)} días simulados desde {sow}.")

    # ---------- Helper: bandas de control como shapes ----------
    def control_shapes_from_windows(preR_window, postR_window, gram_window):
        # compactar conjuntos de fechas en rangos contiguos
        def to_ranges(days_set):
            if not days_set:
                return []
            days = sorted(list(days_set))
            ranges = []
            start = prev = days[0]
            for d in days[1:]:
                if d == prev + dt.timedelta(days=1):
                    prev = d
                else:
                    ranges.append((start, prev))
                    start = prev = d
            ranges.append((start, prev))
            return ranges

    #   Crear shapes
        shapes = []
        for (x0,x1) in to_ranges(preR_window):
            shapes.append(dict(type="rect", xref="x", yref="paper",
                               x0=pd.to_datetime(x0), x1=pd.to_datetime(x1 + dt.timedelta(days=1)),
                               y0=0, y1=1, fillcolor="rgba(255,0,0,0.10)", line=dict(width=0)))
        for (x0,x1) in to_ranges(postR_window):
            shapes.append(dict(type="rect", xref="x", yref="paper",
                               x0=pd.to_datetime(x0), x1=pd.to_datetime(x1 + dt.timedelta(days=1)),
                               y0=0, y1=1, fillcolor="rgba(0,128,0,0.10)", line=dict(width=0)))
        for (x0,x1) in to_ranges(gram_window):
            shapes.append(dict(type="rect", xref="x", yref="paper",
                               x0=pd.to_datetime(x0), x1=pd.to_datetime(x1 + dt.timedelta(days=1)),
                               y0=0, y1=1, fillcolor="rgba(0,0,255,0.10)", line=dict(width=0)))
        return shapes

    shapes_controls = control_shapes_from_windows(preR_window, postR_window, gram_window)

    # ---------- TABS ----------
    t1, t2, t3, t4 = st.tabs([
        "Densidades S1–S4 + Bandas de control",
        "Supervivencia total y mortalidad",
        "Proporción por estado",
        "Datos / Descargar"
    ])

    # --- Tab 1: Densidades por estado + bandas control
    with t1:
        fig = go.Figure()
        for s in ["W1","W2","W3","W4"]:
            fig.add_trace(go.Scatter(x=df["date"], y=df[s], mode="lines", name=s))
        fig.update_layout(
            title="Densidad por estado (S1–S4) con ventanas de control",
            xaxis_title="Fecha", yaxis_title="pl·m⁻²",
            template="plotly_white",
            shapes=shapes_controls,
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Bandas: 🔴 PreR, 🟢 PostR, 🔵 Gram")

    # --- Tab 2: Supervivencia total (Σ S1–S4) y mortalidad acumulada
    with t2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["date"], y=df["W_total_S1S4"], name="Vivos (Σ S1–S4)", mode="lines"))
        fig2.add_trace(go.Scatter(x=df["date"], y=df["Mort_preR_cum"], name="Mort acumulada PreR", mode="lines"))
        fig2.add_trace(go.Scatter(x=df["date"], y=df["Mort_postR_cum"], name="Mort acumulada PostR", mode="lines"))
        fig2.add_trace(go.Scatter(x=df["date"], y=df["Mort_gram_cum"], name="Mort acumulada Gram", mode="lines"))
        fig2.update_layout(
            title="Supervivencia total y mortalidad acumulada por control",
            xaxis_title="Fecha", yaxis_title="pl·m⁻²",
            template="plotly_white",
            shapes=shapes_controls,
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Mortalidad diaria apilada (opcional)
        fig2b = go.Figure()
        fig2b.add_trace(go.Bar(x=df["date"], y=df["Mort_preR"], name="PreR"))
        fig2b.add_trace(go.Bar(x=df["date"], y=df["Mort_postR"], name="PostR"))
        fig2b.add_trace(go.Bar(x=df["date"], y=df["Mort_gram"], name="Gram"))
        fig2b.update_layout(
            barmode="stack",
            title="Mortalidad diaria por control (apilada)",
            xaxis_title="Fecha", yaxis_title="pl·m⁻²·día⁻¹",
            template="plotly_white",
            shapes=shapes_controls,
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig2b, use_container_width=True)

    # --- Tab 3: Proporción por estado fenológico
    with t3:
        fig3 = go.Figure()
        for s in ["W1_prop","W2_prop","W3_prop","W4_prop"]:
            fig3.add_trace(go.Scatter(x=df["date"], y=df[s], mode="lines", name=s.replace("_prop","")))
        fig3.update_layout(
            title="Estructura fenológica (proporción por estado) en la población viva",
            xaxis_title="Fecha", yaxis_title="Proporción (0–1)",
            template="plotly_white",
            shapes=shapes_controls,
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # --- Tab 4: Datos / Descargar
    with t4:
        st.dataframe(df.tail(100), use_container_width=True)
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Descargar CSV", csv, "weedcrop_controls_visual.csv", "text/csv")

else:
    st.info("Configura parámetros y presioná ▶ Ejecutar simulación.")

