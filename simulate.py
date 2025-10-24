# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 WeedCropSystem — v3.13.1 (AUC de competencia + Optimización)
# ---------------------------------------------------------------
# - Pérdida de rinde basada en AUC (competencia acumulada)
# - Gateo jerárquico: preR → preemR → postR → gram
# - Reglas agronómicas:
#     · preR: −30 a −14 días antes de siembra
#     · preemR: 0 a +10 días desde siembra
#     · postR: +20 a +180 días (2 hojas en adelante)
#     · gram: +25 a +35 días (3–4 hojas)
# ===============================================================

import sys, datetime as dt
import numpy as np
import pandas as pd

# ----------------------------
# 🌦️ Generador meteorológico
# ----------------------------
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

# ------------------------------------------------------------
# 🌱 Simulador diario (con jerarquía de controles y AUC)
# ------------------------------------------------------------
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

    # 🧭 Ventanas fenológicas reales
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
        dss = (date - sow).days
        Tmean = (row["tmin"] + row["tmax"]) / 2
        TTw += max(Tmean - Tb, 0)
        Ciec_t, LAI_t = ciec_calendar(dss, LAI_max, t_lag, t_close, LAI_hc, Cs, Ca)
        E_t = 8.0 * emergence_simple(TTw, row["prec"])
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

    # 💡 NUEVO: AUC de competencia normalizado
    df["WC_acum"] = df["WC"].cumsum() / len(df)
    df["Yield_loss_%"] = (alpha*df["WC_acum"]) / (1 + (alpha*df["WC_acum"]/Lmax))
    df["Yield_relative_%"] = 100 - df["Yield_loss_%"]
    df["Yield_abs_kg_ha"] = GY_pot * (df["Yield_relative_%"]/100)
    return df

    








