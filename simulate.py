# WeedCropSystem — Starter Kit (Python)
# ------------------------------------------------------------
# Minimal, runnable scaffold to implement the Molinari et al. (2020)
# weed–crop simulator in layers. Starts with Steps 0–4 working; later
# steps are marked TODO and can be filled without breaking what's done.
#
# Usage (quick test):
#   python simulate.py --years 5 --seed 42
# This will generate output CSVs in ./out/ and basic console logs.
#
# Folder layout (virtual, all in this single file for the starter kit):
#   - config.py          → calendars & parameters
#   - meteo.py           → weather loading/generation
#   - bank.py            → seed bank Eq.(1)
#   - emergence.py       → E_t placeholder (to be replaced by ANN)
#   - phenology.py       → TT & stage transitions Eq.(4–6)
#   - intra.py           → intraspecific competition Eq.(7–8)
#   - crop_comp.py       → Ciec(t) Eq.(9) & simple LAI model
#   - control.py         → post/residual controls Eq.(14)  (TODO later)
#   - stress.py          → environmental stress Eq.(15)    (TODO later)
#   - seeds.py           → seed production Eq.(12)         (TODO later)
#   - yieldloss.py       → WC Eq.(11) & yield Eq.(10)      (TODO later)
#   - env_econ.py        → EIQ & economics                  (TODO later)
#   - simulate.py        → main daily loop orchestrator
# ------------------------------------------------------------

# =========================
# config.py
# =========================
from dataclasses import dataclass
from typing import Dict, List, Tuple
import datetime as dt

@dataclass
class CropParams:
    name: str = "wheat"
    Cs: float = 200.0  # pl m-2 (standard)
    Ca: float = 200.0  # pl m-2 (actual)
    LAI_hc: float = 6.0  # highly competitive LAI
    # Simple LAI curve params (parabola in TT): LAI = a*TT - b*TT^2, clamped ≥0
    lai_a: float = 0.008
    lai_b: float = 0.000004
    Tb_crop: float = 0.0  # base temp for crop TT (simplified)

@dataclass
class WeedParams:
    # Phenology & demography
    Tb: float = 0.0
    Th: Tuple[float, float, float, float] = (70.0, 280.0, 400.0, 300.0)
    tf: int = 258  # floral induction (Julian day)
    K: float = 250.0
    iam: float = 1.0
    f: Tuple[float, float, float, float, float] = (0.15, 0.30, 0.60, 1.0, 0.0)
    # Seed bank
    L: int = 3
    Q: Tuple[float, float, float] = (0.7, 0.2, 0.1)
    sm: float = 0.0732
    ld: float = 0.67
    lb: float = 0.2075

@dataclass
class SimConfig:
    start_year: int = 2021
    nyears: int = 5
    sowing_dates: Dict[int, dt.date] = None
    harvest_dates: Dict[int, dt.date] = None
    seed_bank0: float = 4500.0
    random_seed: int = 42

    def ensure_dates(self):
        if self.sowing_dates is None:
            self.sowing_dates = {}
        if self.harvest_dates is None:
            self.harvest_dates = {}
        for y in range(self.start_year, self.start_year + self.nyears):
            if y not in self.sowing_dates:
                self.sowing_dates[y] = dt.date(y, 6, 1)  # June 1st
            if y not in self.harvest_dates:
                self.harvest_dates[y] = dt.date(y, 12, 1)  # Dec 1st (toy)

# Defaults
CROP = CropParams()
WEED = WeedParams()
SIM = SimConfig()
SIM.ensure_dates()

# =========================
# meteo.py
# =========================
import numpy as np
import pandas as pd

def synthetic_meteo(start: dt.date, end: dt.date, seed: int = 42) -> pd.DataFrame:
    """
    Generate simple synthetic weather: daily tmin/tmax, precipitation spikes.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    # Seasonal temperature trend (°C)
    doy = dates.dayofyear.to_numpy()
    tmean = 12 + 8*np.sin(2*np.pi*(doy-170)/365.0) + rng.normal(0, 1.5, n)
    tmin = tmean - (3 + rng.normal(0, 0.8, n))
    tmax = tmean + (6 + rng.normal(0, 0.8, n))
    # Precipitation: random storms
    prec = rng.choice([0, 0, 0, 0, 3, 8, 15], size=n, p=[0.55,0.15,0.10,0.05,0.07,0.05,0.03])
    df = pd.DataFrame({"date": dates, "tmin": tmin, "tmax": tmax, "prec": prec})
    return df

# =========================
# bank.py — Seed bank Eq.(1)
# =========================
class SeedBank:
    def __init__(self, weed: WeedParams, Sq0: float):
        self.weed = weed
        self.Sq = Sq0  # seeds m-2 quiescent at start of first season
        self.history = []  # log per year
        # Keep last L years of seed production
        self.Sp_hist = []

    def end_of_year_update(self, Sp_year: float):
        """Update quiescent seeds entering next year according to Eq.(1)."""
        w = self.weed
        self.Sp_hist.insert(0, Sp_year)
        self.Sp_hist = self.Sp_hist[:w.L]
        # Quiescent seeds from previous years with fractions Q
        Qvec = list(w.Q)[:len(self.Sp_hist)]
        contrib = 0.0
        for s, q in zip(self.Sp_hist, Qvec):
            contrib += (1 - w.ld) * (1 - w.lb) * q * s
        # Annual mortality on existing bank
        Sq_next = self.Sq * (1 - w.sm) + contrib
        self.history.append(dict(Sq_prev=self.Sq, Sp=Sp_year, Sq_next=Sq_next))
        self.Sq = max(Sq_next, 0.0)
        return self.Sq

# =========================
# emergence.py — E_t (placeholder → later ANN)
# =========================
class EmergenceModel:
    def __init__(self, Tb: float = 0.0):
        self.Tb = Tb
        self.TT = 0.0

    def step(self, tmin: float, tmax: float, prec: float) -> float:
        """Very simple placeholder: logistic on TT with rainfall pulses.
        Returns daily fraction emerged from available quiescent seeds.
        Replace with your ANN later.
        """
        Tmean = (tmin + tmax) / 2.0
        self.TT += max(Tmean - self.Tb, 0.0)
        base = 1 / (1 + np.exp(-(self.TT - 300)/40))  # S-shape vs TT
        pulse = 0.002 if prec >= 5 else 0.0          # rain-triggered pulse
        E_t = max(0.0, min(base*0.003 + pulse, 0.02))  # cap daily 2%
        return float(E_t)

# =========================
# phenology.py — TT & transitions Eq.(4–6)
# =========================
class Phenology:
    def __init__(self, weed: WeedParams):
        self.w = weed
        self.TT = 0.0

    def tt_step(self, tmin: float, tmax: float) -> float:
        Tmean = (tmin + tmax) / 2.0
        dt_tt = max(Tmean - self.w.Tb, 0.0)
        self.TT += dt_tt
        return self.TT

    def stage_thresholds(self):
        return self.w.Th  # (Th1..Th4)

# =========================
# intra.py — Intraspecific competition Eq.(7–8)
# =========================
class IntraSpecific:
    def __init__(self, weed: WeedParams):
        self.w = weed

    def weighted_density(self, W: List[float]) -> float:
        # W is [W1..W5]
        f = self.w.f
        return sum(max(W[i],0.0) * f[i] for i in range(5))

    def survival(self, W: List[float]) -> float:
        Wk = self.weighted_density(W)
        if Wk <= 0:
            return 1.0
        if Wk < self.w.K:
            return max(0.0, 1.0 - self.w.iam * (Wk / self.w.K))
        # if exceeding K, stronger penalty
        return max(0.0, 1.0 - self.w.iam)

# =========================
# crop_comp.py — Ciec Eq.(9) + simple LAI model
# =========================
class CropComp:
    def __init__(self, crop: CropParams):
        self.c = crop
        self.TT = 0.0

    def tt_step(self, tmin: float, tmax: float):
        Tmean = (tmin + tmax) / 2.0
        self.TT += max(Tmean - self.c.Tb_crop, 0.0)
        return self.TT

    def LAI(self) -> float:
        # Simple parabola in TT, clamped ≥0
        a, b = self.c.lai_a, self.c.lai_b
        lai = a*self.TT - b*(self.TT**2)
        return float(max(lai, 0.0))

    def Ciec(self, Ca: float = None) -> float:
        if Ca is None:
            Ca = self.c.Ca
        lai = self.LAI()
        val = (lai / max(self.c.LAI_hc, 1e-6)) * (self.c.Cs / max(Ca, 1e-6))
        return float(max(0.0, min(val, 1.0)))

# =========================
# control.py — (TODO later steps)
# =========================
class Control:
    def __init__(self):
        self.events = []  # list of dicts: {date, type, efficacy_by_stage, residual_days}

    def add_post(self, date: dt.date, eff: Tuple[float, float, float, float, float]):
        self.events.append(dict(date=date, kind="post", eff=eff))

    def add_residual(self, date: dt.date, eff01: float, days: int):
        self.events.append(dict(date=date, kind="residual", eff01=eff01, days=days))

    # Placeholders to be wired in Step 6+

# =========================
# stress.py — (TODO later)
# =========================
class Stress:
    def __init__(self):
        self.dates = set()  # dates with adverse conditions
        self.mstage = (0.0, 0.0, 0.0, 0.0, 0.0)

# =========================
# seeds.py — (TODO later) Eq.(12)
# =========================
class SeedProduction:
    def __init__(self):
        self.groups = [(187.2, 12000), (19.3, 2500)]  # (Fc, max) toy defaults

# =========================
# yieldloss.py — (TODO later) Eq.(11) & Eq.(10)
# =========================
class YieldLoss:
    def __init__(self):
        self.a = 10.0
        self.k = 3.7
        self.Myl = 0.60  # 60% max loss (toy for wheat low-comp)

# =========================
# env_econ.py — (TODO later)
# =========================
class EnvEcon:
    def __init__(self):
        self.eiq_log = []
        self.costs = []

# =========================
# simulate.py — main loop for Steps 0–4 (runnable)
# =========================
import argparse
import os

class Simulator:
    def __init__(self, sim: SimConfig, crop: CropParams, weed: WeedParams):
        self.sim = sim
        self.crop = crop
        self.weed = weed
        self.seedbank = SeedBank(weed, sim.seed_bank0)
        self.emerg = EmergenceModel(Tb=weed.Tb)
        self.pheno = Phenology(weed)
        self.intra = IntraSpecific(weed)
        self.ccomp = CropComp(crop)

    def run(self, meteo_df: pd.DataFrame) -> pd.DataFrame:
        """Run daily loop for nyears using provided meteo df."""
        out_rows = []
        for y in range(self.sim.start_year, self.sim.start_year + self.sim.nyears):
            sow, harv = self.sim.sowing_dates[y], self.sim.harvest_dates[y]
            mask = (meteo_df["date"].dt.date >= sow) & (meteo_df["date"].dt.date <= harv)
            season = meteo_df.loc[mask].reset_index(drop=True)

            # Reset seasonal state
            W = [0.0, 0.0, 0.0, 0.0, 0.0]  # densities per stage s=1..5
            self.emerg.TT = 0.0
            self.pheno.TT = 0.0
            self.ccomp.TT = 0.0
            Sp_year = 0.0  # TODO later when SeedProduction is wired

            for i, row in season.iterrows():
                date = row["date"].date()
                tmin, tmax, prec = float(row["tmin"]), float(row["tmax"]), float(row["prec"])

                # Update TT (weed & crop)
                TTw = self.pheno.tt_step(tmin, tmax)
                TTc = self.ccomp.tt_step(tmin, tmax)

                # Step 4: Crop competition (Ciec) for s=1 via Eq.(9)
                Ciec_t = self.ccomp.Ciec(self.crop.Ca)

                # Step 3: Intra-specific survival based on current W
                surv_intra = self.intra.survival(W)

                # Step 5 (placeholder): emergence fraction E_t (replace with ANN later)
                E_t = self.emerg.step(tmin, tmax, prec)

                # Eq.(3): incoming to s=1 (no residual & pre-seedling control yet)
                I1_t = self.seedbank.Sq * E_t * surv_intra * (1 - Ciec_t) * (1 - 0.0)

                # Stage transitions by simple counters vs thresholds (toy but consistent)
                Th1, Th2, Th3, Th4 = self.weed.Th
                # For the starter kit, approximate time since entry by using TTw
                # (A proper cohort-tracking queue can be added later.)

                # Simple flow logic: move fractions forward if TTw exceeds thresholds
                O1 = I1_t if TTw >= Th1 else 0.0
                O2 = W[1] if TTw >= (Th1 + Th2) else 0.0
                O3 = W[2] if (TTw >= (Th1 + Th2 + Th3) and date.timetuple().tm_yday >= self.weed.tf) else 0.0
                O4 = W[3] if TTw >= (Th1 + Th2 + Th3 + Th4) else 0.0

                # Update stages (very simplified mass balance for the starter)
                W1_next = max(0.0, W[0] + I1_t - O1)
                W2_next = max(0.0, W[1] + O1 - O2)
                W3_next = max(0.0, W[2] + O2 - O3)
                W4_next = max(0.0, W[3] + O3 - O4)
                W5_next = max(0.0, W[4] + O4)  # sink
                W = [W1_next, W2_next, W3_next, W4_next, W5_next]

                out_rows.append({
                    "date": date,
                    "year": y,
                    "TTw": TTw,
                    "TTc": TTc,
                    "LAI": self.ccomp.LAI(),
                    "Ciec": Ciec_t,
                    "E_t": E_t,
                    "I1_t": I1_t,
                    "W1": W[0], "W2": W[1], "W3": W[2], "W4": W[3], "W5": W[4],
                    "Sq": self.seedbank.Sq
                })

            # End of season: update seed bank (Sp_year is 0.0 for now)
            self.seedbank.end_of_year_update(Sp_year)

        return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=SIM.nyears)
    parser.add_argument("--seed", type=int, default=SIM.random_seed)
    args = parser.parse_args()

    # Build meteo covering all years
    start = SIM.sowing_dates[SIM.start_year]
    end = SIM.harvest_dates[SIM.start_year + SIM.nyears - 1]
    meteo_df = synthetic_meteo(start, end, seed=args.seed)

    sim = Simulator(SIM, CROP, WEED)
    df = sim.run(meteo_df)

    os.makedirs("out", exist_ok=True)
    out_csv = os.path.join("out", "daily_states.csv")
    df.to_csv(out_csv, index=False)
    # Simple yearly summary
    summary = df.groupby("year")["W5"].max().rename("max_W5").to_frame()
    summary.to_csv(os.path.join("out", "summary.csv"))

    print(f"Saved {out_csv} and out/summary.csv")
    print("Head:\n", df.head())

if __name__ == "__main__":
    main()

