"""USD high-impact news engine: calendar feed, FRED history, 'actual' predictor."""
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

CALENDAR_URL = os.getenv(
    "CALENDAR_URL", "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
)
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
CONSENSUS_WEIGHT = 0.7  # trust placed in the consensus forecast when blending


@dataclass(frozen=True)
class Indicator:
    series: str   # FRED series id
    kind: str     # level | diff | pct | yoy  (how a series becomes a "print")
    usd_up: bool  # True if a higher-than-expected print is USD-bullish
    scale: float = 1.0


# Keys mirror the event titles used by the calendar feed.
INDICATORS = {
    "Non-Farm Employment Change": Indicator("PAYEMS", "diff", True),
    "Unemployment Rate": Indicator("UNRATE", "level", False),
    "Average Hourly Earnings m/m": Indicator("CES0500000003", "pct", True),
    "CPI m/m": Indicator("CPIAUCSL", "pct", True),
    "Core CPI m/m": Indicator("CPILFESL", "pct", True),
    "CPI y/y": Indicator("CPIAUCNS", "yoy", True),
    "Core PCE Price Index m/m": Indicator("PCEPILFE", "pct", True),
    "PPI m/m": Indicator("PPIFIS", "pct", True),
    "Retail Sales m/m": Indicator("RSAFS", "pct", True),
    "GDP q/q": Indicator("A191RL1Q225SBEA", "level", True),
    "Unemployment Claims": Indicator("ICSA", "level", False, 0.001),
    "Prelim UoM Consumer Sentiment": Indicator("UMCSENT", "level", True),
}
_BY_LOWER = {k.lower(): v for k, v in INDICATORS.items()}


def match_indicator(title):
    t = str(title).lower().strip()
    if t.endswith("gdp q/q"):  # Advance / Prelim / Final GDP q/q
        t = "gdp q/q"
    return _BY_LOWER.get(t)


def parse_num(x):
    """'150K' -> 150.0, '0.3%' -> 0.3, '' -> None (M/B/T are converted to thousands)."""
    text = "" if x is None else str(x)
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*([%KMBT]?)", text.replace(",", ""))
    if not m:
        return None
    return float(m.group(1)) * {"M": 1e3, "B": 1e6, "T": 1e9}.get(m.group(2), 1.0)


def usd_high_impact(url=CALENDAR_URL):
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError("unexpected feed response (possibly rate-limited)")
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = df[(df["country"] == "USD") & (df["impact"] == "High")].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def history(ind, api_key, years=15):
    """Past 'prints' of an indicator, built from FRED observations."""
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    r = requests.get(
        FRED_URL,
        params={"series_id": ind.series, "api_key": api_key,
                "file_type": "json", "observation_start": start},
        timeout=20,
    )
    if not r.ok:  # message deliberately omits the URL so the key never leaks
        raise ValueError(f"FRED returned HTTP {r.status_code} - check your API key")
    s = pd.Series({pd.Timestamp(o["date"]): o["value"] for o in r.json()["observations"]})
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
    if ind.kind == "diff":
        s = s.diff()
    elif ind.kind == "pct":
        s = s.pct_change() * 100
    elif ind.kind == "yoy":
        s = s.pct_change(12) * 100
    return s.dropna() * ind.scale


def _fit_predict(y, lags=3, ridge=1.0):
    """Ridge AR(lags) on z-scored data -> (next value, 1-sigma one-step error)."""
    mu, sig = y.mean(), (y.std() or 1.0)
    z = (y - mu) / sig
    n = len(z) - lags
    X = np.column_stack([np.ones(n)] + [z[i:i + n] for i in range(lags)])
    t = z[lags:]
    pen = ridge * np.eye(lags + 1)
    pen[0, 0] = 0.0  # don't shrink the intercept
    beta = np.linalg.solve(X.T @ X + pen, X.T @ t)
    sd = float((t - X @ beta).std(ddof=lags + 1) * sig)
    nxt = float(np.r_[1.0, z[-lags:]] @ beta * sig + mu)
    return nxt, sd


def backtest(y, n=24):
    """Walk-forward one-step MAE: model vs naive 'repeat the last print'."""
    em, en = [], []
    for i in range(len(y) - n, len(y)):
        pred, _ = _fit_predict(y[:i])
        em.append(abs(pred - y[i]))
        en.append(abs(y[i - 1] - y[i]))
    return float(np.mean(em)), float(np.mean(en))


def analyse(hist, forecast=None, usd_up=True, window=72):
    """Predict the next actual from history, blended with the consensus forecast."""
    y = hist.dropna().to_numpy(dtype=float)[-window:]
    if len(y) < 16:
        raise ValueError("not enough history")
    model, sd = _fit_predict(y)
    predicted = model if forecast is None else (
        CONSENSUS_WEIGHT * forecast + (1 - CONSENSUS_WEIGHT) * model
    )
    out = {"model": model, "predicted": predicted, "sd": sd,
           "vs_forecast": "n/a", "usd_bias": "n/a"}
    if forecast is not None and sd > 0:
        z = (model - forecast) / sd
        if abs(z) < 0.5:
            out.update(vs_forecast="In line", usd_bias="Neutral")
        else:
            above = z > 0
            out.update(vs_forecast="Above" if above else "Below",
                       usd_bias="USD bullish" if above == usd_up else "USD bearish")
    if len(y) >= 48:
        out["mae_model"], out["mae_naive"] = backtest(y)
    return out
  
