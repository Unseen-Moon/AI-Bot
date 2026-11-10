import os

import pandas as pd
import streamlit as st

import engine as eg

st.set_page_config(page_title="USD News Predictor", page_icon="📈", layout="wide")
st.title("📈 USD High-Impact News Predictor")
st.caption("Statistical estimates from FRED history + consensus. Not financial advice.")

key = st.sidebar.text_input("FRED API key", os.getenv("FRED_API_KEY", ""), type="password")
source = st.sidebar.radio("Calendar source", ["Live feed", "Manual entry"])


@st.cache_data(ttl=1800, show_spinner="Loading calendar...")
def load_calendar():
    return eg.usd_high_impact()


@st.cache_data(ttl=86400, show_spinner="Loading FRED history...")
def load_history(title, api_key):
    return eg.history(eg.match_indicator(title), api_key)


if source == "Manual entry":
    title = st.sidebar.selectbox("Event", list(eg.INDICATORS))
    fc_in = st.sidebar.number_input("Forecast", value=0.0, format="%.2f")
    pv_in = st.sidebar.number_input("Previous", value=0.0, format="%.2f")
    events = pd.DataFrame([{"title": title, "date": pd.Timestamp.now(tz="UTC"),
                            "forecast": str(fc_in), "previous": str(pv_in)}])
else:
    try:
        events = load_calendar()
    except Exception as e:  # network error, rate limit, feed format change
        st.error(f"Could not load the calendar feed: {e}. Use Manual entry or set CALENDAR_URL.")
        st.stop()
    if events.empty:
        st.info("No high-impact USD events in the feed this week.")
        st.stop()

rows, hists, results = [], {}, {}
for _, e in events.iterrows():
    ind = eg.match_indicator(e["title"])
    fc, pv = eg.parse_num(e.get("forecast")), eg.parse_num(e.get("previous"))
    res = {}
    if ind and key:
        try:
            hists[e["title"]] = load_history(e["title"], key)
            res = eg.analyse(hists[e["title"]], fc, ind.usd_up)
            results[e["title"]] = res
        except Exception as ex:
            st.warning(f"{e['title']}: {ex}")
    rows.append({
        "Time (UTC)": e["date"].strftime("%a %d %b %H:%M"),
        "Event": e["title"],
        "Previous": pv,
        "Forecast": fc,
        "Model": res.get("model"),
        "Predicted actual": res.get("predicted", fc if fc is not None else pv),
        "±1σ": res.get("sd"),
        "Model vs forecast": res.get("vs_forecast", "n/a"),
        "USD bias": res.get("usd_bias", "n/a"),
    })

st.dataframe(pd.DataFrame(rows).round(2), hide_index=True)
if not key:
    st.info("Add a free FRED API key in the sidebar to unlock model predictions.")

if results:
    pick = st.selectbox("Inspect an event", list(results))
    st.line_chart(hists[pick].tail(48))
    r = results[pick]
    if "mae_model" in r:
        st.metric(
            "Walk-forward error, last 24 prints (model)",
            f"{r['mae_model']:.3f}",
            delta=f"{r['mae_model'] - r['mae_naive']:+.3f} vs naive baseline",
            delta_color="inverse",
        )
