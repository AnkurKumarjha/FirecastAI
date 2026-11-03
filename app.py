"""Forest Fire Burned-Area Estimator. Run with: streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.config import DATA_PATH, FEATURE_SETS, MONTHS, REPORT_DIR, model_path
from src.predict import load_artifacts, predict_area, severity_band, version_warning

st.set_page_config(page_title="Forest Fire Burned-Area Estimator", layout="wide")

# Physical limits for the input boxes (wider than the training data on purpose:
# values outside the training range trigger a warning instead of being blocked).
HARD_LIMITS = {"temp": (-20.0, 50.0), "RH": (0.0, 100.0), "wind": (0.0, 60.0),
               "rain": (0.0, 100.0), "FFMC": (0.0, 101.0), "DMC": (0.0, 500.0),
               "DC": (0.0, 1500.0), "ISI": (0.0, 100.0)}
LABELS = {"temp": "Temperature (°C)", "RH": "Relative humidity (%)",
          "wind": "Wind speed (km/h)", "rain": "Rain (mm/m²)",
          "FFMC": "FFMC (fine fuel moisture code)", "DMC": "DMC (duff moisture code)",
          "DC": "DC (drought code)", "ISI": "ISI (initial spread index)"}
HELP = {"FFMC": "Canadian Fire Weather Index component. Higher = drier fine fuels. Leave the default if unknown.",
        "DMC": "Moisture of the loose organic layer. Leave the default if unknown.",
        "DC": "Long-term dryness of deep layers. Leave the default if unknown.",
        "ISI": "Expected fire spread rate. Leave the default if unknown."}


@st.cache_resource
def get_artifacts(feature_set):
    return load_artifacts(feature_set)


@st.cache_data
def get_data():
    return pd.read_csv(DATA_PATH).drop_duplicates()


available = [fs for fs in FEATURE_SETS if model_path(fs).exists()]

st.title("Forest Fire Burned-Area Estimator")
st.caption("Estimates how many hectares a fire might burn **if one occurs** under the entered conditions. "
           "It does not estimate the probability that a fire starts.")

if not available:
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

tab_est, tab_dash, tab_about = st.tabs(["Estimate", "Dashboard", "About and limitations"])

with tab_est:
    choice = st.radio("Model", available, horizontal=True,
                      format_func=lambda k: {"full": "Weather + fire-weather indices",
                                             "weather": "Weather only"}[k])
    model, meta = get_artifacts(choice)
    if (w := version_warning(meta)):
        st.warning(w)

    cols = st.columns(3)
    inputs = {"month": cols[0].selectbox("Month", MONTHS, index=MONTHS.index("aug"))}
    for i, feat in enumerate(f for f in meta["features"] if f != "month"):
        lo, hi = HARD_LIMITS[feat]
        inputs[feat] = cols[(i + 1) % 3].number_input(
            LABELS[feat], min_value=lo, max_value=hi,
            value=float(min(max(meta["defaults"][feat], lo), hi)),
            help=HELP.get(feat, f"Training range: {meta['ranges'][feat][0]:g} to {meta['ranges'][feat][1]:g}"))

    if st.button("Estimate burned area", type="primary"):
        try:
            area, warnings = predict_area(model, meta, inputs)
        except ValueError as err:
            st.error(str(err))
        else:
            for w in warnings:
                st.warning(w)
            band = severity_band(area, meta)
            c1, c2 = st.columns(2)
            c1.metric("Estimated burned area", f"{area:.2f} ha")
            c2.metric("Relative severity band", band)
            t = meta["band_thresholds_ha"]
            st.caption(f"Bands are relative to historical fires in this dataset: Low < {t['p50']:.1f} ha, "
                       f"Medium < {t['p90']:.1f} ha, High at or above that.")
            if not meta["beats_median_baseline"]:
                st.info("On this data the model did not clearly beat a simple median baseline in "
                        "cross-validation, so treat this as a rough indication only.")

with tab_dash:
    if not DATA_PATH.exists():
        st.info("Place forestfires.csv in the data/ folder to see charts.")
    else:
        df = get_data()
        df["month"] = pd.Categorical(df["month"], categories=MONTHS, ordered=True)
        by_month = df.groupby("month", observed=False)["area"].agg(["count", "mean", "median", "sum"])
        a, b = st.columns(2)
        a.subheader("Recorded fires by month (count)")
        a.bar_chart(by_month["count"])
        b.subheader("Burned area by month (ha)")
        b.bar_chart(by_month[["mean", "median"]])
        st.caption("Mean is pulled up by a few very large fires; compare it with the median "
                   "and the counts (some months have very few records).")

        st.subheader("Average weather by severity band of recorded fires")
        t = meta["band_thresholds_ha"]
        df["outcome"] = pd.cut(df["area"], [-1, t["p50"], t["p90"], float("inf")],
                               labels=["Low", "Medium", "High"])
        st.dataframe(df.groupby("outcome", observed=False)[["temp", "RH", "wind", "rain"]].mean().round(2))

        imp_file = REPORT_DIR / f"importance_{choice}.csv"
        if imp_file.exists():
            st.subheader("Which inputs the model relies on (permutation importance)")
            st.bar_chart(pd.read_csv(imp_file).set_index("feature")["importance_mean"])
            st.caption("Predictive usefulness on held-out data, not proof of cause.")

with tab_about:
    _, meta = get_artifacts(available[0])
    st.markdown(f"""
**Data.** UCI Forest Fires dataset: 517 fires in Montesinho Natural Park, Portugal (2000 to 2003).

**Model.** `{meta['model_name']}`, trained on a log scale and converted back to hectares.

**Limitations**
- One region and four years; may not transfer to other climates or vegetation.
- Weather explains only a small part of burned area here; expect large errors.
- The estimate is *conditional on a fire occurring*. It is not an ignition probability.
- Associations in the data are not causes.
- Not an official warning. Follow forestry and emergency authorities.
""")
    st.json(meta["test_metrics"])
