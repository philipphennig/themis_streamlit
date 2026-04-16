import streamlit as st

from matplotlib import pyplot as plt
# from tueplots import bundles
import pandas as pd
# from tueplots.constants.color import rgb

import numpy as np
import requests
import ssl
import urllib.request

### Color scheme:
# Uni Tuebingen corporate colors: primary
tue_red = np.array([141.0, 45.0, 57.0]) / 255.0

tue_dark = np.array([55.0, 65.0, 74.0]) / 255.0

tue_gray = np.array([175.0, 179.0, 183.0]) / 255.0

tue_gold = np.array([174.0, 159.0, 109.0]) / 255.0

tue_lightgold = np.array([239.0, 236.0, 226.0]) / 255.0

# Uni Tuebingen corporate colors: secondary

tue_darkblue = np.array([65.0, 90.0, 140.0]) / 255.0

tue_blue = np.array([0.0, 105.0, 170.0]) / 255.0

tue_lightblue = np.array([80.0, 170.0, 200.0]) / 255.0

tue_lightgreen = np.array([130.0, 185.0, 160.0]) / 255.0

tue_green = np.array([125.0, 165.0, 75.0]) / 255.0

tue_darkgreen = np.array([50.0, 110.0, 30.0]) / 255.0

tue_ocre = np.array([200.0, 80.0, 60.0]) / 255.0

tue_violet = np.array([175.0, 110.0, 150.0]) / 255.0

tue_mauve = np.array([180.0, 160.0, 150.0]) / 255.0

tue_lightorange = np.array([215.0, 180.0, 105.0]) / 255.0

tue_orange = np.array([210.0, 150.0, 0.0]) / 255.0

tue_brown = np.array([145.0, 105.0, 70.0]) / 255.0


# plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.dpi": 200})
plt.rcParams.update({"figure.figsize": (6, 3)})

st.set_page_config(
    page_title="The Themis Mechanism",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "(cc-0-4.0) Philipp Hennig, 2026"},
)

st.sidebar.markdown("""# The Themis Mechanism""")
UPPER = 122
LOWER = 5

price_preference = st.sidebar.slider("Your preferred price for CO2 emissions", min_value=LOWER, max_value=UPPER, value=60, step=1)

### LOAD DATA:
df = pd.read_csv("themis_data.csv")
df_current = df[df["year"] == 2024]
# actually, remove everyone with code "NaN" (Those are regions, not countries, e.g. "World", "Africa", "Asia", ...)
df_current = df_current[~df_current["code"].isna()]

countries = df_current["entity"].values
country = st.sidebar.selectbox(
    "Your country",
    countries,
)

df_current["share_of_global"] = (
    df_current["emissions_total_as_share_of_global"]
    / df_current["emissions_total_as_share_of_global"].sum()
)
N = len(df_current)

### Sample price preferences for each country:

sample_model = st.sidebar.selectbox(
    "Sample price preferences for the other countries from a distribution",
    [
        "Independent Uniform",
        "Correlated with Share of Global Emissions",
        "Anti-correlated with Share of Global Emissions",
    ],
)

rng = np.random.default_rng(seed=0)

if sample_model == "Independent Uniform":
    price_preferences = rng.uniform(size=N, low=LOWER, high=UPPER)
elif sample_model == "Correlated with Share of Global Emissions":
    price_preferences = LOWER + (UPPER - LOWER) * df_current["share_of_global"].values
elif sample_model == "Anti-correlated with Share of Global Emissions":
    price_preferences = UPPER - (UPPER - LOWER) * df_current["share_of_global"].values

# set my price:
price_preferences[countries == country] = price_preference

### Compute Themis price:
xplot = np.linspace(0, UPPER + 5, 1000)
shares = df_current["share_of_global"].values
SF = (price_preferences[None, :] < xplot[:, None]) * shares[None, :]
SF = 1 - SF.sum(axis=1)

st.sidebar.markdown(f"""
### Your preferred price: {price_preference} EUR/tCO2e
### Themis price: {xplot[np.argmin(np.abs(xplot - price_preference))]:.1f} EUR/tCO2e
""")

### PLOT:
fig, ax = plt.subplots()
bars = ax.bar(
    price_preferences, shares, width=1.0, color=tue_blue, label="countries"
)
ax.axvline(price_preference, color=tue_blue, lw=0.75, label=f"your price: {price_preference} EUR/tCO2e")

ax2 = ax.twinx()
cov = ax2.plot(xplot, SF, color=tue_blue, label="achieved coverage")
ax3 = ax.twinx()
# make extra third yaxis:
ax3.spines["right"].set_position(("outward", 30))
gsp = ax3.plot(xplot, xplot, color=tue_green, label="global set price")

ax4 = ax.twinx()
ax4.spines["right"].set_position(("outward", 70))
tr = ax4.plot(xplot, xplot * SF, color=tue_red, label="Themis revenue")

ax.set_xlabel(r"preferred price $p$ [EUR/tCO2e]")
ax.set_ylabel("share of global emissions")
ax.yaxis.label.set_color(tue_blue)
ax2.set_ylabel("achieved coverage")
ax2.yaxis.label.set_color(tue_blue)
ax3.set_ylabel("global set price $p$ [EUR/tCO2e]")
ax3.yaxis.label.set_color(tue_green)

ax4.set_ylabel("Themis revenue [EUR/tCO2e]")
ax4.yaxis.label.set_color(tue_red)
ax.xaxis.set_major_locator(plt.MultipleLocator(20))
ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
ax.set_xlim(0, UPPER + 5)
ax2.set_ylim(0, 1)
ax3.set_ylim(0, UPPER + 5)
ax4.set_ylim(0, (xplot * SF).max() * 1.1)
max_location = np.argmax(xplot * SF)
ax.axvline(
    x=xplot[max_location], color=tue_red, linestyle="-", label="Themis price"
)

st.pyplot(fig)
