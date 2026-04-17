import streamlit as st

from matplotlib import pyplot as plt
from tueplots import bundles
import pandas as pd

# from tueplots.constants.color import rgb

plt.rcParams.update(bundles.beamer_moml())

import numpy as np
import requests
import ssl
import urllib.request

if "rng_seed" not in st.session_state:
    st.session_state.rng_seed = 0

rng = np.random.default_rng(seed=st.session_state.rng_seed)

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
if st.sidebar.button("Resample countries' preferences"):
    st.session_state.rng_seed = int(np.random.SeedSequence().entropy) % (2**31)
UPPER = 122
LOWER = 5


price_preference = st.sidebar.slider(
    "Your preferred price for CO2 emissions",
    min_value=LOWER,
    max_value=UPPER,
    value=60,
    step=1,
)

### LOAD DATA:
df = pd.read_csv("themis_data.csv")
df_current = df[df["year"] == 2024]
# remove "World":
df_current = df_current[df_current["entity"] != "World"]
# remove everyone with code "NaN" (Those are regions, not countries, e.g. "World", "Africa", "Asia", ...)
df_current = df_current[~df_current["code"].isna()]
# remove everyone whose code starts with "OWID" (Those are also regions, not countries, e.g. "European Union", "High-income countries", ...):
df_current = df_current[~df_current["code"].str.startswith("OWID")]

countries = df_current["entity"].values
country = st.sidebar.selectbox(
    "Your country",
    countries,
)

# df_current["share_of_global"] = (
#     df_current["emissions_total_per_capita"]
#     / df_current["emissions_total_per_capita"].sum()
# )
df_current["share_of_global"] = (
    df_current["emissions_total_as_share_of_global"]
    / df_current["emissions_total_as_share_of_global"].sum()
)
df_current.reset_index(inplace=True, drop=True)

N = len(df_current)
shares = df_current["share_of_global"].values
shares_pp = df_current["emissions_total_per_capita"].values


### some preparation for an overcomplicated diffusion sampling model:
def Build_Brownian_Bridge(N):
    # the covariance of a Brownian bridge on [0, 1] is:
    k = lambda x, y: np.minimum(x, y) - x * y
    # grid of N points between 0 and 1:
    x = np.linspace(0, 1, N)
    # compute the covariance matrix:
    K = k(x[:, None], x[None, :])
    # and its eigendecomposition:
    EWs, EVs = np.linalg.eigh(K)

    # with this we can make a function that samples from the Brownian bridge:
    def sample(S):
        # sample from a standard normal distribution:
        z = rng.normal(0, 1, (len(x), S))
        # and transform it using the eigendecomposition:
        return EVs @ (np.sqrt(EWs)[:, None] * z)

    return x, K, sample

x, K, sample = Build_Brownian_Bridge(11)
bridge = sample(N)

### Sample price preferences for each country:
sample_model = st.sidebar.selectbox(
    "Sample price preferences for the other countries from a distribution",
    [
        "Independent Uniform",
        "Polluters are generous",
        "Polluters are stingy",
    ],
)
prices_start = rng.uniform(size=N, low=LOWER, high=UPPER)

if sample_model == "Independent Uniform":
    price_preferences = prices_start
elif sample_model == "Polluters are generous":
    contribution_order = np.argsort(shares_pp) 
    prices_end = np.zeros_like(prices_start)
    # at the end, the prices are sorted according to the contribution order:
    prices_end[contribution_order] = np.sort(prices_start)

    a = st.sidebar.slider(
        "randomness control parameter",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
    ) # this has 11 values

    prices = (
        prices_start[None, :]
        + 5 * bridge
        + (prices_end - prices_start)[None, :]
        * np.linspace(0, 1, bridge.shape[0])[:, None]
    )
    price_preferences = prices[int(a*10.0), :].copy().ravel()
elif sample_model == "Polluters are stingy":
    contribution_order = np.argsort(shares_pp, axis=0)[::-1] 
    prices_end = np.zeros_like(prices_start)
    # at the end, the prices are sorted according to the contribution order:
    prices_end[contribution_order] = np.sort(prices_start)

    a = st.sidebar.slider(
        "randomness control parameter",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
    )  # this has 11 values

    prices = (
        prices_start[None, :]
        + 5 * bridge
        + (prices_end - prices_start)[None, :]
        * np.linspace(0, 1, bridge.shape[0])[:, None]
    )
    price_preferences = prices[int(a * 10.0), :].copy().ravel()

# set my price:
price_preferences[countries == country] = price_preference

### Compute Themis price:
xplot = np.linspace(0, UPPER + 5, 1000)
SF = (price_preferences[None, :] < xplot[:, None]) * shares[None, :]
SF = 1 - SF.sum(axis=1)

### PLOT:
fig, axs = plt.subplots(2,1, height_ratios=[2, 1], sharex=True)
ax = axs[0]
bars = ax.bar(price_preferences, shares, width=1.0, color=tue_blue, label="countries")
ax.axvline(
    price_preference,
    color=tue_blue,
    lw=0.75,
    label=f"your price: {price_preference} EUR/tCO2e",
)

ax.spines["left"].set_color(tue_blue)
# and the ticks:
ax.tick_params(axis="y", colors=tue_blue)


ax2 = ax.twinx()
cov = ax2.plot(xplot, SF, color=tue_blue, label="achieved coverage")
# also set the color of the spine:
ax2.spines["right"].set_color(tue_blue)
# and the ticks:
ax2.tick_params(axis="y", colors=tue_blue)

ax3 = ax.twinx()
# make extra third yaxis:
ax3.spines["right"].set_position(("outward", 40))
# also set the color of the spine:
ax3.spines["right"].set_color(tue_green)
# and the ticks:
ax3.tick_params(axis="y", colors=tue_green)
gsp = ax3.plot(xplot, xplot, color=tue_green, label="global set price")

ax4 = ax.twinx()
ax4.spines["right"].set_position(("outward", 90))
# also set the color of the spine:
ax4.spines["right"].set_color(tue_red)
# and the ticks:
ax4.tick_params(axis="y", colors=tue_red)
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
ax.axvline(x=xplot[max_location], color=tue_red, linestyle="-", label="Themis price")

st.sidebar.markdown(
    f"""
### Your preferred price: {price_preference} EUR/tCO2e
### Themis price: {xplot[max_location]:.1f} EUR/tCO2e
"""
)

ax = axs[1]
ax.bar(price_preferences, shares_pp, width=1.0, color=tue_lightgreen, label="countries")
ax.set_ylabel("emissions per capita [tCO2e pp]", fontsize='xx-small')

st.pyplot(fig)
