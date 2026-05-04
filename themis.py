import numpy as np
import streamlit as st

from data import load_country_data
from model import SAMPLE_MODELS, compute_themis_price, precompute_random_draws, sample_price_preferences
from plot import make_figure

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="The Themis Mechanism: price elicitation",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "originally built by Philipp Hennig, 2026 in support of Carl Edward Rasmussen. Public domain."},
)

# ── Session state ─────────────────────────────────────────────────────────────

if "rng_seed" not in st.session_state:
    st.session_state.rng_seed = 0
if "price_range" not in st.session_state:
    st.session_state.price_range = (5, 122)
if "country_preferences" not in st.session_state:
    st.session_state.country_preferences = {}
if "selected_country" not in st.session_state:
    st.session_state.selected_country = "United Kingdom"
 
# LOWER/UPPER are read from session state so that all widgets below (especially
# the price_preference slider) see the current range even though the range slider
# itself is placed at the visual bottom of the sidebar.
LOWER, UPPER = st.session_state.price_range

# ── Main-frame layout placeholders (in desired visual order) ──────────────────
# Containers are rendered where they are created; their content can be filled
# later, which lets us define the overlay checkboxes before the figure is built
# while they appear visually below it.

explanations      = st.expander("How to use this app", expanded=False)
summary_container = st.container()
plot_container    = st.container()
overlay_container = st.container()

with overlay_container:
    st.markdown("**Show overlays**")
    ov_col1, ov_col2, ov_col3 = st.columns(3)
    show_coverage  = ov_col1.checkbox("Achieved coverage", value=True)
    show_set_price = ov_col2.checkbox("Global set price",  value=False)
    show_revenue   = ov_col3.checkbox("Themis revenue",    value=True)

# ── Sidebar controls ──────────────────────────────────────────────────────────

st.sidebar.markdown(
    """
                    # The Themis Mechanism: price elicitation
                    For more information, see [Carl Edward Rasmussen's website](https://mlg.eng.cam.ac.uk/carl/climate/)."""
)

if st.sidebar.button("Resample countries' preferences"):
    st.session_state.rng_seed = int(np.random.SeedSequence().entropy) % (2**31)

africa_joint = st.sidebar.checkbox("Africa negotiates jointly")

# ── Data ──────────────────────────────────────────────────────────────────────

df = load_country_data(africa_joint=africa_joint)
countries = df["entity"].values
shares    = df["share_of_global"].values
shares_pp = df["emissions_total_per_capita"].values

def country_label(c: str) -> str:
    if c in st.session_state.country_preferences:
        return f"{c} 🔒"
    return c

country = st.sidebar.selectbox(
    "Your country", countries,
    index=list(countries).index(st.session_state.selected_country), format_func=country_label)
st.session_state.selected_country = country

# on_change fires only when the user actively moves the slider (not on country switch),
# so we store only explicitly set preferences, leaving other countries' random values intact.
def _save_pref():
    c = st.session_state._pref_country
    st.session_state.country_preferences[c] = st.session_state[f"price_pref_{c}"]

st.session_state._pref_country = country

price_preference = st.sidebar.slider(
    "Your preferred price for CO2 emissions",
    min_value=LOWER, max_value=UPPER,
    value=min(max(60, LOWER), UPPER),  # clamp default to current range
    step=1,
    key=f"price_pref_{country}",
    on_change=_save_pref,
)

# ── Sampling model ────────────────────────────────────────────────────────────

sample_model = st.sidebar.selectbox(
    "Sample price preferences for the other countries from a distribution",
    SAMPLE_MODELS,
)

alpha = 0.5  # only used by diffusion models; overridden by slider below
if sample_model != "Independent Uniform":
    alpha = st.sidebar.slider(
        "randomness control parameter",
        min_value=0.0, max_value=1.0, value=0.5, step=0.1,
    )

# ── Price range (placed last so it appears at the bottom of the sidebar) ──────

st.sidebar.markdown("---")
price_range = st.sidebar.slider(
    "Price range [EUR/tCO2e]",
    min_value=0, max_value=200,
    value=(LOWER, UPPER),
    step=1,
)
if price_range != (LOWER, UPPER):
    # Persist the new range and rerun immediately so all widgets above
    # (especially the price_preference slider) reflect the updated bounds.
    st.session_state.price_range = price_range
    st.rerun()

# ── Random draws (cached in session state) ────────────────────────────────────
# prices_base and bridge depend only on the seed, country count, and price range.
# They are pre-computed once and reused across reruns so that moving the alpha
# slider (which only re-indexes the bridge) does not trigger a full resample.

draws_key = (st.session_state.rng_seed, len(countries), LOWER, UPPER)
if st.session_state.get("draws_key") != draws_key:
    rng = np.random.default_rng(seed=st.session_state.rng_seed)
    prices_base, bridge = precompute_random_draws(rng, len(countries), LOWER, UPPER)
    st.session_state.draws_key   = draws_key
    st.session_state.prices_base = prices_base
    st.session_state.bridge      = bridge

price_preferences = sample_price_preferences(
    prices_base=st.session_state.prices_base,
    bridge=st.session_state.bridge,
    shares_pp=shares_pp,
    sample_model=sample_model,
    alpha=alpha,
)

# Apply explicitly-set preferences for countries the user has previously adjusted
for c, pref in st.session_state.country_preferences.items():
    mask = countries == c
    if mask.any():
        price_preferences[mask] = pref
# Always apply the live slider value for the currently selected country
price_preferences[countries == country] = price_preference

# ── Themis computation ────────────────────────────────────────────────────────

xplot, SF, themis_price = compute_themis_price(price_preferences, shares, UPPER)

# ── Fill main-frame containers ────────────────────────────────────────────────

with explanations:
    st.markdown(
        """
        The [Themis Mechanism](https://mlg.eng.cam.ac.uk/carl/climate) by [Carl Edward Rasmussen](https://mlg.eng.cam.ac.uk/carl/) is a proposal for a global cooperative framework for addressing climate change. It relies on reciprocity and common commitments. Part of the proposal is a price elicitation process, which   determines a global CO₂e price from a set of heterogeneous national preferences. It finds the global price that maximizes the global CO₂e revenue (a proxy for the climate impact) when every country only gets to choose whether to participate (i.e., pay the set price on their emissions) or not. No nation is ever asked to do anything inconsistent with their expressed preferences.

        This app simulates the price elicitation process of the Themis Mechanism. Use the sidebar controls to explore how the mechanism works under different assumptions about the distribution of countries' price preferences and the resulting price and coverage.

        - **Pick your country**: Select the country you want to represent in the mechanism. This will determine which price preference you control directly.
        - **Your preferred price**: Set your own preferred CO₂e price using the slider. This represents your country's stated preference in the mechanism. Sequentially choose prices for other nations to build a global picture.
        - **Sampled preferences**: Since this is a simulation, the other countries' preferences are sampled from a probability distribution. You can choose the between 
            + an independent uniform distribution (every country's preference is a random choice in the specified price range, without any correlation between countries),
            + the other two models introduce correlations between countries' preferences, with the slider controlling the strength of these correlations. If you set the randomness control parameter to 0, you get back the independent uniform model. I you set it to 1, the preferences are perfectly ordered by per-capita emissions, either from high to low or from low to high depending on the model. Choices in between give you varying degrees of correlation.
        - **Themis price**: The global CO₂e price determined by the Themis Mechanism based on all countries' preferences and their shares of global emissions.
        - **Overlays**: Use the checkboxes below to show or hide additional information on the plot:
            + *achieved coverage* is the fraction of global emissions covered by the set price, under the assumption that each country with a price preference above the set price chooses to participate and pay the set price on their emissions. 
            + *global set price* is the global CO₂ price to be determined by the mechanism. It can be optionally overlaid as a dashed line for reference, because the Themis price is the price that maximizes the *product* of the set price and the coverage
            * *Themis revenue* is the product of the global set price and the achieved coverage, which is what the mechanism maximizes. 
        - **Emissions per capita**: The bottom plot shows the countries' emissions per capita for reference. These numbers are used to determine the order of countries' preferences in the correlated sampling models (because they correlate with development status and thus likely with willingness to pay), but they do not directly affect the Themis price or coverage.

        Experiment with different settings and see how they affect the outcome. For example, you may find that countries that control a large share of global emissions, like China or the US, can affect the price to some degree, but only if they are willing to pay that price themselves. Smaller countries have less influence on the price, but they can still decide whether to participate or not based on the set price and their own preferences. Most importantly, there is no "free-rider problem" in Themis: if a country prefers a higher price, it can only achieve that by being willing to pay it itself, which means it cannot benefit from others' willingness to pay without contributing itself.
        """
    )

with summary_container:
    s_col1, s_col2 = st.columns(2)
    s_col1.markdown(f"### Your preferred price: {price_preference} EUR/tCO₂e")
    s_col2.markdown(f"### Themis price: {themis_price:.1f} EUR/tCO₂e")

with plot_container:
    fig = make_figure(
        price_preferences, shares, shares_pp,
        price_preference, xplot, SF, themis_price,
        countries=countries,
        show_coverage=show_coverage,
        show_set_price=show_set_price,
        show_revenue=show_revenue,
    )
    st.pyplot(fig)
