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

# LOWER/UPPER are read from session state so that all widgets below (especially
# the price_preference slider) see the current range even though the range slider
# itself is placed at the visual bottom of the sidebar.
LOWER, UPPER = st.session_state.price_range

# ── Main-frame layout placeholders (in desired visual order) ──────────────────
# Containers are rendered where they are created; their content can be filled
# later, which lets us define the overlay checkboxes before the figure is built
# while they appear visually below it.

summary_container = st.container()
plot_container    = st.container()
overlay_container = st.container()

with overlay_container:
    st.markdown("**Show overlays**")
    ov_col1, ov_col2, ov_col3 = st.columns(3)
    show_coverage  = ov_col1.checkbox("Achieved coverage", value=True)
    show_set_price = ov_col2.checkbox("Global set price",  value=True)
    show_revenue   = ov_col3.checkbox("Themis revenue",    value=True)

# ── Sidebar controls ──────────────────────────────────────────────────────────

st.sidebar.markdown(
    """
                    # The Themis Mechanism: price elicitation
                    For more information, see [Carl Edward Rasmussen's website](https://mlg.eng.cam.ac.uk/carl/climate/)."""
)

if st.sidebar.button("Resample countries' preferences"):
    st.session_state.rng_seed = int(np.random.SeedSequence().entropy) % (2**31)

price_preference = st.sidebar.slider(
    "Your preferred price for CO2 emissions",
    min_value=LOWER, max_value=UPPER,
    value=min(max(60, LOWER), UPPER),  # clamp default to current range
    step=1,
)

africa_joint = st.sidebar.checkbox("Africa negotiates jointly")

# ── Data ──────────────────────────────────────────────────────────────────────

df = load_country_data(africa_joint=africa_joint)
countries = df["entity"].values
shares    = df["share_of_global"].values
shares_pp = df["emissions_total_per_capita"].values

default_country_idx = next((i for i, c in enumerate(countries) if c == "United Kingdom"), 0)
country = st.sidebar.selectbox("Your country", countries, index=default_country_idx)

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

# Override with the user's own stated preference
price_preferences[countries == country] = price_preference

# ── Themis computation ────────────────────────────────────────────────────────

xplot, SF, themis_price = compute_themis_price(price_preferences, shares, UPPER)

# ── Fill main-frame containers ────────────────────────────────────────────────

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
