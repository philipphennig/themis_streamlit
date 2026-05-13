import pandas as pd
import streamlit as st

_CSV = "themis_data.csv"

# ISO 3166-1 alpha-3 codes for the 27 EU member states.
# EDGAR already provides an EU27 aggregate row, so these are dropped to avoid
# double-counting.
EU27_CODES = frozenset({
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
})

# EDGAR codes for African countries (mostly ISO alpha-3).
# Used when Africa negotiates as a single bloc.
AFRICA_CODES = frozenset({
    "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD",
    "COM", "COD", "COG", "CIV", "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH",
    "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "LBY", "MDG",
    "MWI", "MLI", "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA",
    "STP", "SEN", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO", "TUN",
    "UGA", "ZMB", "ZWE",
})


@st.cache_data
def load_country_data(africa_joint: bool = False) -> pd.DataFrame:
    """
    Load 2024 GHG data from the pre-processed CSV and return a filtered,
    normalised DataFrame.

    Columns returned:
        entity                      country / bloc name
        code                        EDGAR country code
        share_of_global             normalised share within the negotiating set
        emissions_total_per_capita  GHG per capita [tCO2e/person]

    The EU27 individual members are always dropped (the EU27 aggregate row is
    kept).  When africa_joint is True the individual African countries are
    replaced by a synthetic Africa aggregate.

    To regenerate themis_data.csv from the EDGAR source file, run:
        python3 prepare_data.py
    """
    df = pd.read_csv(_CSV)

    # Give the EU27 aggregate a readable name and drop its individual members
    df.loc[df["code"] == "EU27", "entity"] = "European Union (27)"
    df = df[~df["code"].isin(EU27_CODES)].copy()

    # Africa: build aggregate bloc or keep individual countries
    if africa_joint:
        africa_mask = df["code"].isin(AFRICA_CODES)
        af = df[africa_mask]
        af_total = af["emissions_total"].sum()
        valid = af["emissions_total_per_capita"].notna() & (af["emissions_total_per_capita"] > 0)
        pop_proxy = (af.loc[valid, "emissions_total"] / af.loc[valid, "emissions_total_per_capita"]).sum()
        af_percap = float(af.loc[valid, "emissions_total"].sum() / pop_proxy) if pop_proxy > 0 else float("nan")
        africa_row = pd.DataFrame([{
            "code": "AFR",
            "entity": "Africa",
            "emissions_total": af_total,
            "emissions_total_per_capita": af_percap,
        }])
        df = pd.concat([df[~africa_mask], africa_row], ignore_index=True)

    # Normalise shares to sum to 1 within the negotiating set
    df["share_of_global"] = df["emissions_total"] / df["emissions_total"].sum()

    df = df.reset_index(drop=True)
    return df
