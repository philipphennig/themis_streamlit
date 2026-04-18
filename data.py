import pandas as pd
import streamlit as st

# ISO 3166-1 alpha-3 codes for the 27 EU member states.
# These are always aggregated into the OWID_EU27 group to avoid double-counting.
EU27_CODES = frozenset({
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
})

# ISO codes for the 53 African countries present in the dataset.
# Used when Africa negotiates as a single bloc (OWID_AFR).
AFRICA_CODES = frozenset({
    "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD",
    "COM", "COD", "COG", "CIV", "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH",
    "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "LBY", "MDG",
    "MWI", "MLI", "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA",
    "STP", "SEN", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO", "TUN",
    "UGA", "ZMB", "ZWE",
})


@st.cache_data
def load_country_data(
    africa_joint: bool = False,
    path: str = "themis_data.csv",
) -> pd.DataFrame:
    """
    Load emissions data and return filtered, normalised 2024 rows.

    The EU27 countries are always replaced by the OWID_EU27 aggregate to avoid
    double-counting.  When africa_joint is True, the 53 individual African
    countries are likewise replaced by the OWID_AFR aggregate.
    """
    df = pd.read_csv(path)
    df = df[df["year"] == 2024]
    df = df[~df["code"].isna()]

    # Decide which OWID aggregate groups to keep as single negotiating blocs
    owid_to_include = {"OWID_EU27"}
    if africa_joint:
        owid_to_include.add("OWID_AFR")

    # Drop all other OWID pseudo-codes (continents, income groups, etc.)
    df = df[~df["code"].str.startswith("OWID") | df["code"].isin(owid_to_include)]

    # Drop individual member countries of the included blocs to avoid double-counting
    member_codes = EU27_CODES | (AFRICA_CODES if africa_joint else frozenset())
    df = df[~df["code"].isin(member_codes)]

    # Normalise so shares sum to 1 within this filtered set
    df["share_of_global"] = (
        df["emissions_total_as_share_of_global"]
        / df["emissions_total_as_share_of_global"].sum()
    )
    df.reset_index(inplace=True, drop=True)
    return df
