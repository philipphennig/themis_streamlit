import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from tueplots import bundles
from tueplots.constants.color import rgb

plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.dpi": 200, "figure.figsize": (6, 3)})

# Specs for the three optional right-side overlay axes, in display order.
# Each entry: (key, color, ylabel, ylim_fn, plot_fn)
_RIGHT_AXES = [
    (
        "coverage",
        rgb.tue_blue,
        "lost coverage",
        lambda *_: (0, 1),
        lambda ax, xplot, SF: ax.plot(xplot, SF, color=rgb.tue_blue),
    ),
    (
        "set_price",
        rgb.tue_green,
        "global set price $p$ [EUR/tCO2e]",
        lambda xplot, *_: (0, xplot.max()),
        lambda ax, xplot, *_: ax.plot(xplot, xplot, color=rgb.tue_green, linestyle="--"),
    ),
    (
        "revenue",
        rgb.tue_red,
        "Themis revenue [EUR/tCO2e]",
        lambda xplot, SF: (0, (xplot * SF).max() * 1.1),
        lambda ax, xplot, SF: ax.plot(xplot, xplot * SF, color=rgb.tue_red),
    ),
]


LABELLED_COUNTRIES = {
    "China":                "China",
    "India":                "India",
    "European Union (27)":  "EU27",
    "United Kingdom":       "UK",
    "United States":        "USA",
}


def make_figure(
    price_preferences: np.ndarray,
    shares: np.ndarray,
    shares_pp: np.ndarray,
    user_price: float,
    xplot: np.ndarray,
    SF: np.ndarray,
    themis_price: float,
    countries: np.ndarray | None = None,
    show_coverage: bool = True,
    show_set_price: bool = True,
    show_revenue: bool = True,
) -> Figure:
    """
    Build the two-panel Themis figure.

    Top panel: bar chart of country price preferences (height = share of global
    emissions) overlaid with any combination of the coverage curve, global-set-
    price line, and Themis revenue curve.  Bottom panel: per-capita emissions.
    """
    show = {"coverage": show_coverage, "set_price": show_set_price, "revenue": show_revenue}
    fig, axs = plt.subplots(2, 1, height_ratios=[5, 1], sharex=True)

    # ── Top panel ────────────────────────────────────────────────────────────
    ax = axs[0]
    ax.bar(price_preferences, shares, width=1.0, color=rgb.tue_blue)

    # Labels on selected country bars
    if countries is not None:
        for entity, label in LABELLED_COUNTRIES.items():
            idx = np.where(countries == entity)[0]
            if len(idx):
                ax.text(
                    price_preferences[idx[0]], shares[idx[0]], label,
                    ha="center", va="bottom", fontsize="xx-small", color=rgb.tue_blue,
                )

    # Vline for the user's price with text label
    ax.axvline(user_price, color=rgb.tue_blue, lw=0.75)
    ax.text(
        user_price, 0.99, f"your price: {user_price:.0f}",
        transform=ax.get_xaxis_transform(),
        ha="right", va="top", fontsize="xx-small", color=rgb.tue_blue, rotation=90,
    )

    # Vline for the Themis price with text label
    ax.axvline(themis_price, color=rgb.tue_red, linestyle="-")
    ax.text(
        themis_price, 0.99, f"Themis price: {themis_price:.0f}",
        transform=ax.get_xaxis_transform(),
        ha="right", va="top", fontsize="xx-small", color=rgb.tue_red, rotation=90,
    )
    _style_spine(ax, "left", rgb.tue_blue)
    ax.set_xlabel(r"preferred price $p$ [EUR/tCO2e]")
    ax.set_ylabel("share of global emissions")
    ax.yaxis.label.set_color(rgb.tue_blue)
    ax.set_xlim(0, xplot.max())
    ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.grid(True, which="major", axis="x", color=rgb.tue_dark, zorder=-1)
    ax.grid(True, which="minor", axis="x", color=rgb.tue_gray, zorder=-1)

    # Add right-side overlay axes only for the ones that are enabled.
    # Pack them at 40 pt intervals so there are never gaps in the spine layout.
    outward = 0
    for key, color, ylabel, ylim_fn, plot_fn in _RIGHT_AXES:
        if not show[key]:
            continue
        axi = ax.twinx()
        if outward > 0:
            axi.spines["right"].set_position(("outward", outward))
        _style_spine(axi, "right", color)
        plot_fn(axi, xplot, SF)
        axi.set_ylabel(ylabel)
        axi.yaxis.label.set_color(color)
        axi.set_ylim(*ylim_fn(xplot, SF))
        outward += 40

    # ── Bottom panel: per-capita emissions ───────────────────────────────────
    axs[1].bar(price_preferences, shares_pp, width=1.0, color=rgb.tue_lightgreen)
    axs[1].set_ylabel("emissions\n per capita\n [tCO2e pp]", fontsize="xx-small")
    axs[1].grid(True, which="major", axis="x", color=rgb.tue_dark, zorder=-1)
    axs[1].grid(True, which="minor", axis="x", color=rgb.tue_gray, zorder=-1)

    return fig


def _style_spine(ax, side: str, color) -> None:
    ax.spines[side].set_color(color)
    ax.tick_params(axis="y", colors=color)
