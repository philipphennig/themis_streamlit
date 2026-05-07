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
        "achieved coverage",
        lambda *_: (0, 1.1),
        lambda ax, xplot, SF: ax.plot(np.vstack((xplot[:-1], xplot[1:], xplot[1:])), np.vstack((SF[:-1], SF[:-1], SF[1:])), color=rgb.tue_blue),
    ),
    (
        "set_price",
        rgb.tue_green,
        "global set price $p$ [EUR/tCO2e]",
        lambda xplot, *_: (0, xplot.max() * 1.1),
        lambda ax, xplot, *_: ax.plot(xplot, xplot, color=rgb.tue_green, linestyle="--"),
    ),
    (
        "revenue",
        rgb.tue_red,
        "Themis revenue [EUR/tCO2e]",
        lambda xplot, SF: (0, (xplot * SF).max() * 1.1),
        lambda ax, xplot, SF: ax.plot(np.vstack((xplot[:-1], xplot[1:], xplot[1:])), np.vstack((xplot[:-1]*SF[:-1], xplot[1:]*SF[:-1], xplot[1:]*SF[1:])), color=rgb.tue_red),
    ),
]


LABELLED_COUNTRIES = {
    "China":                         "China",
    "India":                         "India",
    "European Union (27)":           "EU27",
    "United Kingdom":                "UK",
    "United States":                 "USA",
    "Democratic Republic of Congo":  "DRC",
    "Nigeria":                       "Nigeria",
    "Pakistan":                      "Pakistan",
    "Colombia":                      "Colombia",
    "Argentina":                     "Argentina",
    "South Africa":                  "S. Africa",
    "South Korea":                   "S. Korea",
    "New Zealand":                   "NZ",
    "United Arab Emirates":          "UAE",
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
    ax.bar(price_preferences, shares, width=1.0, color=rgb.tue_orange)

    # Labels at top edge, vertically, at each country's preferred-price position
    if countries is not None:
        for entity, label in LABELLED_COUNTRIES.items():
            idx = np.where(countries == entity)[0]
            if len(idx):
                ax.text(
                    price_preferences[idx[0]], 0.99, label,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize="xx-small", color=rgb.tue_blue, rotation=90,
                )

    # Vline for the user's price with text label
    ax.axvline(user_price, color=rgb.tue_blue, lw=0.75)
    ax.text(
        user_price, 0.85, f"your price: {user_price:.0f}",
        transform=ax.get_xaxis_transform(),
        ha="right", va="top", fontsize="xx-small", color=rgb.tue_blue, rotation=90,
    )

    # Vline for the Themis price with text label
    ax.axvline(themis_price, color=rgb.tue_red, linestyle="-")
    ax.text(
        themis_price, 0.85, f"Themis price: {themis_price:.0f}",
        transform=ax.get_xaxis_transform(),
        ha="right", va="top", fontsize="xx-small", color=rgb.tue_red, rotation=90,
    )
    _style_spine(ax, "left", rgb.tue_orange)
    ax.set_xlabel(r"preferred price $p$ [EUR/tCO2e]")
    ax.set_ylabel("share of global emissions")
    ax.yaxis.label.set_color(rgb.tue_orange)
    ax.set_xlim(0, xplot.max())
    ax.set_ylim(0, shares.max() * 1.1)
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


def make_consequences_figure(
    price_preferences: np.ndarray,
    shares_pp: np.ndarray,
    countries: np.ndarray,
    themis_price: float,
    coalition_avg_pp: float,
    g: float = 0.1,
) -> Figure:
    """
    Show domestic payments and international transfers as a function of per-capita
    emissions for coalition members.

    A single straight line y = themis_price * e_i is drawn once.  The left axis
    reads domestic payment directly; the right axis is re-scaled by g and shifted
    so its zero sits at e_i = coalition_avg_pp, reading the international transfer
    g * themis_price * (e_i - coalition_avg_pp).
    """
    coalition_mask = price_preferences >= themis_price
    valid = coalition_mask & np.isfinite(shares_pp) & (shares_pp > 0)
    pp_c = shares_pp[valid]
    countries_c = countries[valid]

    fig, ax = plt.subplots(1, 1)

    if len(pp_c) == 0 or not np.isfinite(coalition_avg_pp):
        ax.text(0.5, 0.5, "No coalition data available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    x_max = 28.0
    x = np.linspace(0, x_max, 300)

    ax.plot(x, themis_price * x, color=rgb.tue_orange, lw=1.5)

    # Coalition member dots (only those within the x range)
    for pp_i in pp_c[pp_c <= x_max]:
        ax.scatter(pp_i, themis_price * pp_i, color=rgb.tue_blue, s=8, zorder=5)

    # Coalition members: labelled set at top edge in blue
    for c, pp_i in zip(countries_c, pp_c):
        if c in LABELLED_COUNTRIES and pp_i <= x_max:
            ax.text(
                pp_i, 0.99, LABELLED_COUNTRIES[c],
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize="xx-small", color=rgb.tue_blue, rotation=90,
            )

    # Non-coalition members: labelled set at top edge in red (no dot — not on the line)
    non_coalition_mask = ~coalition_mask
    valid_nc = non_coalition_mask & np.isfinite(shares_pp) & (shares_pp > 0)
    for c, pp_i in zip(countries[valid_nc], shares_pp[valid_nc]):
        if c in LABELLED_COUNTRIES and pp_i <= x_max:
            ax.text(
                pp_i, 0.99, LABELLED_COUNTRIES[c],
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize="xx-small", color=rgb.tue_red, rotation=90,
            )

    # Annotate any coalition members cut off by the x limit
    cutoff = [(c, pp_i) for c, pp_i in zip(countries_c, pp_c) if pp_i > x_max]
    if cutoff:
        note = "off chart: " + ", ".join(
            f"{LABELLED_COUNTRIES.get(c, c)} ({pp_i:.1f})" for c, pp_i in cutoff
        )
        ax.text(
            0.99, 0.01, note,
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize="xx-small", color=rgb.tue_blue,
        )

    # Vertical line at coalition average (zero-crossing of transfers)
    ax.axvline(coalition_avg_pp, color=rgb.tue_green, lw=0.75, linestyle="--")
    ax.text(
        coalition_avg_pp, 0.85, f"avg: {coalition_avg_pp:.1f}",
        transform=ax.get_xaxis_transform(),
        ha="right", va="top", fontsize="xx-small", color=rgb.tue_green, rotation=90,
    )

    # Left axis: domestic payment = themis_price * e_i
    left_max = themis_price * x_max
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, left_max)
    ax.set_xlabel("per capita emissions [tCO₂e/person]")
    ax.set_ylabel("domestic payment [EUR/person]")
    ax.yaxis.label.set_color(rgb.tue_orange)
    _style_spine(ax, "left", rgb.tue_orange)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(True, which="major", axis="x", color=rgb.tue_dark, zorder=-1)
    ax.grid(True, which="minor", axis="x", color=rgb.tue_gray, zorder=-1)

    # Right axis: international transfer = g * themis_price * (e_i - coalition_avg_pp)
    # Derived by applying the linear map  right = g * (left - themis_price * coalition_avg_pp)
    # to the left-axis limits, giving the correct scale and offset.
    offset = themis_price * coalition_avg_pp
    ax_r = ax.twinx()
    ax_r.set_ylim(g * (0 - offset), g * (left_max - offset))
    ax_r.axhline(0, color=rgb.tue_red, lw=0.75, linestyle="--")
    ax_r.set_ylabel("international transfer [EUR/person]")
    ax_r.yaxis.label.set_color(rgb.tue_red)
    _style_spine(ax_r, "right", rgb.tue_red)

    return fig


def _style_spine(ax, side: str, color) -> None:
    ax.spines[side].set_color(color)
    ax.tick_params(axis="y", colors=color)
