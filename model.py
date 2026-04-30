import numpy as np

SAMPLE_MODELS = [
    "Independent Uniform",
    "Polluters are ambitious",
    "Polluters are stingy",
]

# Number of bridge discretisation steps (must match the alpha slider's step count).
N_BRIDGE_STEPS = 11


def precompute_random_draws(
    rng: np.random.Generator,
    n_countries: int,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw all seed-dependent random quantities up front.

    Returns
    -------
    prices_base : (n_countries,) uniform baseline prices
    bridge      : (N_BRIDGE_STEPS, n_countries) Brownian bridge realisations
    """
    prices_base = rng.uniform(low=lower, high=upper, size=n_countries)
    bridge = _brownian_bridge_samples(N_BRIDGE_STEPS, n_countries, rng)
    return prices_base, bridge


def sample_price_preferences(
    prices_base: np.ndarray,
    bridge: np.ndarray,
    shares_pp: np.ndarray,
    sample_model: str,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Compute a price preference for each country from pre-sampled random draws.

    Parameters
    ----------
    prices_base  : (n_countries,) baseline uniform prices from precompute_random_draws
    bridge       : (N_BRIDGE_STEPS, n_countries) bridge from precompute_random_draws
    shares_pp    : per-capita emissions array (used to order countries)
    sample_model : one of SAMPLE_MODELS
    alpha        : interpolation in [0, 1]; 0 = random baseline, 1 = sorted target
    """
    if sample_model == "Independent Uniform":
        return prices_base.copy()

    # Diffusion models: interpolate via a noisy bridge from a random baseline
    # to a target ordering that reflects per-capita emissions rank.
    if sample_model == "Polluters are ambitious":
        # High per-capita emitters tend toward higher preferred prices
        sort_order = np.argsort(shares_pp)
    else:  # "Polluters are stingy"
        # High per-capita emitters tend toward lower preferred prices
        sort_order = np.argsort(shares_pp)[::-1]

    prices_target = np.zeros_like(prices_base)
    prices_target[sort_order] = np.sort(prices_base)

    prices_path = (
        prices_base[None, :]
        + 5 * bridge
        + (prices_target - prices_base)[None, :] * np.linspace(0, 1, N_BRIDGE_STEPS)[:, None]
    )
    step = int(alpha * (N_BRIDGE_STEPS - 1))
    return prices_path[step].copy()


def compute_themis_price(
    price_preferences: np.ndarray,
    shares: np.ndarray,
    upper: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the Themis revenue-maximising carbon price.

    The Themis price p* maximises p * (fraction of global emissions from countries
    whose preferred price is at least p).

    Returns
    -------
    xplot        : price grid [0, upper + 5]
    SF           : survival function — fraction of global emissions covered at each price
    themis_price : revenue-maximising price
    """
    idx = np.argsort(price_preferences);
    xplot = np.append(np.append(0, price_preferences[idx]), upper + 5)
    SF = np.append(np.append(1, 1 - np.cumsum(shares[idx])), 0)
    themis_price = xplot[np.argmax(xplot[1:] * SF[:-1]) + 1]
    return xplot, SF, themis_price


def _brownian_bridge_samples(
    n_steps: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Return an (n_steps, n_samples) array of Brownian bridge realisations on [0, 1].

    Uses the eigendecomposition of the bridge covariance kernel
    k(s, t) = min(s, t) - s*t for efficient sampling.
    """
    x = np.linspace(0, 1, n_steps)
    K = np.minimum(x[:, None], x[None, :]) - x[:, None] * x[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    z = rng.normal(0, 1, (n_steps, n_samples))
    return eigenvectors @ (np.sqrt(eigenvalues)[:, None] * z)
