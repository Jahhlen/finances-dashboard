# monte_carlo_module.py
import numpy as np
import pandas as pd
import yfinance as yf
import numpy_financial as npf
import matplotlib.pyplot as plt

# -----------------------------
# 1) Data Loader
# -----------------------------
def get_monthly_returns(assets, start="2010-01-01", years=20):
    dfs = []

    ticker_assets = {k: v for k, v in assets.items() if v["type"] == "ticker"}
    fixed_assets  = {k: v for k, v in assets.items() if v["type"] == "fixed"}

    # Download tickers in one go
    if ticker_assets:
        symbols = [cfg["symbol"] for cfg in ticker_assets.values()]
        raw = yf.download(symbols, start=start, progress=False)["Close"]
        raw = raw.dropna()
        if isinstance(raw, pd.Series):
            raw = raw.to_frame()
        monthly_prices = raw.resample("M").last()
        monthly_returns = monthly_prices.pct_change().dropna()
        rename_map = dict(zip(symbols, ticker_assets.keys()))
        monthly_returns = monthly_returns.rename(columns=rename_map)
        dfs.append(monthly_returns)

    # Fixed income synthetic series
    if fixed_assets:
        if dfs:
            common_index = dfs[0].index
        else:
            common_index = pd.date_range(start=start, periods=years*12, freq="M")
        for name, cfg in fixed_assets.items():
            monthly_rate = (1 + cfg["rate"])**(1/12) - 1
            fixed_series = pd.Series(monthly_rate, index=common_index, name=name)
            dfs.append(fixed_series)

    return pd.concat(dfs, axis=1).dropna()

# -----------------------------
# 2) Monte Carlo Simulation
# -----------------------------
def run_monte_carlo(monthly_returns,
                    weights,
                    years=20,
                    n_sims=1000,
                    monthly_contribution=1700,
                    initial_investment=34000,
                    inflation=0.0,
                    seed=None):
    if seed is not None:
        np.random.seed(seed)

    months = years * 12
    sim_ending = np.zeros(n_sims)
    sim_paths = np.zeros((n_sims, months))

    for s in range(n_sims):
        pv = float(initial_investment)
        path = []
        for t in range(months):
            rand_month = monthly_returns.sample(1, replace=True).iloc[0]
            port_return = np.dot(weights, rand_month.values)
            pv += monthly_contribution
            pv *= (1 + port_return)
            if inflation > 0:
                pv /= (1 + inflation/12.0)
            path.append(pv)
        sim_ending[s] = pv
        sim_paths[s, :] = path

    return sim_ending, sim_paths, months

# -----------------------------
# 3) Summary stats
# -----------------------------
def summary_stats(arr):
    return {
        "Median": float(np.median(arr)),
        "Mean": float(np.mean(arr)),
        "10th %ile": float(np.percentile(arr, 10)),
        "90th %ile": float(np.percentile(arr, 90)),
        "1st %ile": float(np.percentile(arr, 1)),
        "99th %ile": float(np.percentile(arr, 99)),
    }

# -----------------------------
# 4) IRR distribution
# -----------------------------
def irr_distribution(sim_paths, months, monthly_contribution=1700, initial_investment=0):
    irr_results = []
    for s in range(sim_paths.shape[0]):
        final_val = sim_paths[s, -1]
        cashflows = [-initial_investment] + [-monthly_contribution] * months
        cashflows[-1] += final_val
        irr_m = npf.irr(cashflows)
        if irr_m is not None and not np.isnan(irr_m):
            irr_y = (1 + irr_m) ** 12 - 1
            irr_results.append(irr_y)
    irr_results = np.array(irr_results)
    return {
        "Median IRR": float(np.percentile(irr_results, 50)),
        "10th %ile IRR": float(np.percentile(irr_results, 10)),
        "90th %ile IRR": float(np.percentile(irr_results, 90)),
    }

# -----------------------------
# 5) Plot (matplotlib for Streamlit)
# -----------------------------
def plot_paths(sim_paths, months, start_year=2025, monthly_contribution=1700,
               allocation_label="", inflation=0.0):
    median_path = np.median(sim_paths, axis=0)
    p10_path = np.percentile(sim_paths, 10, axis=0)
    p25_path = np.percentile(sim_paths, 25, axis=0)
    p75_path = np.percentile(sim_paths, 75, axis=0)
    p90_path = np.percentile(sim_paths, 90, axis=0)

    x = np.arange(1, months + 1) / 12.0 + start_year
    contributions = monthly_contribution * np.arange(1, months + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, median_path, label="Median outcome", linewidth=2, color="navy")
    ax.fill_between(x, p25_path, p75_path, alpha=0.3, color="blue", label="25–75% range")
    ax.fill_between(x, p10_path, p90_path, alpha=0.15, color="blue", label="10–90% range")
    ax.plot(x, contributions, linestyle="--", color="orange", label="Total contributions")

    step = 60
    for i in range(0, len(x), step):
        ax.scatter(x[i], median_path[i], color="red", zorder=5)
        ax.text(x[i], median_path[i], f"${median_path[i]/1e6:.2f}M",
                fontsize=8, ha="left", va="bottom", rotation=30)
    ax.scatter(x[-1], median_path[-1], color="red", zorder=5)
    ax.text(x[-1], median_path[-1], f"${median_path[-1]/1e6:.2f}M",
            fontsize=8, ha="left", va="bottom", rotation=30)

    title = f"Monte Carlo Simulation ({allocation_label})"
    if inflation > 0:
        title += " [Inflation-Adjusted]"
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig