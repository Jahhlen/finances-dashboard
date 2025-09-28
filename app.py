# app.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy_financial as npf
from monte_carlo_module import (
    get_monthly_returns,
    run_monte_carlo,
    summary_stats,
    irr_distribution,
    plot_paths,
)

@st.cache_data(show_spinner=False, ttl=60*60)
def cached_monthly_returns(assets, years):
    return get_monthly_returns(assets, years=years)

st.set_page_config(page_title="Monte Carlo Portfolio Simulator", layout="wide")

st.title("📊 Monte Carlo Portfolio Simulator")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Simulation Parameters")

years = st.sidebar.slider("Years", 5, 40, 20)
n_sims = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, step=100)
monthly_contribution = st.sidebar.number_input("Monthly Contribution (SGD)", 0, 10000, 3000, step=100)
initial_investment = st.sidebar.number_input("Initial Investment (SGD)", 0, 1000000, 0, step=1000)
inflation = st.sidebar.slider("Inflation Rate (%)", 0.0, 0.10, 0.02, step=0.005)

# -----------------------------
# Dynamic Asset Inputs
# -----------------------------
st.sidebar.subheader("Portfolio Assets")

num_assets = st.sidebar.number_input("Number of Assets", 1, 10, 2)

assets = {}
weights = []

for i in range(num_assets):
    st.sidebar.markdown(f"**Asset {i+1}**")
    asset_name = st.sidebar.text_input(f"Name {i+1}", f"Asset{i+1}", key=f"name_{i}")
    asset_type = st.sidebar.selectbox(f"Type {i+1}", ["ticker", "fixed"], key=f"type_{i}")

    if asset_type == "ticker":
        symbol = st.sidebar.text_input(f"Ticker Symbol {i+1}", "IWDA.L", key=f"symbol_{i}")
        assets[asset_name] = {"type": "ticker", "symbol": symbol}
    else:
        rate = st.sidebar.number_input(f"Fixed Annual Rate {i+1}", 0.0, 0.15, 0.03, step=0.001, key=f"rate_{i}")
        assets[asset_name] = {"type": "fixed", "rate": rate}

    weight = st.sidebar.number_input(f"Weight {i+1}", 0.0, 1.0, 1.0 / num_assets, step=0.05, key=f"weight_{i}")
    weights.append(weight)

# Normalise weights
weights = [w / sum(weights) for w in weights]

# -----------------------------
# Run Button
# -----------------------------
if st.sidebar.button("🚀 Run Simulation"):
    st.subheader("Running Monte Carlo Simulation...")

    # Run engine
    monthly_returns = get_monthly_returns(assets, years=years)
    sim_ending, sim_paths, months = run_monte_carlo(
        monthly_returns,
        weights,
        years=years,
        n_sims=n_sims,
        monthly_contribution=monthly_contribution,
        initial_investment=initial_investment,
        inflation=inflation,
    )

    # -----------------------------
    # Projection Summary
    # -----------------------------
    st.markdown("### 📈 Projection Summary")

    summary = summary_stats(sim_ending)

    col1, col2, col3 = st.columns(3)
    col1.metric("Median", f"${summary['Median']:,.0f}")
    col2.metric("10th %ile", f"${summary['10th %ile']:,.0f}")
    col3.metric("90th %ile", f"${summary['90th %ile']:,.0f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Mean", f"${summary['Mean']:,.0f}")
    col5.metric("1st %ile", f"${summary['1st %ile']:,.0f}")
    col6.metric("99th %ile", f"${summary['99th %ile']:,.0f}")

    # -----------------------------
    # Projection Chart
    # -----------------------------
    st.markdown("### 📉 Projection Chart")
    fig = plot_paths(sim_paths, months,
                     start_year=2025,
                     monthly_contribution=monthly_contribution,
                     allocation_label=" + ".join(assets.keys()),
                     inflation=inflation)
    st.pyplot(fig)

    # -----------------------------
    # IRR Distribution
    # -----------------------------
    st.markdown("### 📊 IRR Distribution")
    irr = irr_distribution(sim_paths, months,
                           monthly_contribution=monthly_contribution,
                           initial_investment=initial_investment)

    col1, col2, col3 = st.columns(3)
    col1.metric("Median IRR", f"{irr['Median IRR']:.2%}")
    col2.metric("10th %ile IRR", f"{irr['10th %ile IRR']:.2%}")
    col3.metric("90th %ile IRR", f"{irr['90th %ile IRR']:.2%}")

    # Histogram for IRR
    st.markdown("#### IRR Histogram")
    irr_values = []
    for s in range(sim_paths.shape[0]):
        final_val = sim_paths[s, -1]
        cashflows = [-initial_investment] + [-monthly_contribution] * months
        cashflows[-1] += final_val
        irr_m = npf.irr(cashflows)
        if irr_m is not None:
            irr_y = (1 + irr_m) ** 12 - 1
            irr_values.append(irr_y)

    fig2, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(irr_values, bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Annualised IRR across simulations")
    ax.set_xlabel("IRR")
    ax.set_ylabel("Frequency")
    st.pyplot(fig2)

else:
    st.info("Adjust parameters in the sidebar and click **Run Simulation** to start.")