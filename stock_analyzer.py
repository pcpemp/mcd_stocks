import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import datetime

# --- Configuration ---
TRADING_DAYS_PER_YEAR = 252

# --- Helper Functions ---

def fetch_stock_data(ticker, start_date, end_date):
    """Fetches stock data from Yahoo Finance."""
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            return None, f"No data found for {ticker} in the selected period. Check ticker or date range."
        if len(stock_data) < 2:
            return None, f"Not enough data ({len(stock_data)} days) for {ticker} to calculate returns. Select a wider date range."
        return stock_data, None
    except Exception as e:
        return None, f"Error fetching data for {ticker}: {str(e)}"

def calculate_daily_returns(stock_data_df):
    """Calculates daily returns from stock data."""
    if 'Close' not in stock_data_df.columns or len(stock_data_df) < 2:
        return pd.Series(dtype=float) # Return empty Series
    daily_returns = stock_data_df['Close'].pct_change() * 100
    return daily_returns.dropna()

def create_distribution_chart(returns_series, title_prefix, bin_step=0.5, x_axis_label="Return Bins (%)"):
    """Creates a histogram distribution chart using Altair."""
    if returns_series.empty:
        st.info("No return data to chart.")
        return None

    df_returns = pd.DataFrame({'returns': returns_series})

    # Determine colors based on positive/negative bins
    # This is a bit more complex in Altair directly with binned data for color
    # We can create a 'color_group' column based on the midpoint of the bin
    # For simplicity in this direct translation, we'll let Altair handle colors,
    # or use a simpler coloring scheme if needed.
    # A more advanced approach would involve pre-calculating bin midpoints and assigning colors.

    chart = alt.Chart(df_returns).mark_bar().encode(
        alt.X('returns:Q', bin=alt.Bin(step=bin_step), title=x_axis_label, axis=alt.Axis(format='%')),
        alt.Y('count()', title='Frequency'),
        tooltip=[alt.X('returns:Q', bin=alt.Bin(step=bin_step), title=x_axis_label, format='.2f'), 'count()']
    ).properties(
        title=f'{title_prefix} (Bin Width: {bin_step}%)',
        width='container'
    )
    # Attempt to color bars (this is a common way, but might not split perfectly on 0 with Altair's auto-binning)
    # A more robust solution would be to create the bins manually, then color
    chart = chart.encode(
        color=alt.condition(
            alt.datum.returns >= 0, # This condition applies to the raw data, not the binned value's midpoint
            alt.value('rgba(34, 197, 94, 0.7)'),  # Green for positive
            alt.value('rgba(239, 68, 68, 0.7)')   # Red for negative
        )
    )
    # For a more accurate coloring of bins (e.g. if a bin crosses zero),
    # you would typically pre-calculate bins, determine their midpoint, and assign color.
    # Altair's transform_bin and transform_calculate could be used.

    return chart


# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Stock Analysis & Simulation")

st.title("📈 Stock Analysis & Simulation")
st.markdown("Fetch historical stock data, analyze daily returns, and run Monte Carlo simulations for annual performance.")

# Initialize session state variables
if 'daily_returns_data' not in st.session_state:
    st.session_state.daily_returns_data = None
if 'ticker_symbol' not in st.session_state:
    st.session_state.ticker_symbol = "AAPL" # Default

# --- Inputs ---
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Ticker Symbol:", value=st.session_state.ticker_symbol).upper()

# Set default end date to today, start date one year ago
today = datetime.date.today()
one_year_ago = today - datetime.timedelta(days=365)

start_date = st.sidebar.date_input("Start Date:", value=one_year_ago)
end_date = st.sidebar.date_input("End Date:", value=today)

# --- Daily Returns Analysis ---
st.header("1. Daily Returns Analysis")

if st.button("Analyze Daily Returns"):
    st.session_state.daily_returns_data = None # Reset previous results
    st.session_state.ticker_symbol = ticker # Store for potential re-use

    if not ticker:
        st.error("Please enter a ticker symbol.")
    elif not start_date or not end_date:
        st.error("Please select both start and end dates.")
    elif start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        with st.spinner(f"Fetching and analyzing daily data for {ticker}..."):
            stock_data_df, error_msg = fetch_stock_data(ticker, start_date, end_date)

            if error_msg:
                st.error(error_msg)
            elif stock_data_df is not None:
                daily_returns = calculate_daily_returns(stock_data_df)
                if daily_returns.empty:
                    st.warning("Could not calculate daily returns. Prices might be constant or insufficient distinct data points.")
                else:
                    st.session_state.daily_returns_data = daily_returns
                    st.success(f"Successfully analyzed daily returns for {ticker} ({len(daily_returns)} data points).")

if st.session_state.daily_returns_data is not None:
    st.subheader(f"Daily Gain/Loss Distribution for {st.session_state.ticker_symbol}")
    if not st.session_state.daily_returns_data.empty:
        daily_chart = create_distribution_chart(
            st.session_state.daily_returns_data,
            title_prefix=f"Distribution of Daily Returns for {st.session_state.ticker_symbol}",
            bin_step=0.5, # Adjust bin step as needed for daily returns
            x_axis_label="Daily Return Bins (%)"
        )
        if daily_chart:
            st.altair_chart(daily_chart, use_container_width=True)
        
        avg_daily_return = st.session_state.daily_returns_data.mean()
        std_daily_return = st.session_state.daily_returns_data.std()
        st.metric(label="Average Daily Return", value=f"{avg_daily_return:.2f}%")
        st.metric(label="Std. Dev. of Daily Returns", value=f"{std_daily_return:.2f}%")

    else:
        st.info("No daily returns data available to display chart.")


# --- Annual Return Simulation ---
st.header("2. Annual Return Simulation (Monte Carlo)")

if st.session_state.daily_returns_data is not None and not st.session_state.daily_returns_data.empty:
    num_simulations = st.number_input(
        "Number of Simulations:",
        min_value=100,
        max_value=100000, # Increased max from HTML
        value=1000,
        step=100,
        key="num_sims"
    )

    if st.button("Run Annual Simulations"):
        if st.session_state.daily_returns_data.empty:
            st.error("Please analyze daily returns first to provide a basis for simulation.")
        else:
            with st.spinner(f"Running {num_simulations} annual simulations... This may take a moment."):
                historical_returns_np = st.session_state.daily_returns_data.to_numpy() / 100.0 # Convert to decimal for calculation
                simulated_annual_returns = []

                for _ in range(num_simulations):
                    # Sample with replacement from historical daily returns
                    simulated_daily_path = np.random.choice(historical_returns_np, size=TRADING_DAYS_PER_YEAR, replace=True)
                    
                    # Calculate cumulative return for the year
                    # (1 + r1) * (1 + r2) * ... * (1 + rN)
                    cumulative_return = np.prod(1 + simulated_daily_path)
                    annual_return_percent = (cumulative_return - 1) * 100
                    simulated_annual_returns.append(annual_return_percent)
                
                st.session_state.simulated_annual_returns = pd.Series(simulated_annual_returns)
                st.success(f"Monte Carlo simulation completed with {num_simulations} runs.")

    if 'simulated_annual_returns' in st.session_state and st.session_state.simulated_annual_returns is not None:
        st.subheader(f"Simulated Annual Performance Distribution ({st.session_state.ticker_symbol})")
        annual_chart = create_distribution_chart(
            st.session_state.simulated_annual_returns,
            title_prefix="Simulated Annual Performance",
            bin_step=5.0, # Larger bin step for annual returns
            x_axis_label="Annual Return Bins (%)"
        )
        if annual_chart:
            st.altair_chart(annual_chart, use_container_width=True)

        # Display some key statistics from the simulation
        avg_sim_annual_return = st.session_state.simulated_annual_returns.mean()
        median_sim_annual_return = st.session_state.simulated_annual_returns.median()
        std_sim_annual_return = st.session_state.simulated_annual_returns.std()
        percentile_5 = np.percentile(st.session_state.simulated_annual_returns, 5)
        percentile_95 = np.percentile(st.session_state.simulated_annual_returns, 95)

        col1, col2, col3 = st.columns(3)
        col1.metric(label="Average Simulated Annual Return", value=f"{avg_sim_annual_return:.2f}%")
        col2.metric(label="Median Simulated Annual Return", value=f"{median_sim_annual_return:.2f}%")
        col3.metric(label="Std. Dev. of Sim. Annual Returns", value=f"{std_sim_annual_return:.2f}%")
        
        st.markdown(f"**Confidence Interval:** There's a 90% chance the simulated annual return falls between **{percentile_5:.2f}%** and **{percentile_95:.2f}%**.")

elif st.session_state.daily_returns_data is not None and st.session_state.daily_returns_data.empty:
    st.info("Cannot run annual simulation because no daily returns were calculated (e.g., data was flat).")
else:
    st.info("Analyze daily returns first to enable the annual simulation section.")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with [Streamlit](https://streamlit.io) & [yfinance](https://pypi.org/project/yfinance/).")
st.sidebar.markdown("Inspired by the provided HTML/JS template.")