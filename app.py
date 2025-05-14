import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import datetime
import traceback

# --- Configuration ---
TRADING_DAYS_PER_YEAR = 252

# --- Helper Functions (keep them as they are, they are working) ---
def fetch_stock_data(ticker_symbol, start_date, end_date):
    """Fetches stock data from Yahoo Finance. Returns DataFrame or (None, error_msg)."""
    try:
        stock_data_df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
        if stock_data_df.empty:
            return None, f"No data found for {ticker_symbol} in the selected period. Check ticker or date range."
        if isinstance(stock_data_df.columns, pd.MultiIndex):
            if 'Close' not in stock_data_df.columns.levels[0] or \
               (len(stock_data_df.columns.levels) > 1 and ticker_symbol not in stock_data_df.columns.levels[1]):
                return None, f"Data for {ticker_symbol} (MultiIndex) does not contain an identifiable 'Close' column for this ticker."
        elif 'Close' not in stock_data_df.columns:
            return None, f"Data for {ticker_symbol} does not contain a 'Close' column. Ticker might be invalid or data unavailable."
        if len(stock_data_df) < 2:
            return None, f"Not enough data ({len(stock_data_df)} days) for {ticker_symbol} to calculate returns. Select a wider date range."
        return stock_data_df, None
    except Exception as e:
        st.error(f"Exception in fetch_stock_data for {ticker_symbol}: {str(e)}")
        return None, f"Error fetching data for {ticker_symbol}: {str(e)}. Check if the ticker symbol is valid."

def calculate_daily_returns(data_df_input, ticker_sym):
    """Calculates daily returns from stock data DataFrame. Returns a Pandas Series."""
    # st.write(f"Debug (calc_daily_returns): Input type: {type(data_df_input)}, Shape: {data_df_input.shape if isinstance(data_df_input, pd.DataFrame) else 'N/A'}")
    if not isinstance(data_df_input, pd.DataFrame):
        st.warning(f"calc_daily_returns: Expected DataFrame, got {type(data_df_input)}. Returning empty Series.")
        return pd.Series(dtype=float)
    if len(data_df_input) < 2:
        st.warning(f"calc_daily_returns: Not enough data rows ({len(data_df_input)} < 2). Returning empty Series.")
        return pd.Series(dtype=float)

    close_column_series = None
    try:
        if isinstance(data_df_input.columns, pd.MultiIndex):
            # st.write("Debug (calc_daily_returns): DataFrame has MultiIndex columns.")
            if ('Close', ticker_sym) in data_df_input.columns:
                close_column_series = data_df_input[('Close', ticker_sym)]
            elif 'Close' in data_df_input.columns.get_level_values(0) and \
                 ticker_sym in data_df_input.columns.get_level_values(1):
                 close_column_series = data_df_input.loc[:, ('Close', ticker_sym)]
            else: # Fallback: if only one ticker was downloaded, Close might be ('Close', '') or just 'Close' at level 0
                if 'Close' in data_df_input.columns.get_level_values(0):
                    temp_close_df = data_df_input['Close']
                    if isinstance(temp_close_df, pd.DataFrame) and len(temp_close_df.columns) == 1:
                        close_column_series = temp_close_df.iloc[:, 0]
                    elif isinstance(temp_close_df, pd.Series): # Should not happen if original was MultiIndex, but good to check
                        close_column_series = temp_close_df

            if close_column_series is None:
                st.warning(f"calc_daily_returns: 'Close' data for ticker '{ticker_sym}' not found in MultiIndex columns. Columns: {data_df_input.columns}")
                return pd.Series(dtype=float)

        elif 'Close' in data_df_input.columns:
            # st.write("Debug (calc_daily_returns): DataFrame has single-level columns. Accessing 'Close'.")
            close_column_series = data_df_input['Close']
        else:
            st.warning("calc_daily_returns: 'Close' column not found in DataFrame. Returning empty Series.")
            return pd.Series(dtype=float)

        # st.write(f"Debug (calc_daily_returns): Type of extracted 'close_column_series': {type(close_column_series)}")
        if not isinstance(close_column_series, pd.Series):
            st.error(f"calc_daily_returns: Extracted 'close_column_series' is not a Series (type: {type(close_column_series)}). Cannot proceed.")
            return pd.Series(dtype=float)

        close_prices_numeric = pd.to_numeric(close_column_series, errors='coerce')
        if close_prices_numeric.isnull().all():
            st.warning("calc_daily_returns: 'Close' column has no valid numeric data after coercion. Returning empty Series.")
            return pd.Series(dtype=float)

        daily_returns_series = close_prices_numeric.pct_change() #* 100
        result = daily_returns_series.dropna()

        return result
    except Exception as e:
        st.error(f"!!! Exception INSIDE calculate_daily_returns's try block: {e}")
        st.error(traceback.format_exc())
        return pd.Series(dtype=float)

def create_distribution_chart(returns_data, title_prefix, bin_step=0.5, x_axis_label="Return Bins (%)"):
    if not isinstance(returns_data, pd.Series): # Should be a series by now
        st.error(f"Chart function received non-Series: {type(returns_data)}")
        return None
    if returns_data.empty :
        st.info("No return data available to chart (Series is empty).")
        return None

    df_for_chart = pd.DataFrame({'returns_col': returns_data})
    chart = alt.Chart(df_for_chart).mark_bar().encode(
        alt.X('returns_col:Q', bin=alt.Bin(step=bin_step), title=x_axis_label, axis=alt.Axis(format='%')),
        alt.Y('count()', title='Frequency'),
        tooltip=[alt.X('returns_col:Q', bin=alt.Bin(step=bin_step), title=x_axis_label, format='.2%'), 'count()']
    ).properties(
        title=f'{title_prefix} (Bin Width: {bin_step}%)',
        width='container', # Ensures chart fits container width
        height=300 # You can set a fixed height or let it be responsive
    )
    chart = chart.encode(
        color=alt.condition(
            alt.datum.returns_col >= 0,
            alt.value('rgba(34, 197, 94, 0.7)'),
            alt.value('rgba(239, 68, 68, 0.7)')
        )
    )
    return chart
# --- END OF HELPER FUNCTIONS ---

st.set_page_config(layout="wide", page_title="Stock Analysis & Simulation")
st.title("📈 Stock Analysis & Simulation")

# --- Initialize session state variables ---
if 'fetched_stock_data_df' not in st.session_state:
    st.session_state.fetched_stock_data_df = None
if 'daily_returns_series' not in st.session_state:
    st.session_state.daily_returns_series = None
if 'ticker_symbol' not in st.session_state:
    st.session_state.ticker_symbol = "AAPL"
if 'simulated_annual_returns_series' not in st.session_state:
    st.session_state.simulated_annual_returns_series = None

# --- Sidebar for Inputs and Actions ---
st.sidebar.header("Input Parameters")
ticker_input = st.sidebar.text_input("Ticker Symbol:", value=st.session_state.ticker_symbol).upper()
today = datetime.date.today()
one_year_ago = today - datetime.timedelta(days=365) # Default start date
start_date_input = st.sidebar.date_input("Start Date:", value=one_year_ago)
end_date_input = st.sidebar.date_input("End Date:", value=today)

st.sidebar.markdown("---") # Separator

if st.sidebar.button("Analyze Daily Returns"):
    st.session_state.fetched_stock_data_df = None
    st.session_state.daily_returns_series = None
    st.session_state.simulated_annual_returns_series = None # Reset simulation too
    st.session_state.ticker_symbol = ticker_input

    if not ticker_input:
        st.error("Please enter a ticker symbol.")
    elif not start_date_input or not end_date_input:
        st.error("Please select both start and end dates.")
    elif start_date_input >= end_date_input:
        st.error("Start date must be before end date.")
    else:
        with st.spinner(f"Fetching and analyzing daily data for {ticker_input}..."):
            fetched_df, error_msg = fetch_stock_data(ticker_input, start_date_input, end_date_input)
            if error_msg:
                st.error(error_msg)
            elif fetched_df is not None:
                st.session_state.fetched_stock_data_df = fetched_df
                calculated_returns = calculate_daily_returns(st.session_state.fetched_stock_data_df, ticker_input)
                if not isinstance(calculated_returns, pd.Series):
                    st.error("Internal error: Daily returns calculation did not return a Series as expected.")
                    st.session_state.daily_returns_series = None
                elif calculated_returns.empty:
                    st.warning("Could not calculate daily returns. Prices might be constant, data insufficient, or ticker invalid for the period.")
                    st.session_state.daily_returns_series = calculated_returns
                else:
                    st.session_state.daily_returns_series = calculated_returns
                    st.success(f"Successfully analyzed daily returns for {ticker_input} ({len(st.session_state.daily_returns_series)} data points).")
            else:
                st.error("Unknown error: Fetched data is None but no error message was provided by fetch_stock_data.")

st.sidebar.markdown("---")
st.sidebar.header("Monte Carlo Simulation")

# Simulation input and button are only active if daily returns are available
enable_simulation_widgets = st.session_state.daily_returns_series is not None and \
                           not st.session_state.daily_returns_series.empty

num_simulations_input = st.sidebar.number_input(
    "Number of Simulations:",
    min_value=100, max_value=100000,
    value=10000, step=100, key="num_sims",
    disabled=not enable_simulation_widgets # Disable if no daily returns
)

if st.sidebar.button("Run Annual Simulations", disabled=not enable_simulation_widgets):
    if st.session_state.daily_returns_series is None or st.session_state.daily_returns_series.empty:
        st.warning("Please analyze daily returns first to provide a basis for simulation.")
    else:
        with st.spinner(f"Running {num_simulations_input} annual simulations..."):
            if st.session_state.daily_returns_series.isnull().all():
                st.warning("Cannot run simulation: all daily returns are NaN.")
                st.session_state.simulated_annual_returns_series = pd.Series(dtype=float)
            else:
                historical_returns_np = st.session_state.daily_returns_series.dropna().to_numpy() / 100.0
                if historical_returns_np.size == 0:
                    st.warning("Cannot run simulation: no valid daily returns after dropping NaN.")
                    st.session_state.simulated_annual_returns_series = pd.Series(dtype=float)
                else:
                    simulated_annual_returns_list = []
                    for _ in range(num_simulations_input):
                        simulated_daily_path = np.random.choice(historical_returns_np, size=TRADING_DAYS_PER_YEAR, replace=True)
                        cumulative_return = np.prod(1 + simulated_daily_path)
                        annual_return_percent = (cumulative_return - 1) * 100
                        simulated_annual_returns_list.append(annual_return_percent)
                    st.session_state.simulated_annual_returns_series = pd.Series(simulated_annual_returns_list)
                    st.success(f"Monte Carlo simulation completed with {num_simulations_input} runs.")

# --- Main Area for Tabs and Visualizations ---
if st.session_state.daily_returns_series is not None or st.session_state.simulated_annual_returns_series is not None:
    tab1, tab2 = st.tabs(["🗓️ Daily Return Distribution", "🎲 Annual Return Simulation"])

    with tab1:
        st.header(f"Daily Gain/Loss Distribution for {st.session_state.ticker_symbol}")
        if st.session_state.daily_returns_series is not None:
            if st.session_state.daily_returns_series.empty:
                st.info("No daily returns data available to display chart (e.g., data was flat, insufficient, or ticker invalid).")
            else:
                daily_chart = create_distribution_chart(
                    st.session_state.daily_returns_series,
                    title_prefix=f"Daily Returns for {st.session_state.ticker_symbol}",
                    bin_step=0.005,
                    x_axis_label="Daily Return Bins (%)"
                )
                if daily_chart:
                    st.altair_chart(daily_chart, use_container_width=True)
                avg_daily_return = st.session_state.daily_returns_series.mean()*100
                std_daily_return = st.session_state.daily_returns_series.std()*100
                col1, col2 = st.columns(2)
                col1.metric(label="Average Daily Return", value=f"{avg_daily_return:.2f}%" if pd.notna(avg_daily_return) else "N/A")
                col2.metric(label="Std. Dev. of Daily Returns", value=f"{std_daily_return:.2f}%" if pd.notna(std_daily_return) else "N/A")
        else:
            st.info("Analyze daily returns first to see the distribution.")

    with tab2:
        st.header(f"Simulated Annual Performance for {st.session_state.ticker_symbol}")
        if st.session_state.simulated_annual_returns_series is not None:
            if st.session_state.simulated_annual_returns_series.empty:
                st.info("No simulated annual returns data to display. Run the simulation first or check daily returns data.")
            else:
                annual_chart = create_distribution_chart(
                    st.session_state.simulated_annual_returns_series,
                    title_prefix="Simulated Annual Performance",
                    bin_step=0.05,
                    x_axis_label="Annual Return Bins (%)"
                )
                if annual_chart:
                    st.altair_chart(annual_chart, use_container_width=True)

                avg_sim_annual_return = st.session_state.simulated_annual_returns_series.mean()*100
                median_sim_annual_return = st.session_state.simulated_annual_returns_series.median()*100
                std_sim_annual_return = st.session_state.simulated_annual_returns_series.std()*100
                sim_returns_for_percentile = st.session_state.simulated_annual_returns_series.dropna()*100
                percentile_5, percentile_95 = np.nan, np.nan # Initialize
                if not sim_returns_for_percentile.empty:
                    percentile_5 = np.percentile(sim_returns_for_percentile, 2.5)
                    percentile_95 = np.percentile(sim_returns_for_percentile, 97.5)

                col1, col2, col3 = st.columns(3)
                col1.metric(label="Avg. Sim. Annual Return", value=f"{avg_sim_annual_return:.2f}%" if pd.notna(avg_sim_annual_return) else "N/A")
                col2.metric(label="Median Sim. Annual Return", value=f"{median_sim_annual_return:.2f}%" if pd.notna(median_sim_annual_return) else "N/A")
                col3.metric(label="Std. Dev. Sim. Annual Returns", value=f"{std_sim_annual_return:.2f}%" if pd.notna(std_sim_annual_return) else "N/A")
                if pd.notna(percentile_5) and pd.notna(percentile_95):
                    st.markdown(f"**95% Confidence Interval (Simulated):** {percentile_5:.2f}% to {percentile_95:.2f}%")
                else:
                    st.markdown("**95% Confidence Interval (Simulated):** Not enough data to calculate.")
        elif st.session_state.daily_returns_series is not None and not st.session_state.daily_returns_series.empty:
             st.info("Run the annual simulation to see results here.")
        else:
            st.info("Analyze daily returns and then run the annual simulation to see results here.")
else:
    st.info("Enter a ticker symbol and click 'Analyze Daily Returns' in the sidebar to begin.")


st.sidebar.markdown("---")
st.sidebar.markdown("Built with [Streamlit](https://streamlit.io) & [yfinance](https://pypi.org/project/yfinance/).")