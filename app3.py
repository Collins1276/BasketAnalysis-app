import streamlit as st
import pandas as pd
from apyori import apriori
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Supermarket Basket Analysis", layout="wide")

# --- Upload Files ---
st.sidebar.header("Upload Data Files")
stats_file = st.sidebar.file_uploader("Upload Business Statistics Data (CSV)", type=["csv"], key="stats")
arl_file = st.sidebar.file_uploader("Upload ARL Data (CSV)", type=["csv"], key="arl")

# --- Load Statistics Data ---
@st.cache_data
def load_stats_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# --- Load ARL Data ---
@st.cache_data
def load_arl_data(file):
    df = pd.read_csv(file, header=None)
    return df

# --- Prepare Transactions for ARL ---
def prepare_transactions(df):
    transactions = []
    for i in range(len(df)):
        transaction = [str(df.values[i, j]) for j in range(df.shape[1]) if pd.notnull(df.values[i, j])]
        transactions.append(transaction)
    return transactions

# --- Compute Business Statistics ---
def get_statistics(df):
    stats = {}
    stats['Total Transactions'] = df['TransactionID'].nunique()
    stats['Total Unique Items Sold'] = df['Item'].nunique()
    stats['Total Quantity Sold'] = len(df)
    stats['Total Revenue'] = df['Price'].sum() if 'Price' in df.columns else 'N/A'
    basket_sizes = df.groupby('TransactionID')['Item'].count()
    stats['Average Basket Size'] = round(basket_sizes.mean(), 2)
    top_items = df['Item'].value_counts().head(5)
    stats['Most Bought Item'] = top_items.index[0]
    stats['Top 5 Items'] = top_items.to_dict()
    return stats, top_items

# --- Business Statistics Section ---
if stats_file:
    df_stats = load_stats_data(stats_file)

    st.title("🛒 Supermarket Basket Analysis Dashboard")

    # --- Filters ---
    if 'Category' in df_stats.columns:
        categories = df_stats['Category'].unique().tolist()
        selected_category = st.selectbox("Filter by Category", ["All"] + categories)
        if selected_category != "All":
            df_stats = df_stats[df_stats['Category'] == selected_category]

    time_option = st.selectbox("Select timeframe", ["Last 1 Day", "Last 7 Days", "Last 30 Days"])
    today = datetime.today()
    if time_option == "Last 1 Day":
        filtered_df = df_stats[df_stats['Date'] >= today - timedelta(days=1)]
    elif time_option == "Last 7 Days":
        filtered_df = df_stats[df_stats['Date'] >= today - timedelta(days=7)]
    else:
        filtered_df = df_stats[df_stats['Date'] >= today - timedelta(days=30)]

    # --- Display Statistics ---
    st.subheader("📊 Business Statistics")
    if filtered_df.empty:
        st.warning("No transactions found for this period.")
    else:
        stats, top_items = get_statistics(filtered_df)
        for key, val in stats.items():
            st.markdown(f"**{key}:** {val}")

        # --- Top 5 Items Chart ---
        st.subheader("🧱 Top 5 Items Sold")
        top_df = top_items.reset_index()
        top_df.columns = ['Item', 'Count']
        st.bar_chart(top_df.set_index('Item'))

        # --- Daily Sales Trend ---
        st.subheader("📈 Daily Sales Trend")
        daily_trend = filtered_df.groupby(filtered_df['Date'].dt.date).size().reset_index(name='Sales')
        fig = px.line(daily_trend, x='Date', y='Sales', markers=True, title='Number of Transactions Per Day')
        st.plotly_chart(fig, use_container_width=True)

# --- Association Rule Learning Section ---
if arl_file:
    st.subheader("🤖 Association Rules Mining (ARL)")
    df_arl = load_arl_data(arl_file)
    transactions = prepare_transactions(df_arl)

    with st.spinner("Mining association rules..."):
        rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
        results = list(rules)

    if not results:
        st.info("No strong association rules found.")
    else:
        def inspect(results):
            lhs = [tuple(result.items)[0] for result in results]
            rhs = [tuple(result.ordered_statistics[0].items_add)[0] for result in results]
            supports = [result.support for result in results]
            confidences = [result.ordered_statistics[0].confidence for result in results]
            lifts = [result.ordered_statistics[0].lift for result in results]
            return list(zip(lhs, rhs, supports, confidences, lifts))

        results_df = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
        st.dataframe(results_df)

        csv = results_df.to_csv(index=False).encode()
        st.download_button("Download ARL Results", data=csv, file_name="association_rules.csv", mime='text/csv')
