import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests

st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}


@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.apply(pd.to_numeric, errors="coerce")
    return data


def add_indicators(data):

    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()

    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    exp12 = data["Close"].ewm(span=12, adjust=False).mean()
    exp26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = exp12 - exp26
    data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_hist"] = data["MACD"] - data["MACD_signal"]

    return data


def get_news_sentiment(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=54e9b6501b464fc9b583a604496f1897"
    response = requests.get(url)
    articles = response.json().get("articles", [])

    sentiments = []
    wordcloud_text = []
    for article in articles[:10]:
        text = article["title"] + " " + article.get("content", "")
        analysis = TextBlob(text)
        sentiments.append(analysis.sentiment.polarity)
        wordcloud_text.append(text)

    return {
        "sentiments": sentiments,
        "wordcloud": " ".join(wordcloud_text),
        "articles": articles[:10],
    }


st.title("📈 Stock Market Dashboard")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Page",
        ["Market Overview", "Stock Analysis", "Market News", "Investor Education"],
    )

if page == "Stock Analysis":
    with st.sidebar:
        st.header("Analysis Controls")
        ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", datetime.now())
        selected_indicators = st.multiselect(
            "Technical Indicators",
            ["SMA_50", "SMA_200", "RSI", "MACD"],
            default=["SMA_50", "RSI"],
        )

    data = load_data(ticker, start_date, end_date)
    if data.empty:
        st.error("Invalid ticker or no data available")

    data = add_indicators(data)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Real-time Analysis",
            "Historical Trends",
            "Portfolio Tracker",
            "News Sentiment",
        ]
    )

    with tab1:

        st.subheader(f"Real-time Analysis for {ticker}")
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
        )

        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"].values,
                high=data["High"].values,
                low=data["Low"].values,
                close=data["Close"].values,
                name="Price",
            ),
            row=1,
            col=1,
        )

        if "SMA_50" in selected_indicators:
            fig.add_trace(
                go.Scatter(x=data.index, y=data["SMA_50"].values, name="50-day SMA"),
                row=1,
                col=1,
            )

        if "SMA_200" in selected_indicators:
            fig.add_trace(
                go.Scatter(x=data.index, y=data["SMA_200"].values, name="200-day SMA"),
                row=1,
                col=1,
            )

        if "RSI" in selected_indicators:
            fig.add_trace(
                go.Scatter(x=data.index, y=data["RSI"].values, name="RSI"),
                row=2,
                col=1,
            )

        if "MACD" in selected_indicators:
            fig.add_trace(
                go.Bar(x=data.index, y=data["MACD_hist"].values, name="MACD Histogram"),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data["MACD"].values, name="MACD Line"),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data["MACD_signal"].values, name="Signal Line"
                ),
                row=2,
                col=1,
            )

        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:

        st.subheader("Historical Price Analysis")

        col1, col2 = st.columns(2)
        with col1:

            st.markdown("**Volume Analysis**")
            fig = go.Figure(go.Bar(x=data.index, y=data["Volume"]))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:

            st.markdown("**Daily Returns Distribution**")
            returns = data["Close"].pct_change().dropna()
            fig = go.Figure(go.Histogram(x=returns, nbinsx=50))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Feature Correlation Matrix**")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = data[numeric_cols].corr()
        fig = go.Figure(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
            )
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:

        st.subheader("Portfolio Tracker")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Add Stock to Portfolio**")
            new_ticker = st.text_input("Ticker", key="pt_ticker").upper()
            shares = st.number_input("Shares", min_value=0.1, step=0.1)
            if st.button("Add to Portfolio"):
                if new_ticker:
                    st.session_state.portfolio[new_ticker] = shares

        with col2:
            st.markdown("**Current Portfolio**")
            if st.session_state.portfolio:
                portfolio_df = (
                    pd.DataFrame.from_dict(
                        st.session_state.portfolio,
                        orient="index",
                        columns=["Shares"],
                    )
                    .reset_index()
                    .rename(columns={"index": "Ticker"})
                )
                st.dataframe(portfolio_df, hide_index=True)
            else:
                st.info("No stocks in portfolio")

        if st.session_state.portfolio:
            fig = go.Figure(
                go.Pie(
                    labels=list(st.session_state.portfolio.keys()),
                    values=pd.Series(st.session_state.portfolio),
                )
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:

        st.subheader("News Sentiment Analysis")
        sentiment_data = get_news_sentiment(ticker)

        col1, col2 = st.columns(2)
        with col1:

            st.markdown("**News Word Cloud**")
            if sentiment_data["wordcloud"]:
                wordcloud = WordCloud(width=800, height=400).generate(
                    sentiment_data["wordcloud"]
                )
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud)
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.warning("No news articles found")

        with col2:

            st.markdown("**Sentiment Scores**")
            if sentiment_data["sentiments"]:
                avg_sentiment = np.mean(sentiment_data["sentiments"])
                st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")

                fig = go.Figure(go.Histogram(x=sentiment_data["sentiments"], nbinsx=10))
                fig.update_layout(height=300, title="Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No sentiment data available")

        st.markdown("**Recent News Articles**")
        for article in sentiment_data["articles"]:
            st.markdown(
                f"""
            - **{article['title']}**  
            {article['description']}  
            [Read more]({article['url']})
            """
            )

elif page == "Market News":
    st.subheader("Financial Market News")

    url = f"https://newsapi.org/v2/everything?domains=wsj.com&apiKey=54e9b6501b464fc9b583a604496f1897"

    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])

        cols = st.columns(3)
        for i, article in enumerate(articles[:9]):
            with cols[i % 3]:
                st.markdown(
                    f"""
                <div class="metric-box">
                    <h4>{article['title']}</h4>
                    <p>{article['description'][:100]}...</p>
                    <a href="{article['url']}" target="_blank">Read more</a>
                </div>
                """,
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.error(f"Error loading news: {str(e)}")

elif page == "Investor Education":
    st.subheader("Investor Education Resources")

    with st.expander("📚 Basic Concepts"):
        st.markdown(
            """
        - **Stocks**: Ownership in a company
        - **Bonds**: Debt instruments
        - **ETF**: Exchange-Traded Funds
        - **Market Cap**: Company valuation
        """
        )

    with st.expander("📈 Technical Analysis"):
        st.markdown(
            """
        - Support and Resistance levels
        - Moving Averages
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        """
        )

    with st.expander("📊 Fundamental Analysis"):
        st.markdown(
            """
        - Financial Statements
        - P/E Ratio
        - EPS (Earnings Per Share)
        - Dividend Yield
        """
        )

    st.markdown(
        """
    
    - [Investopedia](https://www.investopedia.com)
    - [Morningstar Education](https://www.morningstar.com/education)
    - [SEC Investor Education](https://www.investor.gov)
    """
    )

elif page == "Market Overview":
    st.header("Market Overview")

    st.subheader("Major Indices")
    indices = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei 225",
    }

    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (ticker, name) in enumerate(indices.items()):
        try:
            with cols[i % 3]:
                start_date = datetime.now() - timedelta(days=365)
                end_date = datetime.now()
                data = yf.download(ticker, start=start_date, end=end_date)
                if not data.empty:
                    current_close = data["Close"].iloc[-1].item()
                    prev_close = data["Close"].iloc[0].item()
                    change = ((current_close - prev_close) / prev_close) * 100
                    st.markdown(
                        f"""
                    <div class="metric-box">
                        <h3>{name}</h3>
                        <h4>${current_close:,.2f}</h4>
                        <p style="color: {'#16e16e' if change >=0 else 'red'}">
                            {change:.2f}%
                        </p>
                        <small>1 Year Change</small>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(f"Data unavailable for {name}")
        except Exception as e:
            st.error(f"Error loading {name}: {str(e)}")

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    st.subheader("Major Stocks")
    stocks = {
        "AAPL": "Apple",
        "GOOGL": "Alphabet (Google)",
        "NVDA": "NVIDIA",
        "AMZN": "Amazon",
        "META": "Meta",
        "JPM": "JP Morgan",
        "TSLA": "Tesla",
        "MSFT": "Microsoft",
    }

    st.markdown("<br>", unsafe_allow_html=True)

    stock_cols = st.columns(3)
    for i, (ticker, name) in enumerate(stocks.items()):
        try:
            with stock_cols[i % 3]:
                start_date = datetime.now() - timedelta(days=365)
                end_date = datetime.now()
                data = yf.download(ticker, start=start_date, end=end_date)
                if not data.empty:
                    current_price = data["Close"].iloc[-1].item()
                    prev_price = data["Close"].iloc[0].item()
                    change = ((current_price - prev_price) / prev_price) * 100
                    st.markdown(
                        f"""
                    <div class="metric-box">
                        <h3>{name}</h3>
                        <h4>${current_price:,.2f}</h4>
                        <p style="color: {'#16e16e' if change >=0 else 'red'}">
                            {change:.2f}%
                        </p>
                        <small>1 Year Performance</small>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(f"Data unavailable for {name}")
        except Exception as e:
            st.error(f"Error loading {name}: {str(e)}")
