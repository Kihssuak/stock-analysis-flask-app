from flask import Flask, request, render_template  
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

app = Flask(__name__)

# Mapping dictionary: keys can be ticker symbols (without .NS) or full company names
company_mapping = {
    "RELIANCE": "RELIANCE.NS",
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "TATA CONSULTANCY SERVICES": "TCS.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "INFY": "INFY.NS",
    "INFOSYS": "INFY.NS",
    "SBIN": "SBIN.NS",
    "STATE BANK OF INDIA": "SBIN.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "AXISBANK": "AXISBANK.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "MARUTI": "MARUTI.NS",
    "MARUTI SUZUKI": "MARUTI.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "TITAN": "TITAN.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "ADANIENT": "ADANIENT.NS",
    "WIPRO": "WIPRO.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "TECHM": "TECHM.NS",
    "POWERGRID": "POWERGRID.NS",
    "NESTLEIND": "NESTLEIND.NS",
    "DIVISLAB": "DIVISLAB.NS",
    "M&M": "M&M.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    "NTPC": "NTPC.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    "HCLTECH": "HCLTECH.NS",
    "HDFCLIFE": "HDFCLIFE.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "ONGC": "ONGC.NS",
    "GRASIM": "GRASIM.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "DRREDDY": "DRREDDY.NS",
    "COALINDIA": "COALINDIA.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "EICHERMOT": "EICHERMOT.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "BPCL": "BPCL.NS",
    "CIPLA": "CIPLA.NS",
    "UPL": "UPL.NS",
    "SBILIFE": "SBILIFE.NS",
    "ICICIPRULI": "ICICIPRULI.NS",
    "HINDALCO": "HINDALCO.NS",
    "DABUR": "DABUR.NS"
}

def strip_html(raw_text):
    """Removes HTML tags using BeautifulSoup."""
    if not raw_text:
        return ""
    return BeautifulSoup(raw_text, "html.parser").get_text()

def get_stock_news(stock):
    """
    Fetches the top 5 news articles related to the stock using Google News RSS feed.
    Returns:
      - news_articles: list of dicts with keys "title" and "link"
      - combined_text: a single string for summarization
    """
    try:
        rss_url = f"https://news.google.com/rss/search?q={stock}"
        feed = feedparser.parse(rss_url)
        articles = feed.entries[:5]

        news_articles = []
        combined_text = ""
        for article in articles:
            title = strip_html(article.title)
            summary = strip_html(article.summary)
            link = article.link
            combined_text += f"{title}. {summary} "
            news_articles.append({"title": title, "link": link})
        return news_articles, combined_text if combined_text else "No news available."
    except Exception as e:
        return [], "No news available."

def summarize_text(text, sentence_count=3):
    """
    Summarizes the provided text using Sumy's LSA summarizer.
    Returns the original text if summarization fails.
    """
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, sentence_count)
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        return summary if summary else text
    except Exception:
        return text

def get_stock_analysis(stock):
    try:
        df = yf.download(stock, period='3mo', interval='1d', auto_adjust=False)
        if df.empty or len(df.index) < 44:
            return {"error": "Not enough data for analysis."}

        df['SMA_44'] = df['Close'].rolling(window=44).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()

        latest_close = float(df['Close'].iloc[-1])
        latest_sma = float(df['SMA_44'].iloc[-1])
        latest_ema = float(df['EMA_20'].iloc[-1])
        support = float(df['Support'].iloc[-1])
        resistance = float(df['Resistance'].iloc[-1])

        if pd.isna(latest_close) or pd.isna(latest_ema) or pd.isna(support) or pd.isna(resistance):
            return {"error": "Data contains NaN values. Not enough valid data for analysis."}

        signal = "Buy" if latest_close > latest_ema else "Sell"

        risk = latest_close - support
        reward = resistance - latest_close
        risk_reward_ratio = round(reward / risk, 2) if risk > 0 else None

        hist_data = df.tail(30)
        if isinstance(hist_data.columns, pd.MultiIndex):
            hist_data.columns = hist_data.columns.droplevel(1)
        dates = hist_data.index.strftime("%Y-%m-%d").tolist()
        prices = hist_data["Close"].round(2).tolist()

        analysis = {
            'stock': stock,
            'close': latest_close,
            'SMA_44': latest_sma,
            'EMA_20': latest_ema,
            'support': support,
            'resistance': resistance,
            'signal': signal,
            'risk_reward_ratio': risk_reward_ratio,
            'chart_data': {
                'dates': dates,
                'prices': prices
            }
        }

        news_articles, combined_text = get_stock_news(stock)
        summary = summarize_text(combined_text)
        analysis['news_summary'] = summary
        analysis['news_articles'] = news_articles

        return analysis

    except Exception as e:
        return {"error": f"Error fetching data: {str(e)}"}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        user_input = request.form.get("symbol")
        if user_input:
            user_input = user_input.strip().upper()
            if user_input in company_mapping:
                stock_symbol = company_mapping[user_input]
            else:
                stock_symbol = user_input if user_input.endswith(".NS") else user_input + ".NS"
            analysis = get_stock_analysis(stock_symbol)
            if analysis is None or "error" in analysis:
                error = analysis.get("error", "Error occurred during analysis.")
            else:
                result = analysis
        else:
            error = "Please enter a valid stock symbol or company name."
    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
