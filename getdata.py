import yfinance as yf

data = yf.download("AAPL", start="2000-01-01")
data.to_csv("data/APPL.csv")

data = yf.download("GOOGL", start="2000-01-01")
data.to_csv("data/google.csv")

data = yf.download("MSFT", start="2000-01-01")
data.to_csv("data/MSFT.csv")

data = yf.download("AMZN", start="2000-01-01")
data.to_csv("data/AMZN.csv")

data = yf.download("NVDA", start="2000-01-01")
data.to_csv("data/nvda.csv")
