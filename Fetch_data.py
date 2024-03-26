import yfinance as yf

# Function to fetch latest stock data from Yahoo Finance
def fetch_stock_data(symbol):
    try:
        stock_data = yf.download(symbol, start='2022-01-01', end='2024-03-25')
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {symbol} from Yahoo Finance API:", e)
        return None

# Example usage
symbol = 'SPY'  # Replace 'SPY' with the symbol of the stock you want to fetch data for
latest_data = fetch_stock_data(symbol)

if latest_data is not None:
    latest_open_price = latest_data.iloc[-1]['Open']
    latest_high_price = latest_data.iloc[-1]['High']
    latest_low_price = latest_data.iloc[-1]['Low']
    latest_volume = latest_data.iloc[-1]['Volume']
    print(f"Latest Open Price: {latest_open_price}")
    print(f"Latest High Price: {latest_high_price}")
    print(f"Latest Low Price: {latest_low_price}")
    print(f"Latest Volume: {latest_volume}")
else:
    print("Data fetch unsuccessful.")
