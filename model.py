import json
import os
import pickle
from zipfile import ZipFile
import pandas as pd
import requests
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor  # Import kNN
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKEN, MODEL, CG_API_KEY, DATA_PROVIDER
from sklearn.metrics import mean_absolute_error

binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")

def create_lagged_features(df, asset, num_lags=10):
    """
    Create lagged features for a given asset.
    """
    lagged_data = pd.DataFrame()
    for col in ["open", "high", "low", "close"]:
        for lag in range(1, num_lags + 1):
            lagged_data[f"{col}_{asset}_lag{lag}"] = df[col].shift(lag)

    return lagged_data

def train_knn_model():
    """
    Train kNN model using the structured dataset.
    """
    print("📊 Training kNN model using kNN_eth.py approach...")

    data_path = "/app/data/ETHUSDT_1h_spot_forecast_training_new.csv"
    model_save_path = "/app/data/model.pkl"  # Save as model.pkl to match inference

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"🚨 Dataset {data_path} not found! Run update first.")

    # Load and preprocess data
    df = pd.read_csv(data_path)
    df = df.ffill().bfill()  # Fill missing values

    # Define features and target variable
    X = df.drop(columns=['target_ETHUSDT'])
    y = df['target_ETHUSDT']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Define kNN model and hyperparameters
    knn = KNeighborsRegressor()
    param_grid = {
        "n_neighbors": [500, 750, 1000],  # Increase k further
        "weights": ["uniform", "distance"],  # Try both weighting strategies
        "metric": ["minkowski", "manhattan"]  # Check performance of Manhattan distance
    }

    # Optimize model using Mean Absolute Error (MAE)
    grid_search = GridSearchCV(
        knn,
        param_grid,
        cv=tscv,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        n_jobs=-1,  # Use all CPU cores
        verbose=2  # Show detailed logs
    )

    # Train model
    grid_search.fit(X_scaled, y)

    # Get the best model
    best_knn = grid_search.best_estimator_

    # Print best hyperparameters
    print(f"✅ Best k: {best_knn.n_neighbors}, Metric: {best_knn.metric}, Weighting: {best_knn.weights}")

    # Save the trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(best_knn, f)

    print(f"✅ Trained kNN model saved to {model_save_path}")

def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data_coingecko(token, training_days):
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data(token, training_days, region, data_provider):
    if data_provider == "coingecko":
        return download_data_coingecko(token, int(training_days))
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")

def fetch_binance_ohlc(symbol, interval, start_time, end_time):
    """
    Fetch historical OHLC data from Binance.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000
    }

    all_data = []
    while True:
        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        all_data.extend(data)
        params["startTime"] = data[-1][0] + 1  # Increment to fetch the next batch

        if len(data) < 1000:
            break  # Exit if there are no more data points

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    df.rename(columns={'open_time': 'timestamp'}, inplace=True)

    return df

def format_data(files, data_provider, force_update=False):
    if not files and not force_update:
        print("Already up to date")
        return

    if force_update and os.path.exists(training_price_data_path):
        os.remove(training_price_data_path)  # Clear old corrupted data

    if data_provider == "binance":
        files = sorted([x for x in os.listdir(binance_data_path) if x.startswith(f"{TOKEN}USDT")])
    elif data_provider == "coingecko":
        files = sorted([x for x in os.listdir(coingecko_data_path) if x.endswith(".json")])

    if len(files) == 0:
        print("No files to process")
        return

    price_df = pd.DataFrame()
    
    if data_provider == "binance":
        for file in files:
            zip_file_path = os.path.join(binance_data_path, file)
            if not zip_file_path.endswith(".zip"):
                continue

            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]

            df.columns = [
                "start_time", "open", "high", "low", "close", "volume", "end_time",
                "volume_usd", "n_trades", "taker_volume", "taker_volume_usd"
            ]

            # Filter timestamps (max: 2100-01-01 in microseconds)
            df = df[df["end_time"] < 4102444800000000]  # Adjusted for microseconds

            # Convert timestamps from microseconds to seconds
            df["timestamp"] = pd.to_datetime(df["start_time"] // 1_000_000, unit="s")

            print(f"Filtered Binance data sample: {df[['timestamp', 'open', 'close']].head()}")

            # Set index for sorting
            df.set_index("timestamp", inplace=True)
            price_df = pd.concat([price_df, df])

        if not price_df.empty:
            price_df.sort_index().to_csv(training_price_data_path)
            print(f"✅ Saved Binance data to {training_price_data_path}")
        else:
            print("❌ No valid Binance data to save")

    elif data_provider == "coingecko":
        for file in files:
            with open(os.path.join(coingecko_data_path, file), "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df.columns = ["timestamp", "open", "high", "low", "close"]

                # Filter timestamps (max: 2100-01-01 in milliseconds)
                df = df[df["timestamp"] < 4102444800000]  # Milliseconds for CoinGecko
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                print(f"Filtered CoinGecko data sample: {df[['timestamp', 'open', 'close']].head()}")

                df.set_index("timestamp", inplace=True)
                price_df = pd.concat([price_df, df])

        if not price_df.empty:
            price_df.sort_index().to_csv(training_price_data_path)
            print(f"✅ Saved CoinGecko data to {training_price_data_path}")
        else:
            print("❌ No valid CoinGecko data to save")

    """
    Fetch ETHUSDT & BTCUSDT data, generate the forecast dataset, and prepare for model training.
    """
    if not force_update and os.path.exists("/app/data/ETHUSDT_1h_spot_forecast_training_new.csv"):
        print("✅ Data is already up to date.")
        return

# ✅ If using kNN and Binance, generate the dataset
if MODEL == "kNN" and DATA_PROVIDER == "binance":
    print("⏳ Fetching ETHUSDT & BTCUSDT data from Binance...")

    # Define the start time (April 1, 2023, 10:00 AM) to current time
    start_time_str = '2023-04-01 10:00:00'
    end_time_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    start_time_ms = int(pd.Timestamp(start_time_str).timestamp() * 1000)
    end_time_ms = int(pd.Timestamp(end_time_str).timestamp() * 1000)

    # Fetch fresh data
    df_eth = fetch_binance_ohlc('ETHUSDT', '1h', start_time_ms, end_time_ms)
    df_btc = fetch_binance_ohlc('BTCUSDT', '1h', start_time_ms, end_time_ms)

    df = df_eth.merge(df_btc, on="timestamp", suffixes=("_ETHUSDT", "_BTCUSDT"))

    # Generate lagged features
    def create_lagged_features(df, asset, num_lags=10):
        lagged_data = pd.DataFrame()
        for col in ["open", "high", "low", "close"]:
            for lag in range(1, num_lags + 1):
                lagged_data[f"{col}_{asset}_lag{lag}"] = df[f"{col}_{asset}"].shift(lag)
        return lagged_data

    eth_lags = create_lagged_features(df, "ETHUSDT")
    btc_lags = create_lagged_features(df, "BTCUSDT")

    df_final = pd.concat([eth_lags, btc_lags], axis=1)
    df_final["hour_of_day"] = df["timestamp"].dt.hour
    df_final["target_ETHUSDT"] = df["close_ETHUSDT"].shift(-1)
    df_final = df_final[:-1]  # Drop last row to prevent NaN targets

    df_final.dropna(inplace=True)

    # Save the dataset
    output_path = "/app/data/ETHUSDT_1h_spot_forecast_training_new.csv"
    df_final.to_csv(output_path, index=False)
    print(f"✅ Saved dataset: {output_path}")

def load_frame(frame, timeframe):
    """
    Loads and processes the OHLCV data, ensuring the presence of a valid timestamp column.
    """
    print("📊 Loading data...")

    df = frame.copy()  # Copy to avoid modifying original data

    # Ensure a valid timestamp column exists
    if "timestamp" not in df.columns:
        if "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"])  # Convert 'date' column to timestamp
        elif "start_time" in df.columns:
           df["timestamp"] = pd.to_datetime(df["start_time"] // 1_000_000, unit="s")  # Convert microseconds to seconds
        elif "end_time" in df.columns:
           df["timestamp"] = pd.to_datetime(df["end_time"] // 1_000_000, unit="s")  # Convert microseconds to seconds
        else:
           raise ValueError(f"🚨 No valid timestamp column found! Available columns: {df.columns.tolist()}")

    # Keep OHLC and timestamp columns
    df = df.loc[:, ["timestamp", "open", "high", "low", "close"]].dropna()

    # Set timestamp as index
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    return df

def train_model(timeframe):
    """
    General training function that supports multiple models, including kNN.
    """
    if MODEL == "kNN":
        print("📊 Training kNN model using kNN_eth.py approach...")
        train_knn_model()  # Use kNN training method
        return

    # Load the price data
    price_data = pd.read_csv(training_price_data_path)
    print("Raw CSV data sample:", price_data.head())  # Debug output
    df = load_frame(price_data, timeframe)

    print("Processed data tail:", df.tail())

    y_train = df['close'].shift(-1).dropna().values
    X_train = df[:-1]

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")

    # Check for empty data
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        raise ValueError("No valid training data available after processing. Check data source or timeframe.")

    # Define the model
    if MODEL == "LinearRegression":
        model = LinearRegression()
    elif MODEL == "SVR":
        model = SVR()
    elif MODEL == "KernelRidge":
        model = KernelRidge()
    elif MODEL == "BayesianRidge":
        model = BayesianRidge()
    else:
        raise ValueError("Unsupported model")
    
    # Train the model
    model.fit(X_train, y_train)

    # Create the model's parent directory if it doesn't exist
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    # Save the trained model to a file
    joblib.dump(model, model_file_path)  # Save all models using joblib

    print(f"Trained model saved to {model_file_path}")

def get_inference(token, timeframe, region, data_provider):
    """
    Load the trained kNN model and generate price predictions.
    """
    print("📊 Loading kNN model for inference...")

    # Check if the model exists
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"🚨 Model file {model_file_path} not found! Run update first.")

    # Load the trained model
    with open(model_file_path, "rb") as f:
        model = pickle.load(f)

    print("✅ Model loaded successfully.")

    # Load the latest price data
    if data_provider == "coingecko":
        df = load_frame(download_coingecko_current_day_data(token, CG_API_KEY), timeframe)

        # Ensure latest data is available
        if df.empty:
            raise ValueError("🚨 No data available from CoinGecko for inference!")

        print(f"✅ Latest CoinGecko data loaded: {df.shape}")

        # Generate lagged features for ETHUSDT & BTCUSDT
        eth_lags = create_lagged_features(df, "ETHUSDT", num_lags=10)
        btc_lags = create_lagged_features(df, "BTCUSDT", num_lags=10)

        # Combine ETH and BTC features
        X_new = pd.concat([eth_lags, btc_lags], axis=1)

        # Ensure hour_of_day is included
        X_new["hour_of_day"] = df.index[-1].hour  # Extract the hour from the latest timestamp

    else:
        # **For Binance, load the latest processed dataset instead of using `download_binance_current_day_data()`**
        data_path = "/app/data/ETHUSDT_1h_spot_forecast_training_new.csv"

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"🚨 Dataset {data_path} not found! Run update first.")

        df = pd.read_csv(data_path)

        # Drop the target column to retain only the features
        X_new = df.drop(columns=['target_ETHUSDT'], errors='ignore')

        # Keep only the latest row for inference
        X_new = X_new.iloc[-1:].dropna()

        print(f"✅ Latest Binance data loaded: {X_new.shape}")

    # Ensure input shape matches (1, 81)
    expected_features = 81  # 40 ETH lags + 40 BTC lags + 1 hour_of_day
    if X_new.shape[1] != expected_features:
        raise ValueError(f"🚨 Expected input shape (1, {expected_features}), but got {X_new.shape}")

    # **Standardize input using the same scaler from training**
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_new)

    # Make prediction
    prediction = model.predict(X_scaled)

    print(f"✅ Prediction: {prediction[0]}")
    return prediction[0]
