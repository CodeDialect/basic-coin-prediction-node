import os
import pickle
from zipfile import ZipFile
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from updater import download_binance_monthly_data, download_binance_daily_data
from config import data_base_path, model_file_path

binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
training_price_data_path = os.path.join(data_base_path, "eth_price_data.csv")

class ETHDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def download_data():
    cm_or_um = "um"
    symbols = ["ETHUSDT"]
    intervals = ["1d"]
    years = ["2020", "2021", "2022", "2023", "2024"]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    download_path = binance_data_path
    download_binance_monthly_data(
        cm_or_um, symbols, intervals, years, months, download_path
    )
    print(f"Downloaded monthly data to {download_path}.")
    current_datetime = datetime.now()
    current_year = current_datetime.year
    current_month = current_datetime.month
    download_binance_daily_data(
        cm_or_um, symbols, intervals, current_year, current_month, download_path
    )
    print(f"Downloaded daily data to {download_path}.")

def format_data():
    files = sorted([x for x in os.listdir(binance_data_path)])

    # No files to process
    if len(files) == 0:
        return

    price_df = pd.DataFrame()
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
            "start_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "end_time",
            "volume_usd",
            "n_trades",
            "taker_volume",
            "taker_volume_usd",
        ]
        df.index = [pd.Timestamp(x + 1, unit="ms") for x in df["end_time"]]
        df.index.name = "date"
        price_df = pd.concat([price_df, df])

    price_df.sort_index().to_csv(training_price_data_path)

def train_model():
    # Load the eth price data
    price_data = pd.read_csv(training_price_data_path)
    df = pd.DataFrame(price_data)

    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Use all available data instead of just the last 6 months
    # six_months_ago = datetime.now() - timedelta(days=180)
    # df = df[df['date'] > six_months_ago]

    # Feature engineering
    df['price'] = df[['open', 'close', 'high', 'low']].mean(axis=1)
    df['log_price'] = np.log(df['price'])  # Use log price for better scaling
    df['volume'] = df['volume']
    df['log_volume'] = np.log(df['volume'] + 1)  # Use log volume for better scaling
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['price_change'] = df['price'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    # Create lag features
    for lag in [1, 3, 7, 14, 30]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

    # Add rolling mean features
    for window in [7, 14, 30]:
        df[f'price_rolling_mean_{window}'] = df['price'].rolling(window=window).mean()
        df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window=window).mean()

    # Add exponential moving averages
    for span in [7, 14, 30]:
        df[f'price_ema_{span}'] = df['price'].ewm(span=span, adjust=False).mean()
        df[f'volume_ema_{span}'] = df['volume'].ewm(span=span, adjust=False).mean()

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Prepare features and target
    features = ['day_of_week', 'month', 'year', 'log_volume', 'price_change', 'volume_change'] + \
               [f'price_rolling_mean_{window}' for window in [7, 14, 30]] + \
               [f'volume_rolling_mean_{window}' for window in [7, 14, 30]] + \
               [f'price_ema_{span}' for span in [7, 14, 30]] + \
               [f'volume_ema_{span}' for span in [7, 14, 30]] + \
               [f'price_lag_{lag}' for lag in [1, 3, 7, 14, 30]] + \
               [f'volume_lag_{lag}' for lag in [1, 3, 7, 14, 30]]
    
    X = df[features].values
    y = df['price'].values

    # Scale the features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Scale the target variable
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Reshape X_scaled to include time steps (we'll use a single time step)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Create dataset and dataloader
    dataset = ETHDataset(X_scaled, y_scaled)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize the model
    input_dim = len(features)
    hidden_dim = 128
    num_layers = 3
    output_dim = 1
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

    # Training loop
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 200

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(torch.tensor(X_scaled, dtype=torch.float32)).squeeze().numpy()
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # Print model results
    print("\nLSTM Model Results:")
    print(f"R-squared score: {r2_score(y, predictions):.6f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y, predictions):.2f}")

    # Get current features for prediction
    current_date = datetime.now()
    current_features = np.array([
        current_date.weekday(),
        current_date.month,
        current_date.year,
        np.log(df['volume'].iloc[-1] + 1),
        df['price'].iloc[-1] / df['price'].iloc[-2] - 1,
        df['volume'].iloc[-1] / df['volume'].iloc[-2] - 1,
    ] + [df['price'].rolling(window=window).mean().iloc[-1] for window in [7, 14, 30]] +
      [df['volume'].rolling(window=window).mean().iloc[-1] for window in [7, 14, 30]] +
      [df['price'].ewm(span=span, adjust=False).mean().iloc[-1] for span in [7, 14, 30]] +
      [df['volume'].ewm(span=span, adjust=False).mean().iloc[-1] for span in [7, 14, 30]] +
      [df['price'].iloc[-lag] for lag in [1, 3, 7, 14, 30]] +
      [df['volume'].iloc[-lag] for lag in [1, 3, 7, 14, 30]])

    current_features_scaled = scaler_X.transform(current_features.reshape(1, -1))
    current_features_scaled = current_features_scaled.reshape((1, 1, -1))
    
    with torch.no_grad():
        current_prediction_scaled = model(torch.tensor(current_features_scaled, dtype=torch.float32)).item()
        current_prediction = scaler_y.inverse_transform([[current_prediction_scaled]])[0][0]

    print(f"Predicted ETH price for current time: ${current_prediction:.2f}")

    # Print a few random predictions
    print("\nRandom predictions from the dataset:")
    for _ in range(5):
        random_index = np.random.randint(0, len(X_scaled))
        random_features = X_scaled[random_index:random_index+1]
        with torch.no_grad():
            random_prediction_scaled = model(torch.tensor(random_features, dtype=torch.float32)).item()
            random_prediction = scaler_y.inverse_transform([[random_prediction_scaled]])[0][0]
        actual_price = y[random_index]
        print(f"Random prediction: ${random_prediction:.2f}, Actual price: ${actual_price:.2f}")

    # Print the last few rows of the dataframe
    print("\nLast 5 rows of the dataframe:")
    print(df[['date', 'price', 'volume']].tail())

    print(f"\nMin price in dataset: ${df['price'].min():.2f}")
    print(f"Max price in dataset: ${df['price'].max():.2f}")

    # Save the trained model and scaler to files
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    with open(model_file_path, 'wb') as f:
        pickle.dump((model, scaler_X, scaler_y, features), f)

    print(f"\nTrained LSTM model, scalers, and feature list saved to {model_file_path}")
