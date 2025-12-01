import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
from tqdm import tqdm
import json

# ---------------- CONFIGURATION ----------------
# File Paths
TRAIN_PATH = 'data/forecasting_data/synthetic_train_dataset.csv' # Rename to your actual file
TEST_PATH = 'data/forecasting_data/synthetic_test_dataset.csv'   # Rename to your actual file
MODEL_SAVE_DIR = 'outputs/forecast_model/'

# Training Hyperparameters
WINDOW_SIZE = 10        # How many past records to look at
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001

# Feature Configuration
# These are the raw columns from your CSV to use as inputs
NUMERICAL_FEATURES = ['Avg_Temp_C', 'Avg_Rain_mm', 'Avg_Humidity']
CATEGORICAL_FEATURES = ['Location', 'Soil_Type']
TARGET_COLUMN = 'Yield_kg'

# ---------------- LOGGING SETUP ----------------
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{MODEL_SAVE_DIR}/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ---------------- DATA PREPROCESSING ----------------
class PineapplePreprocessor:
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.one_hot_cols = []
        
    def fit_transform(self, df):
        # 1. One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)
        # Convert bool to int (True/False -> 1/0)
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'bool':
                df_encoded[col] = df_encoded[col].astype(int)
                
        self.one_hot_cols = [c for c in df_encoded.columns if c not in NUMERICAL_FEATURES and c != TARGET_COLUMN and 'Date' not in c]
        
        # 2. Normalize Numerical Features
        df_encoded[NUMERICAL_FEATURES] = self.scaler_X.fit_transform(df_encoded[NUMERICAL_FEATURES])
        
        # 3. Normalize Target
        df_encoded[[TARGET_COLUMN]] = self.scaler_y.fit_transform(df_encoded[[TARGET_COLUMN]])
        
        return df_encoded

    def transform(self, df):
        # Used for Test data - applies same scaling/encoding as Train
        df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)
        
        # Align columns (add missing cols with 0, remove extra cols)
        for col in self.one_hot_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Ensure column order matches
        # We need all feature columns + target
        all_cols = NUMERICAL_FEATURES + self.one_hot_cols + [TARGET_COLUMN]
        # Note: In production prediction, TARGET_COLUMN might not exist, but for Test set it does.
        
        # Convert bools
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'bool':
                df_encoded[col] = df_encoded[col].astype(int)

        df_encoded[NUMERICAL_FEATURES] = self.scaler_X.transform(df_encoded[NUMERICAL_FEATURES])
        if TARGET_COLUMN in df_encoded.columns:
            df_encoded[[TARGET_COLUMN]] = self.scaler_y.transform(df_encoded[[TARGET_COLUMN]])
            
        return df_encoded

# ---------------- DATASET CLASS ----------------
class WindowedDataset(Dataset):
    def __init__(self, data, window_size, numerical_cols, cat_cols, target_col):
        self.data = data
        self.window_size = window_size
        self.num_cols = numerical_cols
        self.cat_cols = cat_cols
        self.target_col = target_col
        
        # Pre-calculate feature lists
        self.all_features = self.num_cols + self.cat_cols
        # The history includes yield!
        self.history_features = self.all_features + [self.target_col]
        
    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # 1. Historical Window (t-10 to t-1)
        # Includes Weather + Yield
        history_slice = self.data.iloc[idx : idx + self.window_size]
        history_x = history_slice[self.history_features].values.astype(np.float32)
        
        # 2. Current Day Features (t)
        # Includes Weather ONLY (No Yield)
        current_row = self.data.iloc[idx + self.window_size]
        current_x = current_row[self.all_features].values.astype(np.float32)
        
        # 3. Target Label (t)
        label = current_row[self.target_col].astype(np.float32)
        
        return history_x, current_x, label

# ---------------- MODEL ARCHITECTURE ----------------
class HybridForecastModel(nn.Module):
    def __init__(self, history_input_dim, current_input_dim, hidden_dim=64):
        super(HybridForecastModel, self).__init__()
        
        # Branch 1: LSTM for History
        self.lstm = nn.LSTM(input_size=history_input_dim, 
                            hidden_size=hidden_dim, 
                            batch_first=True,
                            num_layers=1)
        
        # Branch 2: Dense for Current Features
        self.current_feature_encoder = nn.Linear(current_input_dim, 32)
        
        # Fusion Layer
        # LSTM Output (hidden_dim) + Current Encoded (32)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # Output: Predicted Yield
        )
        
    def forward(self, history, current):
        # 1. Process History through LSTM
        lstm_out, _ = self.lstm(history) 
        # We only care about the output of the last time step
        last_hidden = lstm_out[:, -1, :] 
        
        # 2. Process Current Features
        current_encoded = torch.relu(self.current_feature_encoder(current))
        
        # 3. Concatenate
        combined = torch.cat((last_hidden, current_encoded), dim=1)
        
        # 4. Final Prediction
        prediction = self.fusion(combined)
        return prediction.squeeze()

# ---------------- TRAINING LOOP ----------------
def train():
    # 1. Load Data
    logger.info("Loading Data...")
    try:
        df_train = pd.read_csv(TRAIN_PATH)
        df_test = pd.read_csv(TEST_PATH)
    except FileNotFoundError:
        logger.error("Data files not found! Please ensure CSVs exist.")
        return

    # 2. Preprocess
    preprocessor = PineapplePreprocessor()
    train_processed = preprocessor.fit_transform(df_train)
    test_processed = preprocessor.transform(df_test)
    
    # Save preprocessor for later inference
    joblib.dump(preprocessor, f"{MODEL_SAVE_DIR}/preprocessor.pkl")
    
    # 3. Create Datasets
    train_dataset = WindowedDataset(
        train_processed, WINDOW_SIZE, NUMERICAL_FEATURES, preprocessor.one_hot_cols, TARGET_COLUMN
    )
    test_dataset = WindowedDataset(
        test_processed, WINDOW_SIZE, NUMERICAL_FEATURES, preprocessor.one_hot_cols, TARGET_COLUMN
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Initialize Model
    # Calculate input dimensions automatically
    sample_hist, sample_curr, _ = train_dataset[0]
    
    model = HybridForecastModel(
        history_input_dim=sample_hist.shape[1], # Cols in history (features + yield)
        current_input_dim=sample_curr.shape[0]  # Cols in current (features only)
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    
    logger.info("Starting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # Create progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for hist_batch, curr_batch, label_batch in loop:
            optimizer.zero_grad()
            
            # Forward Pass
            outputs = model(hist_batch, curr_batch)
            loss = criterion(outputs, label_batch)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- TEST STEP (After Each Epoch) ---
        model.eval()
        test_loss = 0
        
        # Variables for real-world metrics
        total_abs_diff = 0
        total_rel_diff = 0
        total_samples = 0
        
        with torch.no_grad():
            for hist_batch, curr_batch, label_batch in test_loader:
                outputs = model(hist_batch, curr_batch)
                loss = criterion(outputs, label_batch)
                test_loss += loss.item()
                
                # --- Calculate Real-world Metrics ---
                # 1. Convert tensors to numpy
                preds = outputs.cpu().numpy().reshape(-1, 1)
                actuals = label_batch.cpu().numpy().reshape(-1, 1)
                
                # 2. Inverse Transform (Back to original kg)
                preds_kg = preprocessor.scaler_y.inverse_transform(preds)
                actuals_kg = preprocessor.scaler_y.inverse_transform(actuals)
                
                # 3. Calculate Differences
                abs_diff = np.abs(preds_kg - actuals_kg)
                # Avoid division by zero
                rel_diff = abs_diff / (actuals_kg + 1e-6)
                
                total_abs_diff += np.sum(abs_diff)
                total_rel_diff += np.sum(rel_diff)
                total_samples += len(abs_diff)
        
        avg_test_loss = test_loss / len(test_loader)
        avg_mae = total_abs_diff / total_samples
        avg_mape = (total_rel_diff / total_samples) * 100
        
        # Logging with new metrics
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f} | MAE: {avg_mae:.2f} kg | MAPE: {avg_mape:.2f}%")
        
        # Save checkpoints
        # Save Last
        torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/last.pt")
        
        # Save Best
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/best.pt")
            logger.info(f"--> New Best Model Saved (Loss: {best_loss:.6f})")

    logger.info("Training Complete.")

    # --- Training summary: log and save a concise summary JSON ---
    try:
        summary = {
            'best_test_loss': float(best_loss),
            'final_train_loss': float(avg_train_loss),
            'final_test_loss': float(avg_test_loss),
            'final_mae_kg': float(avg_mae),
            'final_mape_pct': float(avg_mape)
        }
        logger.info(
            f"Training Summary -- Best Test Loss: {summary['best_test_loss']:.6f} | "
            f"Final Train Loss: {summary['final_train_loss']:.6f} | "
            f"Final Test Loss: {summary['final_test_loss']:.6f} | "
            f"MAE: {summary['final_mae_kg']:.2f} kg | MAPE: {summary['final_mape_pct']:.2f}%"
        )

        # Save summary to disk for later inspection
        summary_path = os.path.join(MODEL_SAVE_DIR, 'training_summary.json')
        with open(summary_path, 'w') as sf:
            json.dump(summary, sf, indent=2)
        logger.info(f"Training summary saved to: {summary_path}")
    except NameError:
        logger.warning('Training summary not available (training may not have completed).')

if __name__ == "__main__":
    train()