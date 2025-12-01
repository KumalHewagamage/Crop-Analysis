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
import math

# ---------------- CONFIGURATION ----------------
# File Paths
TRAIN_PATH = 'forecasting_data/synthetic_train_dataset.csv' 
TEST_PATH = 'forecasting_data/synthetic_test_dataset.csv'   
MODEL_SAVE_DIR = 'runs/forecast_model/'

# Training Hyperparameters
WINDOW_SIZE = 10        
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0001  # Lower LR is crucial for Encoder-Decoder stability
TOLERANCE = 15  # Early stopping patience (epochs) based on MAPE

# Feature Configuration
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
        df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'bool':
                df_encoded[col] = df_encoded[col].astype(int)
        
        self.one_hot_cols = [c for c in df_encoded.columns if c not in NUMERICAL_FEATURES and c != TARGET_COLUMN and 'Date' not in c]
        
        df_encoded[NUMERICAL_FEATURES] = self.scaler_X.fit_transform(df_encoded[NUMERICAL_FEATURES])
        df_encoded[[TARGET_COLUMN]] = self.scaler_y.fit_transform(df_encoded[[TARGET_COLUMN]])
        
        return df_encoded

    def transform(self, df):
        df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)
        for col in self.one_hot_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
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
        self.all_features = self.num_cols + self.cat_cols
        self.history_features = self.all_features + [self.target_col]
        
    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        history_slice = self.data.iloc[idx : idx + self.window_size]
        history_x = history_slice[self.history_features].values.astype(np.float32)
        
        current_row = self.data.iloc[idx + self.window_size]
        current_x = current_row[self.all_features].values.astype(np.float32)
        
        label = current_row[self.target_col].astype(np.float32)
        return history_x, current_x, label

# ---------------- IMPROVED ARCHITECTURE: ENCODER-DECODER ----------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CrossAttentionForecastModel(nn.Module):
    def __init__(self, history_input_dim, current_input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(CrossAttentionForecastModel, self).__init__()
        
        # 1. Feature Projections
        # Scale history features up to d_model
        self.history_proj = nn.Linear(history_input_dim, d_model)
        # Scale current context features up to d_model
        self.current_proj = nn.Linear(current_input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # 3. Transformer Encoder (Process History)
        # Captures trends and sequential dependencies in the past data
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Cross-Attention Decoder Block (Process Current Context)
        # Instead of a full TransformerDecoder, we implement the specific Cross-Attention mechanism
        # Query = Current Context, Key/Value = Encoded History
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Feed Forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_model)
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        
        # 5. Final Prediction Head
        self.output_head = nn.Linear(d_model, 1)
        
    def forward(self, history, current):
        # --- PHASE 1: ENCODE HISTORY ---
        # history: [Batch, Window_Size, Feats] -> [Batch, Window_Size, d_model]
        src = self.history_proj(history)
        src = self.pos_encoder(src)
        
        # memory: [Batch, Window_Size, d_model]
        # This is the "Knowledge Base" the model will query
        memory = self.transformer_encoder(src)
        
        # --- PHASE 2: PREPARE QUERY (CURRENT CONTEXT) ---
        # current: [Batch, Feats] -> [Batch, 1, d_model]
        # We unsqueeze to make it a sequence of length 1
        query = self.current_proj(current).unsqueeze(1)
        
        # --- PHASE 3: CROSS-ATTENTION ---
        # "Given today's weather (Query), which parts of history (Memory) are relevant?"
        attn_output, _ = self.multihead_attn(query, memory, memory)
        
        # Residual Connection + Norm
        x = self.norm(query + self.dropout_layer(attn_output))
        
        # Feed Forward + Norm
        ffn_output = self.ffn(x)
        x = self.norm_ffn(x + self.dropout_layer(ffn_output))
        
        # --- PHASE 4: PREDICT ---
        # x shape: [Batch, 1, d_model] -> squeeze -> [Batch, d_model]
        prediction = self.output_head(x.squeeze(1))
        
        return prediction.squeeze()

# ---------------- TRAINING LOOP ----------------
def train():
    logger.info("Loading Data...")
    try:
        df_train = pd.read_csv(TRAIN_PATH)
        df_test = pd.read_csv(TEST_PATH)
    except FileNotFoundError:
        logger.error("Data files not found! Please ensure CSVs exist.")
        return

    # Preprocess
    preprocessor = PineapplePreprocessor()
    train_processed = preprocessor.fit_transform(df_train)
    test_processed = preprocessor.transform(df_test)
    joblib.dump(preprocessor, f"{MODEL_SAVE_DIR}/preprocessor.pkl")
    
    # Datasets
    train_dataset = WindowedDataset(train_processed, WINDOW_SIZE, NUMERICAL_FEATURES, preprocessor.one_hot_cols, TARGET_COLUMN)
    test_dataset = WindowedDataset(test_processed, WINDOW_SIZE, NUMERICAL_FEATURES, preprocessor.one_hot_cols, TARGET_COLUMN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    sample_hist, sample_curr, _ = train_dataset[0]
    
    # Using the NEW CrossAttentionForecastModel
    model = CrossAttentionForecastModel(
        history_input_dim=sample_hist.shape[1], 
        current_input_dim=sample_curr.shape[0],
        d_model=128,  # Increased model capacity
        nhead=4,
        num_layers=3  # Deeper encoder
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_loss = float('inf')
    # Early stopping (use MAPE as main metric)
    best_mape = float('inf')
    best_mape_epoch = -1
    patience = 0
    
    logger.info("Starting Training with Encoder-Decoder Cross-Attention...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for hist_batch, curr_batch, label_batch in loop:
            optimizer.zero_grad()
            outputs = model(hist_batch, curr_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Test Step
        model.eval()
        test_loss = 0
        total_abs_diff = 0
        total_rel_diff = 0
        total_samples = 0
        
        with torch.no_grad():
            for hist_batch, curr_batch, label_batch in test_loader:
                outputs = model(hist_batch, curr_batch)
                loss = criterion(outputs, label_batch)
                test_loss += loss.item()
                
                preds = outputs.cpu().numpy().reshape(-1, 1)
                actuals = label_batch.cpu().numpy().reshape(-1, 1)
                preds_kg = preprocessor.scaler_y.inverse_transform(preds)
                actuals_kg = preprocessor.scaler_y.inverse_transform(actuals)
                
                abs_diff = np.abs(preds_kg - actuals_kg)
                rel_diff = abs_diff / (actuals_kg + 1e-6)
                
                total_abs_diff += np.sum(abs_diff)
                total_rel_diff += np.sum(rel_diff)
                total_samples += len(abs_diff)
        
        avg_test_loss = test_loss / len(test_loader)
        avg_mae = total_abs_diff / total_samples
        avg_mape = (total_rel_diff / total_samples) * 100
        
        scheduler.step(avg_test_loss)
        
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f} | MAE: {avg_mae:.2f} kg | MAPE: {avg_mape:.2f}%")
        
        torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/last.pt")

        # Use MAPE for deciding best model and early stopping
        if avg_mape < best_mape:
            best_mape = avg_mape
            best_mape_epoch = epoch + 1
            patience = 0
            torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/best.pt")
            logger.info(f"--> New Best Model Saved (MAPE: {best_mape:.4f}%) at epoch {best_mape_epoch}")
        else:
            patience += 1

        if patience >= TOLERANCE:
            logger.info(f"Early stopping triggered. No MAPE improvement in last {TOLERANCE} epochs.")
            break

    logger.info("Training Complete.")
    
    try:
        summary = {
            'best_test_loss': float(best_loss),
            'final_train_loss': float(avg_train_loss),
            'final_test_loss': float(avg_test_loss),
            'final_mae_kg': float(avg_mae),
            'final_mape_pct': float(avg_mape)
        }
        with open(os.path.join(MODEL_SAVE_DIR, 'training_summary.json'), 'w') as sf:
            json.dump(summary, sf, indent=2)
    except:
        pass

if __name__ == "__main__":
    train()