import torch
import pandas as pd
import numpy as np
import joblib
import os

from train_forecast import (
    CrossAttentionForecastModel, 
    PineapplePreprocessor, 
    PositionalEncoding,
    NUMERICAL_FEATURES, 
    CATEGORICAL_FEATURES, 
    TARGET_COLUMN, 
    WINDOW_SIZE
)

# ---------------- CONFIGURATION ----------------

MODEL_WEIGHTS_PATH = 'models/pineapple_forecast/best.pt'
PREPROCESSOR_PATH = 'models/pineapple_forecast/preprocessor.pkl' 

# Model Hyperparameters (MUST match training config)
D_MODEL = 128
N_HEAD = 4
NUM_LAYERS = 3

class PineappleYieldPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading resources on: {self.device}")

        # 1. Load Preprocessor
        if not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}. Train model first.")
        try:
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        except ModuleNotFoundError as e:
            if 'numpy._core' in str(e):
                raise RuntimeError(
                    "NumPy version mismatch. The preprocessor was saved with a different NumPy version. "
                    "Please retrain the model using 'python train_forecast.py' or downgrade NumPy to match the training environment."
                ) from e
            else:
                raise
        
        # 2. Initialize Model Architecture
        # Count one-hot encoded columns + numerical columns
        num_encoded_features = len(self.preprocessor.one_hot_cols) + len(NUMERICAL_FEATURES)
        
        # History has features + yield
        history_dim = num_encoded_features + 1 
        # Current context has features only
        current_dim = num_encoded_features
        
        self.model = CrossAttentionForecastModel(
            history_input_dim=history_dim,
            current_input_dim=current_dim,
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_layers=NUM_LAYERS
        )

        # 3. Load Weights
        if not os.path.exists(MODEL_WEIGHTS_PATH):
             raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS_PATH}.")
             
        self.model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def predict(self, history_df, current_conditions):
        """
        history_df: DataFrame with last 10 records. MUST contain Weather + Yield_kg
        current_conditions: Dictionary with Today's Weather + Location
        """
        # Validate Input
        if len(history_df) != WINDOW_SIZE:
            print(f"Warning: History should be {WINDOW_SIZE} days. Got {len(history_df)}.")
            history_df = history_df.tail(WINDOW_SIZE)

        # 1. Prepare Data for Preprocessor

        # Convert dict to DataFrame
        current_df = pd.DataFrame([current_conditions])
        # Add dummy yield to current row
        current_df[TARGET_COLUMN] = 0 
        
        # Combine
        full_df = pd.concat([history_df, current_df], ignore_index=True)
        
        # 2. Transform (Scale & One-Hot Encode)
        processed_df = self.preprocessor.transform(full_df)
        
        # 3. Convert to Tensors
        # History: First 10 rows, All columns (Features + Normalized Yield)
        history_data = processed_df.iloc[:-1].values.astype(np.float32)
        history_tensor = torch.tensor(history_data).unsqueeze(0).to(self.device) # [1, 10, Features]
        
        # Current: Last row, Features ONLY (Drop the dummy yield at the end)
        current_data = processed_df.iloc[-1].values.astype(np.float32)
        # Ideally check column index, but our preprocessor puts target last.
        current_inputs = current_data[:-1] 
        current_tensor = torch.tensor(current_inputs).unsqueeze(0).to(self.device) # [1, Features]

        # 4. Inference
        with torch.no_grad():
            predicted_normalized = self.model(history_tensor, current_tensor).item()
            
        # 5. Inverse Transform (Get kg back)
        # We use the y_scaler to convert 0.5 -> 15000 kg
        dummy_array = np.zeros((1, 1))
        dummy_array[0, 0] = predicted_normalized
        predicted_kg = self.preprocessor.scaler_y.inverse_transform(dummy_array)[0, 0]
        
        return predicted_kg

# ==========================================
#              RUN DUMMY TEST
# ==========================================
if __name__ == "__main__":
    
    # 1. Create Dummy History (Last 10 harvests/days)
    # Simulating a pattern where it was raining decently
    history_data = {
        'Avg_Temp_C': [26, 27, 25, 26, 28, 27, 26, 25, 26, 27],
        'Avg_Rain_mm': [100, 110, 90, 120, 80, 95, 105, 115, 100, 90],
        'Avg_Humidity': [70, 72, 68, 75, 65, 71, 73, 74, 70, 69],
        'Location': ['Block_A_North'] * 10,
        'Soil_Type': ['Sandy Loam'] * 10,
        'Yield_kg': [14500, 14800, 14200, 15100, 13900, 14600, 14900, 15050, 14700, 14400]
    }
    history_df = pd.DataFrame(history_data)

    # 2. Create Target Conditions (The future we want to predict)
    # Scenario: Slightly hotter and drier than average
    current_conditions = {
        'Avg_Temp_C': 29.5,   # Hotter
        'Avg_Rain_mm': 60.0,  # Less Rain
        'Avg_Humidity': 60.0, # Drier
        'Location': 'Block_A_North',
        'Soil_Type': 'Sandy Loam'
    }

    print("\n--- PINEAPPLE YIELD FORECAST SYSTEM ---")
    
    predictor = PineappleYieldPredictor()
    
    print("\nInput History (Last 3 days shown):")
    print(history_df.tail(3))
    print("\nTarget Conditions:")
    print(current_conditions)
    
    prediction = predictor.predict(history_df, current_conditions)
    
    print("\n" + "="*40)
    print(f"PREDICTED YIELD: {prediction:,.2f} kg")
    print("="*40 + "\n")
        
