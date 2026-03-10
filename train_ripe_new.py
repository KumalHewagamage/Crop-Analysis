from ultralytics import YOLO

def train_model():
    model = YOLO('yolo12m.pt') 
    results = model.train(
        # Dataset and basic settings
        data='config/ripe.yaml', 
        epochs=100, 
        imgsz=640,
        batch=16,
        name='pineapple_ripe_model_yolo12m',
        
        # # Optimizer
        # optimizer='AdamW',
        # lr0=0.001,
        
        # Data Augmentation (Tuned for Conveyor)
        hsv_h=0.015,           # Slight color shifts
        hsv_s=0.7,             # Helps with lighting changes on the belt
        hsv_v=0.4,             # Helps if shadows vary
        # degrees=180.0,         # Assuming pineapples can spin on the belt
        fliplr=0.5,
        
        # Training settings
        patience=20,  # Early stopping patience (epochs without improvement)

    
        
        # Hardware
        device='cuda',  # Auto-detect device (use GPU if available)
        workers=24,  # Number of worker threads for data loading
        
        # Other
        verbose=True,  # Verbose output
        seed=0,  # Random seed for reproducibility
    )

if __name__ == '__main__':
    train_model()