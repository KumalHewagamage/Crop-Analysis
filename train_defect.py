from ultralytics import YOLO

def train_model():
    model = YOLO('models/pretrained/yolo11m.pt') 
    results = model.train(
        data='config/defect.yaml', 
        epochs=50, 
        imgsz=640,
        batch=16,
        name='pineapple_defect_model'
    )

if __name__ == '__main__':
    train_model()