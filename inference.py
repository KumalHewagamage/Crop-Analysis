
from ultralytics import YOLO
import os

def extract_detections(result):
    path = result.path
    filename = os.path.basename(path)
    img_shape = result.orig_shape

    print(f"Image: {filename}")
    print(f"Path:  {path}")
    print(f"Size:  {img_shape[1]}x{img_shape[0]} (WxH)")
    print("-" * 30)

    boxes = result.boxes

    if len(boxes) == 0:
        print("No defects detected.")
    else:
        for i, box in enumerate(boxes):
            
            cls_id = int(box.cls[0]) 
            class_name = result.names[cls_id]
            
            conf = float(box.conf[0])
            
            # CHANGED HERE: Access .xywh instead of .xyxy
            coords = box.xywh[0].tolist() 
            x_center, y_center, width, height = [round(x, 2) for x in coords]

            print(f"Detection #{i+1}:")
            print(f"  • Type:       {class_name.upper()}")
            print(f"  • Confidence: {conf:.2f}")
            # Updated label to match the data
            print(f"  • Box (xywh): [Xc:{x_center}, Yc:{y_center}, W:{width}, H:{height}]")

    print("="*50 + "\n")


# ------------------ CONFIG  ------------------
DEFECT_WEIGHTS = "models/pineapple_defect/weights/best.pt"
RIPE_WEIGHTS = "models/pineapple_ripe/weights/best.pt"
# SOURCE = "data/pineapple_defect/test/images"
SOURCE = "test_imgs"

CONF = 0.25
IMG_SIZE = 640
DEVICE = '0' # 'cpu' or GPU id like '0'


OUTPUT_PROJECT = "runs/infer"
RIPE_OUTPUT_NAME = "ripe"
DEFECT_OUTPUT_NAME = "defect"
# Overwrite existing output folder if True
EXIST_OK = True
# ---------------------------------------------------------


if not os.path.exists(DEFECT_WEIGHTS):
    raise FileNotFoundError(f"Weights not found: {DEFECT_WEIGHTS}")
if not os.path.exists(RIPE_WEIGHTS):
    raise FileNotFoundError(f"Weights not found: {RIPE_WEIGHTS}")


defect_model = YOLO(DEFECT_WEIGHTS)
ripe_model = YOLO(RIPE_WEIGHTS)

defect_results = defect_model.predict(
    source=SOURCE,
    conf=CONF,
    imgsz=IMG_SIZE,
    device=DEVICE,
    project=OUTPUT_PROJECT,
    name=DEFECT_OUTPUT_NAME,
    exist_ok=EXIST_OK,
    save=True,
)

ripe_results = ripe_model.predict(
    source=SOURCE,
    conf=CONF,
    imgsz=IMG_SIZE,
    device=DEVICE,
    project=OUTPUT_PROJECT,
    name=RIPE_OUTPUT_NAME,
    exist_ok=EXIST_OK,
    save=True,
)

for defect_result,ripe_result in zip(defect_results, ripe_results):
    extract_detections(defect_result)
    extract_detections(ripe_result)