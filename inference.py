
from ultralytics import YOLO
import os

def extract_detections(result):
    """
    Parses a single YOLOv8 'result' object and prints detection details.
    """
    # 1. Basic Image Info
    path = result.path
    filename = os.path.basename(path)
    img_shape = result.orig_shape  # (height, width)

    print("\n" + "="*50)
    print(f"Image: {filename}")
    print(f"Path:  {path}")
    print(f"Size:  {img_shape[1]}x{img_shape[0]} (WxH)")
    print("-" * 30)

    # 2. Get the Boxes object
    boxes = result.boxes

    if len(boxes) == 0:
        print("No detection.")
    else:
        for i, box in enumerate(boxes):
            
            
            # 1. Class ID & Name
            cls_id = int(box.cls.cpu().item())
            class_name = result.names[cls_id]

            # 2. Confidence Score
            conf = float(box.conf.cpu().item())

            # 3. Bounding Box Coordinates
            # box.xyxy is a tensor of shape (1, 4). 
            coords = box.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = [round(x, 2) for x in coords]

            # --- PRINT ---
            print(f"Detection #{i+1}:")
            print(f"  • Type:       {class_name.upper()}")
            print(f"  • Confidence: {conf:.2f}")
            print(f"  • BBox (xyxy): [{x1}, {y1}, {x2}, {y2}]")
            
            # Print width/height of the defect
            w = x2 - x1
            h = y2 - y1
            print(f"  • Dim (WxH):   {w:.1f} x {h:.1f} px")
            print("-" * 15)

    print("="*50 + "\n")
    return cls_id


# ------------------ CONFIG  ------------------
RIPE_WEIGHTS = "runs/detect/pineapple_ripe_model_v3/weights/best.pt"


# SOURCE = "test_imgs" # multiple images
SOURCE = "test_imgs/IMG_0886_JPG_jpg.rf.9d28427eeeb47dc85a5f56fec72be3f3.jpg" # single image

CONF = 0.25
IMG_SIZE = 640
DEVICE = '0' # 'cpu' or GPU id like '0'


OUTPUT_PROJECT = "runs/infer"
RIPE_OUTPUT_NAME = "ripe"
# Overwrite existing output folder if True
EXIST_OK = True
# ---------------------------------------------------------


if not os.path.exists(RIPE_WEIGHTS):
    raise FileNotFoundError(f"Weights not found: {RIPE_WEIGHTS}")


ripe_model = YOLO(RIPE_WEIGHTS)


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

print(f"\n[Ripe Detection] Total images processed: {len(ripe_results)}")

for ripe_result in ripe_results:
    print(len(ripe_result.boxes), "ripe detections found.")
    class_id = extract_detections(ripe_result)
    print(f"Ripe Class ID: {class_id}")
    