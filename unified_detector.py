
from ultralytics import YOLO
import os
from models.grader import grade_pineapple


def extract_detections(result, detection_type=""):
    """
    Parses a single YOLO result object and prints detection details.
    Returns a dictionary with class IDs, names, confidences, and bounding boxes.
    """
    path = result.path
    filename = os.path.basename(path)
    img_shape = result.orig_shape  # (height, width)

    print(f"\n[{detection_type}]")
    print("-" * 30)

    boxes = result.boxes
    detections = {
        'class_ids': [],
        'class_names': [],
        'confidences': [],
        'bboxes': []
    }

    if len(boxes) == 0:
        print(f"  No {detection_type.lower()} detected.")
    else:
        for i, box in enumerate(boxes):
            # Class ID & Name
            cls_id = int(box.cls.cpu().item())
            class_name = result.names[cls_id]
            detections['class_ids'].append(cls_id)
            detections['class_names'].append(class_name)

            # Confidence Score
            conf = float(box.conf.cpu().item())
            detections['confidences'].append(conf)

            # Bounding Box Coordinates
            coords = box.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = [round(x, 2) for x in coords]
            detections['bboxes'].append([x1, y1, x2, y2])

            # Print details
            w = x2 - x1
            h = y2 - y1
            print(f"  Detection #{i+1}: {class_name.upper()}")
            print(f"    Confidence: {conf:.2f}")
            print(f"    BBox: [{x1}, {y1}, {x2}, {y2}]")
            print(f"    Size: {w:.1f} x {h:.1f} px")

    return detections


def unified_detect(image_source, ripe_model, defect_model, conf=0.5, img_size=640, device='0', save_results=True):
    """
    Run both ripeness and defect detection on the same image(s).
    Returns combined results with grading information.
    """
    print("\n" + "="*70)
    print("UNIFIED PINEAPPLE DETECTOR")
    print("="*70)
    
    # Run ripeness detection
    print("\n[PHASE 1: Ripeness Detection]")
    ripe_results = ripe_model.predict(
        source=image_source,
        conf=conf,
        imgsz=img_size,
        device=device,
        save=save_results,
        project="runs/infer",
        name="unified_ripe",
        exist_ok=True,
        verbose=False
    )
    
    # Run defect detection
    print("\n[PHASE 2: Defect Detection]")
    defect_results = defect_model.predict(
        source=image_source,
        conf=conf,
        imgsz=img_size,
        device=device,
        save=save_results,
        project="runs/infer",
        name="unified_defect",
        exist_ok=True,
        verbose=False
    )
    
    # Process results for each image
    unified_results = []
    
    print("\n" + "="*70)
    print("COMBINED RESULTS")
    print("="*70)
    
    for idx, (ripe_result, defect_result) in enumerate(zip(ripe_results, defect_results)):
        filename = os.path.basename(ripe_result.path)
        img_shape = ripe_result.orig_shape
        
        print(f"\n{'='*70}")
        print(f"Image #{idx+1}: {filename}")
        print(f"Size: {img_shape[1]}x{img_shape[0]} (WxH)")
        print("="*70)
        
        # Extract ripeness detections
        ripe_detections = extract_detections(ripe_result, "RIPENESS")
        
        # Extract defect detections
        defect_detections = extract_detections(defect_result, "DEFECTS")
        
        # Calculate grade based on defects
        defect_class_ids = defect_detections['class_ids']
        bs_count = defect_class_ids.count(0)  # Black spots
        holes_count = defect_class_ids.count(1)  # Holes
        wf_count = defect_class_ids.count(2)  # White fungus
        
        grade_info = grade_pineapple(bs_count, holes_count, wf_count)
        
        # Compile unified result
        result_data = {
            'filename': filename,
            'img_shape': img_shape,
            'ripeness': {
                'detections': len(ripe_detections['class_ids']),
                'classes': ripe_detections['class_names'],
                'confidences': ripe_detections['confidences']
            },
            'defects': {
                'total': len(defect_class_ids),
                'black_spots': bs_count,
                'holes': holes_count,
                'white_fungus': wf_count,
                'classes': defect_detections['class_names'],
                'confidences': defect_detections['confidences']
            },
            'grade': grade_info
        }
        
        # Print summary
        print("\n" + "-"*70)
        print("SUMMARY")
        print("-"*70)
        print(f"Ripeness Status: {', '.join(ripe_detections['class_names']) if ripe_detections['class_names'] else 'No ripeness detected'}")
        print(f"Total Defects: {result_data['defects']['total']}")
        print(f"  • Black Spots: {bs_count}")
        print(f"  • Holes: {holes_count}")
        print(f"  • White Fungus: {wf_count}")
        print(f"\nQuality Grade: {grade_info['grade']}")
        print(f"Quality Score: {grade_info['score']:.2f}")
        print(f"Severity Levels: {grade_info['levels']}")
        print("="*70 + "\n")
        
        unified_results.append(result_data)
    
    return unified_results


# ------------------ CONFIGURATION ------------------
RIPE_WEIGHTS = "models/unified_models/pineapple_ripe/weights/best.pt"
DEFECT_WEIGHTS = "models/unified_models/pineapple_defect/weights/best.pt"

# Image source - can be a single image or directory
SOURCE = "temp"  # multiple images
# SOURCE = "test_imgs/IMG_0803_JPG_jpg.rf.4aa2d93dd827287723a64a11de6b5f1f.jpg"  # single image

CONF = 0.35
IMG_SIZE = 640
DEVICE = '0'  # 'cpu' or GPU id like '0'
SAVE_RESULTS = True

# ---------------------------------------------------

if __name__ == "__main__":
    # Verify model weights exist
    if not os.path.exists(RIPE_WEIGHTS):
        raise FileNotFoundError(f"Ripeness model weights not found: {RIPE_WEIGHTS}")
    if not os.path.exists(DEFECT_WEIGHTS):
        raise FileNotFoundError(f"Defect model weights not found: {DEFECT_WEIGHTS}")
    
    # Load models
    print("Loading models...")
    ripe_model = YOLO(RIPE_WEIGHTS)
    defect_model = YOLO(DEFECT_WEIGHTS)
    print("Models loaded successfully!\n")
    
    # Run unified detection
    results = unified_detect(
        image_source=SOURCE,
        ripe_model=ripe_model,
        defect_model=defect_model,
        conf=CONF,
        img_size=IMG_SIZE,
        device=DEVICE,
        save_results=SAVE_RESULTS
    )
    
    print(f"\n✓ Processing complete! Total images analyzed: {len(results)}")