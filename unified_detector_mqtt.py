
from ultralytics import YOLO
import os
import json
import base64
import io
import time
from datetime import datetime
from PIL import Image
import paho.mqtt.client as mqtt
from models.grader import grade_pineapple


def extract_detections(result, detection_type="", verbose=True):
    """
    Parses a single YOLO result object and returns detection details.
    Returns a dictionary with class IDs, names, confidences, and bounding boxes.
    """
    boxes = result.boxes
    detections = {
        'class_ids': [],
        'class_names': [],
        'confidences': [],
        'bboxes': []
    }

    if len(boxes) == 0:
        if verbose:
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

            if verbose:
                w = x2 - x1
                h = y2 - y1
                print(f"  Detection #{i+1}: {class_name.upper()}")
                print(f"    Confidence: {conf:.2f}")
                print(f"    BBox: [{x1}, {y1}, {x2}, {y2}]")
                print(f"    Size: {w:.1f} x {h:.1f} px")

    return detections


def process_image(image_path, ripe_model, defect_model, conf=0.5, img_size=640, device='0', verbose=True):
    """
    Process a single image through both ripeness and defect detection models.
    Returns a dictionary with ripeness class and grade information.
    """
    if verbose:
        print(f"\n[Processing Image: {os.path.basename(image_path)}]")
    
    # Run ripeness detection
    ripe_results = ripe_model.predict(
        source=image_path,
        conf=conf,
        imgsz=img_size,
        device=device,
        save=False,
        verbose=False
    )
    
    # Run defect detection
    defect_results = defect_model.predict(
        source=image_path,
        conf=conf,
        imgsz=img_size,
        device=device,
        save=False,
        verbose=False
    )
    
    # Process results (taking first result since single image)
    ripe_result = ripe_results[0]
    defect_result = defect_results[0]
    
    # Extract ripeness detections
    ripe_detections = extract_detections(ripe_result, "RIPENESS", verbose=verbose)
    
    # Extract defect detections
    defect_detections = extract_detections(defect_result, "DEFECTS", verbose=verbose)
    
    # Calculate grade based on defects
    defect_class_ids = defect_detections['class_ids']
    bs_count = defect_class_ids.count(0)  # Black spots
    holes_count = defect_class_ids.count(1)  # Holes
    wf_count = defect_class_ids.count(2)  # White fungus
    
    grade_info = grade_pineapple(bs_count, holes_count, wf_count)
    
    # Determine ripeness status
    ripeness_status = "unknown"
    ripeness_confidence = 0.0
    if len(ripe_detections['class_names']) > 0:
        ripeness_status = ripe_detections['class_names'][0]  # Take first/dominant class
        ripeness_confidence = float(ripe_detections['confidences'][0])
    
    # Compile result for MQTT
    result_data = {
        'timestamp': datetime.now().isoformat(),
        'ripeness': {
            'status': ripeness_status,
            'confidence': round(ripeness_confidence, 2),
            'all_detections': ripe_detections['class_names']
        },
        'defects': {
            'total': len(defect_class_ids),
            'black_spots': bs_count,
            'holes': holes_count,
            'white_fungus': wf_count
        },
        'grade': {
            'grade': grade_info['grade'],
            'score': round(grade_info['score'], 2),
            'levels': grade_info['levels']
        }
    }
    
    if verbose:
        print(f"\n[RESULTS]")
        print(f"Ripeness: {ripeness_status} ({ripeness_confidence:.2f})")
        print(f"Grade: {grade_info['grade']} (Score: {grade_info['score']:.2f})")
        print(f"Defects: {len(defect_class_ids)} total")
    
    return result_data


class PineappleDetectorMQTT:
    """
    MQTT-enabled Pineapple Detector that receives images and sends back results.
    """
    
    def __init__(self, ripe_weights, defect_weights, broker_address, broker_port=1883,
                 input_topic="pineapple/image", output_topic="pineapple/results",
                 conf=0.35, img_size=640, device='0'):
        """
        Initialize the MQTT detector.
        
        Args:
            ripe_weights: Path to ripeness model weights
            defect_weights: Path to defect model weights
            broker_address: MQTT broker IP/hostname
            broker_port: MQTT broker port (default 1883)
            input_topic: Topic to subscribe for incoming images
            output_topic: Topic to publish detection results
            conf: Confidence threshold
            img_size: Image size for inference
            device: Device for inference ('cpu' or '0', '1', etc.)
        """
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.conf = conf
        self.img_size = img_size
        self.device = device
        
        # Load models
        print("Loading models...")
        if not os.path.exists(ripe_weights):
            raise FileNotFoundError(f"Ripeness model weights not found: {ripe_weights}")
        if not os.path.exists(defect_weights):
            raise FileNotFoundError(f"Defect model weights not found: {defect_weights}")
        
        self.ripe_model = YOLO(ripe_weights)
        self.defect_model = YOLO(defect_weights)
        print("✓ Models loaded successfully!")
        
        # Create temp directory for saving images
        self.temp_dir = "temp_mqtt"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize MQTT client
        self.client = mqtt.Client(client_id="pineapple_detector")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        print(f"Connecting to MQTT broker at {broker_address}:{broker_port}...")
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        if rc == 0:
            print(f"✓ Connected to MQTT broker!")
            print(f"Subscribing to topic: {self.input_topic}")
            client.subscribe(self.input_topic)
            print("Waiting for images...\n")
        else:
            print(f"✗ Connection failed with code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            print(f"\n{'='*70}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Image received on topic: {msg.topic}")
            print('='*70)
            
            # Decode the message
            payload = json.loads(msg.payload.decode())
            
            # Extract image data (assuming base64 encoded)
            if 'image' in payload:
                image_data = base64.b64decode(payload['image'])
                
                # Save image temporarily
                timestamp = int(time.time() * 1000)
                image_filename = f"{self.temp_dir}/temp_{timestamp}.jpg"
                
                # Convert bytes to image and save
                image = Image.open(io.BytesIO(image_data))
                image.save(image_filename)
                
                # Process the image
                result = process_image(
                    image_path=image_filename,
                    ripe_model=self.ripe_model,
                    defect_model=self.defect_model,
                    conf=self.conf,
                    img_size=self.img_size,
                    device=self.device,
                    verbose=True
                )
                
                # Add any metadata from original message
                if 'id' in payload:
                    result['request_id'] = payload['id']
                
                # Publish results
                result_json = json.dumps(result)
                self.client.publish(self.output_topic, result_json)
                
                print(f"\n✓ Results published to topic: {self.output_topic}")
                print(f"{'='*70}\n")
                
                # Clean up temp file
                try:
                    os.remove(image_filename)
                except:
                    pass
                    
            else:
                print("✗ No 'image' field in payload")
                
        except Exception as e:
            print(f"✗ Error processing message: {e}")
            error_response = {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            }
            self.client.publish(self.output_topic, json.dumps(error_response))
    
    def start(self):
        """Start the MQTT client and begin listening"""
        self.client.connect(self.broker_address, self.broker_port, 60)
        print("Detector is running. Press Ctrl+C to stop.\n")
        self.client.loop_forever()
    
    def stop(self):
        """Stop the MQTT client"""
        self.client.disconnect()
        print("\n✓ Detector stopped.")


# ------------------ CONFIGURATION ------------------
RIPE_WEIGHTS = "models/unified_models/pineapple_ripe/weights/best.pt"
DEFECT_WEIGHTS = "models/unified_models/pineapple_defect/weights/best.pt"

# MQTT Configuration
MQTT_BROKER = "localhost"  # Change to your MQTT broker address
MQTT_PORT = 1883
INPUT_TOPIC = "pineapple/image"      # Topic to receive images
OUTPUT_TOPIC = "pineapple/results"   # Topic to send results

# Detection Configuration
CONF = 0.35
IMG_SIZE = 640
DEVICE = '0'  # 'cpu' or GPU id like '0'

# ---------------------------------------------------

if __name__ == "__main__":
    try:
        # Initialize MQTT detector
        detector = PineappleDetectorMQTT(
            ripe_weights=RIPE_WEIGHTS,
            defect_weights=DEFECT_WEIGHTS,
            broker_address=MQTT_BROKER,
            broker_port=MQTT_PORT,
            input_topic=INPUT_TOPIC,
            output_topic=OUTPUT_TOPIC,
            conf=CONF,
            img_size=IMG_SIZE,
            device=DEVICE
        )
        
        # Start listening for images
        detector.start()
        
    except KeyboardInterrupt:
        print("\n\nShutting down detector...")
        detector.stop()
    except Exception as e:
        print(f"\n✗ Error: {e}")