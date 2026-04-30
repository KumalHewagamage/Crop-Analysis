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


class PineappleDetectorMQTT3View:
    """
    MQTT-enabled Pineapple Detector that receives images from three views and sends back results.
    """
    
    def __init__(self, ripe_weights, defect_weights, broker_address, broker_port=1883,
                 input_topic_a="pineapple/imageA", input_topic_b="pineapple/imageB", input_topic_c="pineapple/imageC",
                 output_topic="pineapple/results", output_topic_simple="pineapple/result_simple",
                 conf=0.35, img_size=640, device='0'):
        """
        Initialize the MQTT detector for 3 views.
        """
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.input_topic_a = input_topic_a
        self.input_topic_b = input_topic_b
        self.input_topic_c = input_topic_c
        self.output_topic = output_topic
        self.output_topic_simple = output_topic_simple
        self.conf = conf
        self.img_size = img_size
        self.device = device
        
        # Storage for pairing images from three views
        self.image_pairs = {}  # {request_id: {'imageA': data, 'imageB': data, 'imageC': data}}
        
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
        self.client = mqtt.Client(client_id="pineapple_detector_3view")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        print(f"Connecting to MQTT broker at {broker_address}:{broker_port}...")
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        if rc == 0:
            print(f"✓ Connected to MQTT broker!")
            print(f"Subscribing to topics:")
            print(f"  - {self.input_topic_a}")
            print(f"  - {self.input_topic_b}")
            print(f"  - {self.input_topic_c}")
            client.subscribe(self.input_topic_a)
            client.subscribe(self.input_topic_b)
            client.subscribe(self.input_topic_c)
            print("Waiting for images...\n")
        else:
            print(f"✗ Connection failed with code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            # Decode the message
            payload = json.loads(msg.payload.decode())
            
            # Check for required fields
            if 'image' not in payload:
                print("✗ No 'image' field in payload")
                return
            
            if 'id' not in payload:
                print("✗ No 'id' field in payload - cannot pair images")
                return
            
            request_id = payload['id']
            
            # Determine which view this is
            if msg.topic == self.input_topic_a:
                view = 'imageA'
            elif msg.topic == self.input_topic_b:
                view = 'imageB'
            elif msg.topic == self.input_topic_c:
                view = 'imageC'
            else:
                print(f"✗ Unknown topic: {msg.topic}")
                return
            
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Received {view} for ID: {request_id}")
            
            # Store the image data
            if request_id not in self.image_pairs:
                self.image_pairs[request_id] = {}
            
            self.image_pairs[request_id][view] = payload['image']
            
            # Check if we have all three images
            if 'imageA' in self.image_pairs[request_id] and 'imageB' in self.image_pairs[request_id] and 'imageC' in self.image_pairs[request_id]:
                print(f"✓ All three views received for ID: {request_id}. Processing...")
                print('='*70)
                
                # Process all images
                self.process_image_set(request_id)
                
                # Clean up the pair
                del self.image_pairs[request_id]
            else:
                print(f"  Waiting for other views... (have: {list(self.image_pairs[request_id].keys())})")
                
        except Exception as e:
            print(f"✗ Error processing message: {e}")
            error_response = {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            }
            self.client.publish(self.output_topic, json.dumps(error_response))
    
    def process_image_set(self, request_id):
        """Process a set of images from all three views"""
        try:
            image_data_a = self.image_pairs[request_id]['imageA']
            image_data_b = self.image_pairs[request_id]['imageB']
            image_data_c = self.image_pairs[request_id]['imageC']
            
            # Decode and save images
            timestamp = int(time.time() * 1000)
            image_a_path = f"{self.temp_dir}/temp_{timestamp}_A.jpg"
            image_b_path = f"{self.temp_dir}/temp_{timestamp}_B.jpg"
            image_c_path = f"{self.temp_dir}/temp_{timestamp}_C.jpg"
            
            # Save imageA
            img_bytes_a = base64.b64decode(image_data_a)
            image_a = Image.open(io.BytesIO(img_bytes_a))
            image_a.save(image_a_path)
            
            # Save imageB
            img_bytes_b = base64.b64decode(image_data_b)
            image_b = Image.open(io.BytesIO(img_bytes_b))
            image_b.save(image_b_path)
            
            # Save imageC
            img_bytes_c = base64.b64decode(image_data_c)
            image_c = Image.open(io.BytesIO(img_bytes_c))
            image_c.save(image_c_path)
            
            print("\n[Processing View A]")
            result_a = process_image(
                image_path=image_a_path, ripe_model=self.ripe_model, defect_model=self.defect_model,
                conf=self.conf, img_size=self.img_size, device=self.device, verbose=True
            )
            
            print("\n[Processing View B]")
            result_b = process_image(
                image_path=image_b_path, ripe_model=self.ripe_model, defect_model=self.defect_model,
                conf=self.conf, img_size=self.img_size, device=self.device, verbose=True
            )
            
            print("\n[Processing View C]")
            result_c = process_image(
                image_path=image_c_path, ripe_model=self.ripe_model, defect_model=self.defect_model,
                conf=self.conf, img_size=self.img_size, device=self.device, verbose=True
            )
            
            # Combine results
            combined_result = self.combine_results(result_a, result_b, result_c, request_id)
            
            # Publish full results
            result_json = json.dumps(combined_result)
            self.client.publish(self.output_topic, result_json)
            print(f"\n✓ Results published to: {self.output_topic}")
            
            # Publish simple result
            grade = combined_result['grade']['grade']
            simple_result = 'good' if grade in ['A', 'B'] else 'bad'
            simple_payload = json.dumps({
                'request_id': request_id,
                'result': simple_result,
                'grade': grade,
                'timestamp': combined_result['timestamp']
            })
            self.client.publish(self.output_topic_simple, simple_payload)
            print(f"✓ Simple result '{simple_result}' published to: {self.output_topic_simple}")
            print(f"{'='*70}\n")
            
            # Clean up temp files
            try:
                os.remove(image_a_path)
                os.remove(image_b_path)
                os.remove(image_c_path)
            except:
                pass
                
        except Exception as e:
            print(f"✗ Error processing image set: {e}")
            error_response = {
                'timestamp': datetime.now().isoformat(),
                'request_id': request_id,
                'error': str(e),
                'status': 'failed'
            }
            self.client.publish(self.output_topic, json.dumps(error_response))
    
    def combine_results(self, result_a, result_b, result_c, request_id):
        """Combine results from all three views according to requirements"""
        
        # For ripeness: take the one with highest confidence
        views_ripeness = [
            ('A', result_a['ripeness']),
            ('B', result_b['ripeness']),
            ('C', result_c['ripeness'])
        ]
        best_view = max(views_ripeness, key=lambda item: item[1]['confidence'])
        ripeness_result = best_view[1]
        print(f"\n[Ripeness] Using View {best_view[0]}: {ripeness_result['status']} ({ripeness_result['confidence']:.2f})")
        
        # For defects: sum from all views
        total_black_spots = result_a['defects']['black_spots'] + result_b['defects']['black_spots'] + result_c['defects']['black_spots']
        total_holes = result_a['defects']['holes'] + result_b['defects']['holes'] + result_c['defects']['holes']
        total_white_fungus = result_a['defects']['white_fungus'] + result_b['defects']['white_fungus'] + result_c['defects']['white_fungus']
        total_defects = total_black_spots + total_holes + total_white_fungus
        
        print(f"[Defects] Combined from all views:")
        print(f"  Black spots: {total_black_spots} (A:{result_a['defects']['black_spots']} + B:{result_b['defects']['black_spots']} + C:{result_c['defects']['black_spots']})")
        print(f"  Holes: {total_holes} (A:{result_a['defects']['holes']} + B:{result_b['defects']['holes']} + C:{result_c['defects']['holes']})")
        print(f"  White fungus: {total_white_fungus} (A:{result_a['defects']['white_fungus']} + B:{result_b['defects']['white_fungus']} + C:{result_c['defects']['white_fungus']})")
        print(f"  Total: {total_defects}")
        
        # Calculate grade based on combined defects
        grade_info = grade_pineapple(total_black_spots, total_holes, total_white_fungus)
        
        print(f"[Final Grade] {grade_info['grade']} (Score: {grade_info['score']:.2f})")
        
        # Compile combined result
        combined_result = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'ripeness': ripeness_result,
            'defects': {
                'total': total_defects,
                'black_spots': total_black_spots,
                'holes': total_holes,
                'white_fungus': total_white_fungus,
                'view_a': result_a['defects'],
                'view_b': result_b['defects'],
                'view_c': result_c['defects']
            },
            'grade': {
                'grade': grade_info['grade'],
                'score': round(grade_info['score'], 2),
                'levels': grade_info['levels']
            }
        }
        
        return combined_result
    
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
INPUT_TOPIC_A = "pineapple/imageA"      # Topic to receive view A images
INPUT_TOPIC_B = "pineapple/imageB"      # Topic to receive view B images
INPUT_TOPIC_C = "pineapple/imageC"      # Topic to receive view C images
OUTPUT_TOPIC = "pineapple/results"      # Topic to send detailed results
OUTPUT_TOPIC_SIMPLE = "pineapple/result_simple"  # Topic to send simple good/bad results

# Detection Configuration
CONF = 0.35
IMG_SIZE = 640
DEVICE = '0'  # 'cpu' or GPU id like '0'

# ---------------------------------------------------

if __name__ == "__main__":
    try:
        # Initialize MQTT detector
        detector = PineappleDetectorMQTT3View(
            ripe_weights=RIPE_WEIGHTS,
            defect_weights=DEFECT_WEIGHTS,
            broker_address=MQTT_BROKER,
            broker_port=MQTT_PORT,
            input_topic_a=INPUT_TOPIC_A,
            input_topic_b=INPUT_TOPIC_B,
            input_topic_c=INPUT_TOPIC_C,
            output_topic=OUTPUT_TOPIC,
            output_topic_simple=OUTPUT_TOPIC_SIMPLE,
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
