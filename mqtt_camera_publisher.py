import cv2
import json
import base64
import time
import uuid
from datetime import datetime
import paho.mqtt.client as mqtt


class DualCameraPublisher:
    """
    Captures images from two USB webcams and publishes them to MQTT topics.
    """
    
    def __init__(self, broker_address, broker_port=1883,
                 topic_a="pineapple/imageA", topic_b="pineapple/imageB",
                 trigger_topic="pineapple/capture_trigger",
                 camera_a_index=0, camera_b_index=1):
        """
        Initialize the dual camera publisher.
        
        Args:
            broker_address: MQTT broker IP/hostname
            broker_port: MQTT broker port (default 1883)
            topic_a: Topic to publish camera A images
            topic_b: Topic to publish camera B images
            trigger_topic: Topic to listen for capture triggers
            camera_a_index: Index of camera A (usually 0)
            camera_b_index: Index of camera B (usually 1)
        """
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.topic_a = topic_a
        self.topic_b = topic_b
        self.trigger_topic = trigger_topic
        self.camera_a_index = camera_a_index
        self.camera_b_index = camera_b_index
        
        # Initialize MQTT client
        self.client = mqtt.Client(client_id="dual_camera_publisher")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.connected = False
        
        # Initialize cameras
        print("Initializing cameras...")
        self.cap_a = None
        self.cap_b = None
        self.initialize_cameras()
        
    def initialize_cameras(self):
        """Initialize both USB cameras"""
        # Initialize Camera A
        print(f"Opening Camera A (index {self.camera_a_index})...")
        self.cap_a = cv2.VideoCapture(self.camera_a_index)
        if not self.cap_a.isOpened():
            raise RuntimeError(f"Failed to open Camera A at index {self.camera_a_index}")
        
        # Set camera A properties for better quality
        self.cap_a.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap_a.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap_a.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        print("✓ Camera A initialized")
        
        # Initialize Camera B
        print(f"Opening Camera B (index {self.camera_b_index})...")
        self.cap_b = cv2.VideoCapture(self.camera_b_index)
        if not self.cap_b.isOpened():
            raise RuntimeError(f"Failed to open Camera B at index {self.camera_b_index}")
        
        # Set camera B properties for better quality
        self.cap_b.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap_b.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap_b.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        print("✓ Camera B initialized")
        
        # Warm up cameras
        print("Warming up cameras...")
        for _ in range(5):
            self.cap_a.read()
            self.cap_b.read()
            time.sleep(0.1)
        print("✓ Cameras ready!\n")
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        if rc == 0:
            print(f"✓ Connected to MQTT broker at {self.broker_address}:{self.broker_port}")
            print(f"✓ Subscribing to trigger topic: {self.trigger_topic}\n")
            client.subscribe(self.trigger_topic)
            self.connected = True
        else:
            print(f"✗ Connection failed with code {rc}")
            self.connected = False
    
    def on_message(self, client, userdata, msg):
        """Callback when message received on trigger topic"""
        try:
            print(f"\n[✓ Capture trigger received on topic: {msg.topic}]")
            
            # Optionally parse payload for request ID
            request_id = None
            try:
                payload = json.loads(msg.payload.decode())
                if 'id' in payload:
                    request_id = payload['id']
            except:
                pass
            
            # Trigger capture and publish
            self.capture_and_publish(request_id=request_id)
            
        except Exception as e:
            print(f"✗ Error processing trigger message: {e}")
    
    def capture_and_publish(self, request_id=None):
        """Capture images from both cameras and publish to MQTT topics"""
        if not self.connected:
            print("✗ Not connected to MQTT broker")
            return False
        
        # Generate unique request ID if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        timestamp = datetime.now().isoformat()
        
        print(f"\n{'='*70}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Capturing images...")
        print(f"Request ID: {request_id}")
        print('='*70)
        
        # Capture from Camera A
        ret_a, frame_a = self.cap_a.read()
        if not ret_a:
            print("✗ Failed to capture from Camera A")
            return False
        
        print("✓ Camera A captured")
        
        # Capture from Camera B
        ret_b, frame_b = self.cap_b.read()
        if not ret_b:
            print("✗ Failed to capture from Camera B")
            return False
        
        print("✓ Camera B captured")
        
        # Encode images to JPEG
        _, buffer_a = cv2.imencode('.jpg', frame_a, [cv2.IMWRITE_JPEG_QUALITY, 90])
        _, buffer_b = cv2.imencode('.jpg', frame_b, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Convert to base64
        image_a_b64 = base64.b64encode(buffer_a).decode('utf-8')
        image_b_b64 = base64.b64encode(buffer_b).decode('utf-8')
        
        # Create payloads
        payload_a = json.dumps({
            'id': request_id,
            'image': image_a_b64,
            'timestamp': timestamp,
            'camera': 'A'
        })
        
        payload_b = json.dumps({
            'id': request_id,
            'image': image_b_b64,
            'timestamp': timestamp,
            'camera': 'B'
        })
        
        # Publish to MQTT topics
        result_a = self.client.publish(self.topic_a, payload_a)
        result_b = self.client.publish(self.topic_b, payload_b)
        
        if result_a.rc == mqtt.MQTT_ERR_SUCCESS and result_b.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"✓ Published to {self.topic_a}")
            print(f"✓ Published to {self.topic_b}")
            print(f"{'='*70}\n")
            return True
        else:
            print("✗ Failed to publish images")
            return False
    
    def show_preview(self):
        """Show live preview from both cameras (optional manual trigger with 'c', press 'q' to quit)"""
        print("\n" + "="*70)
        print("LIVE PREVIEW MODE")
        print("="*70)
        print("Controls:")
        print("  'c' or SPACE - Manual capture and publish (for testing)")
        print("  'q' or ESC - Quit preview")
        print(f"Listening for MQTT triggers on: {self.trigger_topic}")
        print("="*70 + "\n")
        
        while True:
            # Read frames
            ret_a, frame_a = self.cap_a.read()
            ret_b, frame_b = self.cap_b.read()
            
            if not ret_a or not ret_b:
                print("✗ Failed to read from cameras")
                break
            
            # Add labels to frames
            frame_a_display = frame_a.copy()
            frame_b_display = frame_b.copy()
            
            cv2.putText(frame_a_display, "Camera A", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_b_display, "Camera B", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frames
            cv2.imshow('Camera A', frame_a_display)
            cv2.imshow('Camera B', frame_b_display)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') or key == ord(' '):  # Manual capture
                print("\n[Manual capture triggered]")
                self.capture_and_publish()
            elif key == ord('q') or key == 27:  # Quit (ESC key is 27)
                print("\nExiting preview mode...")
                break
        
        # Clean up windows
        cv2.destroyAllWindows()
    
    def run_headless(self):
        """Run without preview window - only MQTT triggered captures"""
        print("\n" + "="*70)
        print("HEADLESS MODE (No preview window)")
        print("="*70)
        print(f"Listening for MQTT triggers on: {self.trigger_topic}")
        print("Press Ctrl+C to stop...")
        print("="*70 + "\n")
        
        try:
            # Keep the script running
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
    
    def connect(self):
        """Connect to MQTT broker"""
        print(f"Connecting to MQTT broker at {self.broker_address}:{self.broker_port}...")
        self.client.connect(self.broker_address, self.broker_port, 60)
        self.client.loop_start()
        
        # Wait for connection
        timeout = 5
        start_time = time.time()
        while not self.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not self.connected:
            raise RuntimeError("Failed to connect to MQTT broker")
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()
    
    def cleanup(self):
        """Release camera resources"""
        print("\nCleaning up...")
        if self.cap_a is not None:
            self.cap_a.release()
        if self.cap_b is not None:
            self.cap_b.release()
        cv2.destroyAllWindows()
        print("✓ Cameras released")


def list_available_cameras(max_test=10):
    """List all available camera indices"""
    print("Scanning for available cameras...")
    available_cameras = []
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"  ✓ Camera found at index {i}")
            cap.release()
    
    if not available_cameras:
        print("  ✗ No cameras found")
    
    print()
    return available_cameras


# ------------------ CONFIGURATION ------------------
MQTT_BROKER = "localhost"  # Change to your MQTT broker address
MQTT_PORT = 1883
TOPIC_A = "pineapple/imageA"
TOPIC_B = "pineapple/imageB"
TRIGGER_TOPIC = "pineapple/capture_trigger"  # Topic to listen for capture commands

# Camera indices (usually 0 and 1 for two USB cameras)
CAMERA_A_INDEX = 0
CAMERA_B_INDEX = 1

# Mode: 'preview' for live preview window, 'headless' for no window
RUN_MODE = 'preview'  # Change to 'headless' for production

# ---------------------------------------------------

if __name__ == "__main__":
    try:
        print("="*70)
        print("DUAL CAMERA MQTT PUBLISHER")
        print("="*70 + "\n")
        
        # List available cameras
        available = list_available_cameras()
        
        if len(available) < 2:
            print("✗ Error: At least 2 cameras required!")
            print(f"  Found only {len(available)} camera(s)")
            if len(available) == 1:
                print(f"  Available at index: {available[0]}")
            exit(1)
        
        print(f"Using cameras at indices: {CAMERA_A_INDEX} and {CAMERA_B_INDEX}\n")
        
        # Initialize publisher
        publisher = DualCameraPublisher(
            broker_address=MQTT_BROKER,
            broker_port=MQTT_PORT,
            topic_a=TOPIC_A,
            topic_b=TOPIC_B,
            trigger_topic=TRIGGER_TOPIC,
            camera_a_index=CAMERA_A_INDEX,
            camera_b_index=CAMERA_B_INDEX
        )
        
        # Connect to MQTT broker
        publisher.connect()
        
        # Run in selected mode
        if RUN_MODE == 'preview':
            # Show live preview (can also manually trigger with 'c')
            publisher.show_preview()
        else:
            # Run headless (MQTT triggers only)
            publisher.run_headless()
        
        # Cleanup
        publisher.disconnect()
        publisher.cleanup()
        
        print("\n✓ Publisher stopped successfully")
        
    except KeyboardInterrupt:
        print("\n\nShutting down publisher...")
        try:
            publisher.disconnect()
            publisher.cleanup()
        except:
            pass
    except Exception as e:
        print(f"\n✗ Error: {e}")
        try:
            publisher.cleanup()
        except:
            pass
