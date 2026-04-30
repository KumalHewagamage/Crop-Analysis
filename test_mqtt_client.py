"""
Test client for sending images to the MQTT pineapple detector
"""
import paho.mqtt.client as mqtt
import json
import base64
import time
import sys


# ------------------ CONFIGURATION ------------------
BROKER = "broker.hivemq.com"
PORT = 1883
NUM_CAMS = 3  # Set to 2 or 3
IMAGE_PATH = "test_imgs/pineapple.jpg"  # The sample image to send as all views
# ---------------------------------------------------


def on_connect(client, userdata, flags, rc):
    """Callback when connected to broker"""
    if rc == 0:
        print(f"✓ Connected to MQTT broker (Testing for {NUM_CAMS} views)")
        # Subscribe to results topics
        client.subscribe("pineapple/results")
        client.subscribe("pineapple/result_simple")
        print("✓ Subscribed to results topics\n")
    else:
        print(f"✗ Connection failed with code {rc}")


def on_message(client, userdata, msg):
    """Callback when results received"""
    print("\n" + "="*70)
    print("RESULTS RECEIVED")
    print("="*70)
    
    try:
        result = json.loads(msg.payload.decode())
        print(json.dumps(result, indent=2))
        print("="*70 + "\n")
    except Exception as e:
        print(f"Error parsing results: {e}")


def send_image_set(client, image_path, num_views, request_id):
    """
    Send the same image to multiple view topics to simulate a multi-camera set
    """
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        payload = {
            'image': image_base64,
            'id': request_id
        }
        message = json.dumps(payload)
        
        topics = ["pineapple/imageA", "pineapple/imageB"]
        if num_views >= 3:
            topics.append("pineapple/imageC")
            
        print(f"Sending image set for ID: {request_id}")
        for topic in topics:
            client.publish(topic, message)
            print(f"  ✓ Published to {topic}")
        print("✓ Image set sent successfully\n")
        
    except FileNotFoundError:
        print(f"✗ Error: Image file not found: {image_path}")
    except Exception as e:
        print(f"✗ Error sending image set: {e}")


if __name__ == "__main__":
    # Create MQTT client
    client = mqtt.Client(client_id="test_client_multi")
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        print(f"Connecting to MQTT broker at {BROKER}:{PORT}...")
        client.connect(BROKER, PORT, 60)
        
        client.loop_start()
        time.sleep(1) # Wait for connection
        
        # Send the images based on config
        req_id = f"test_{int(time.time())}"
        send_image_set(client, IMAGE_PATH, NUM_CAMS, req_id)
        
        print("Waiting for results... (Press Ctrl+C to exit)")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping client...")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        client.loop_stop()
        client.disconnect()
        print("✓ Disconnected")
