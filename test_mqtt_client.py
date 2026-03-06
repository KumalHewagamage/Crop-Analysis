"""
Test client for sending images to the MQTT pineapple detector
"""
import paho.mqtt.client as mqtt
import json
import base64
import time
import sys


def on_connect(client, userdata, flags, rc):
    """Callback when connected to broker"""
    if rc == 0:
        print("✓ Connected to MQTT broker")
        # Subscribe to results topic
        client.subscribe("pineapple/results")
        print("✓ Subscribed to results topic\n")
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


def send_image(client, image_path, request_id=None):
    """
    Send an image to the detector via MQTT
    
    Args:
        client: MQTT client
        image_path: Path to image file
        request_id: Optional request ID for tracking
    """
    try:
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Create payload
        payload = {
            'image': image_base64,
        }
        
        if request_id:
            payload['id'] = request_id
        
        # Convert to JSON and publish
        message = json.dumps(payload)
        
        print(f"Sending image: {image_path}")
        print(f"Image size: {len(image_data)} bytes")
        print(f"Encoded size: {len(message)} bytes")
        
        client.publish("pineapple/image", message)
        print("✓ Image sent successfully\n")
        
    except FileNotFoundError:
        print(f"✗ Error: Image file not found: {image_path}")
    except Exception as e:
        print(f"✗ Error sending image: {e}")


if __name__ == "__main__":
    # Configuration
    BROKER = "localhost"  # Change to your MQTT broker address
    PORT = 1883
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_mqtt_client.py <image_path>")
        print("Example: python test_mqtt_client.py test_imgs/pineapple.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Create MQTT client
    client = mqtt.Client(client_id="test_client")
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        print(f"Connecting to MQTT broker at {BROKER}:{PORT}...")
        client.connect(BROKER, PORT, 60)
        
        # Start network loop in background
        client.loop_start()
        
        # Wait for connection
        time.sleep(2)
        
        # Send image
        send_image(client, image_path, request_id="test_001")
        
        # Wait for response
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
