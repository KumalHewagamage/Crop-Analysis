# Two-View Pineapple Detection System

This system captures images from two USB cameras, processes them through YOLO models for ripeness and defect detection, and publishes results via MQTT.

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐         ┌─────────────────────┐
│  USB Cameras (A+B)  │         │   MQTT Broker        │         │  Detection System   │
│                     │         │   (mosquitto)        │         │  (YOLO Models)      │
└──────────┬──────────┘         └──────────┬───────────┘         └──────────┬──────────┘
           │                               │                                │
           │ Captures                      │                                │
           │ when triggered                │                                │
           │                               │                                │
           v                   ┌───────────v──────────┐          ┌──────────v──────────┐
    ┌──────────────────┐      │  capture_trigger     │          │   imageA/imageB     │
    │  Camera Publisher ├─────>│  (trigger topic)     │          │   (input topics)    │
    └──────────────────┘      └──────────────────────┘          └─────────────────────┘
                                                                            │
           Publishes imageA + imageB                                       │
           with matching request_id ─────────────────────────────────────►│
                                                                            │
                                                                            v
                                      Process both views:
                                      - Compare ripeness (highest confidence)
                                      - Sum defects from both views
                                      - Calculate final grade
                                                                            │
                              ┌────────────────────────────────────────────┘
                              │
                              v
               ┌──────────────────────────────────┬──────────────────────────┐
               │                                  │                          │
    ┌──────────v──────────┐          ┌───────────v──────────┐   ┌──────────v──────────┐
    │  pineapple/results  │          │ pineapple/result_    │   │  Other subscribers  │
    │  (detailed JSON)    │          │  simple (good/bad)   │   │                     │
    └─────────────────────┘          └──────────────────────┘   └─────────────────────┘
```

## Components

### 1. Camera Publisher (`mqtt_camera_publisher.py`)
Captures images from two USB webcams and publishes them to MQTT topics.

**Features:**
- Listens for capture triggers via MQTT
- Captures synchronized images from both cameras
- Publishes with matching request IDs for pairing
- Supports preview mode (live window) or headless mode

**MQTT Topics:**
- **Subscribes:** `pineapple/capture_trigger` - receives capture commands
- **Publishes:** 
  - `pineapple/imageA` - Camera A images (base64 encoded)
  - `pineapple/imageB` - Camera B images (base64 encoded)

### 2. Detection System (`unified_detector_mqtt_2view.py`)
Processes images through YOLO models and combines results from both views.

**MQTT Topics:**
- **Subscribes:**
  - `pineapple/imageA` - Camera A images
  - `pineapple/imageB` - Camera B images
- **Publishes:**
  - `pineapple/results` - Detailed detection results
  - `pineapple/result_simple` - Simple good/bad classification

**Processing Logic:**
- **Ripeness:** Selects result with highest confidence from either view
- **Defects:** Sums all detected defects from both views
- **Grade:** Calculated based on combined defect counts (A/B = good, C/D = bad)

### 3. Trigger Sender (`mqtt_trigger_sender.py`)
Utility script to send capture triggers via MQTT.

**Modes:**
- Single trigger: `python mqtt_trigger_sender.py`
- Continuous: `python mqtt_trigger_sender.py --continuous [interval_seconds]`

## Installation

### Prerequisites
```bash
# Install MQTT broker (if not already installed)
sudo apt-get update
sudo apt-get install mosquitto mosquitto-clients

# Start mosquitto service
sudo systemctl start mosquitto
sudo systemctl enable mosquitto
```

### Python Dependencies
```bash
pip install ultralytics opencv-python paho-mqtt pillow
```

## Configuration

### Camera Publisher Configuration
Edit `mqtt_camera_publisher.py`:
```python
MQTT_BROKER = "localhost"          # MQTT broker address
MQTT_PORT = 1883                   # MQTT broker port
TOPIC_A = "pineapple/imageA"       # Camera A topic
TOPIC_B = "pineapple/imageB"       # Camera B topic
TRIGGER_TOPIC = "pineapple/capture_trigger"  # Trigger topic
CAMERA_A_INDEX = 0                 # First USB camera index
CAMERA_B_INDEX = 1                 # Second USB camera index
RUN_MODE = 'preview'               # 'preview' or 'headless'
```

### Detection System Configuration
Edit `unified_detector_mqtt_2view.py`:
```python
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
INPUT_TOPIC_A = "pineapple/imageA"
INPUT_TOPIC_B = "pineapple/imageB"
OUTPUT_TOPIC = "pineapple/results"
OUTPUT_TOPIC_SIMPLE = "pineapple/result_simple"
CONF = 0.35                        # Detection confidence threshold
IMG_SIZE = 640                     # Image size for inference
DEVICE = '0'                       # '0' for GPU, 'cpu' for CPU
```

## Usage

### Step 1: Start MQTT Broker
```bash
# Check if mosquitto is running
sudo systemctl status mosquitto

# If not running, start it
sudo systemctl start mosquitto
```

### Step 2: Start Detection System
```bash
python unified_detector_mqtt_2view.py
```

Expected output:
```
Loading models...
✓ Models loaded successfully!
✓ Connected to MQTT broker!
Subscribing to topics:
  - pineapple/imageA
  - pineapple/imageB
Waiting for images...
```

### Step 3: Start Camera Publisher
```bash
python mqtt_camera_publisher.py
```

Expected output:
```
DUAL CAMERA MQTT PUBLISHER
======================================================================

Scanning for available cameras...
  ✓ Camera found at index 0
  ✓ Camera found at index 1

Using cameras at indices: 0 and 1

Opening Camera A (index 0)...
✓ Camera A initialized
Opening Camera B (index 1)...
✓ Camera B initialized
✓ Cameras ready!

✓ Connected to MQTT broker
✓ Subscribing to trigger topic: pineapple/capture_trigger

LIVE PREVIEW MODE
======================================================================
Controls:
  'c' or SPACE - Manual capture and publish (for testing)
  'q' or ESC - Quit preview
Listening for MQTT triggers on: pineapple/capture_trigger
======================================================================
```

### Step 4: Send Capture Trigger

**Option A: Single capture**
```bash
python mqtt_trigger_sender.py
```

**Option B: Continuous capture (every 5 seconds)**
```bash
python mqtt_trigger_sender.py --continuous 5
```

**Option C: Manual trigger (when in preview mode)**
- Press 'c' or SPACE in the preview window

## Output Format

### Detailed Results (`pineapple/results`)
```json
{
  "timestamp": "2026-03-07T10:30:45.123456",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "ripeness": {
    "status": "ripe",
    "confidence": 0.92,
    "all_detections": ["ripe"]
  },
  "defects": {
    "total": 5,
    "black_spots": 3,
    "holes": 1,
    "white_fungus": 1,
    "view_a": {
      "total": 3,
      "black_spots": 2,
      "holes": 1,
      "white_fungus": 0
    },
    "view_b": {
      "total": 2,
      "black_spots": 1,
      "holes": 0,
      "white_fungus": 1
    }
  },
  "grade": {
    "grade": "B",
    "score": 85.5,
    "levels": {
      "black_spots": "low",
      "holes": "none",
      "white_fungus": "none"
    }
  }
}
```

### Simple Results (`pineapple/result_simple`)
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "result": "good",
  "grade": "B",
  "timestamp": "2026-03-07T10:30:45.123456"
}
```

Values:
- `"good"` - Grade A or B
- `"bad"` - Grade C or D

## Testing

### Test MQTT Topics
```bash
# Subscribe to all topics to monitor messages
mosquitto_sub -h localhost -t 'pineapple/#' -v

# Or subscribe to specific topics
mosquitto_sub -h localhost -t 'pineapple/results'
mosquitto_sub -h localhost -t 'pineapple/result_simple'
```

### Send Manual Trigger
```bash
# Using mosquitto_pub
mosquitto_pub -h localhost -t 'pineapple/capture_trigger' -m '{"id":"test-123","trigger":true}'

# Or use the trigger sender script
python mqtt_trigger_sender.py
```

## Troubleshooting

### Camera Not Found
```bash
# List video devices
ls -l /dev/video*

# Test camera with ffplay (if available)
ffplay /dev/video0
```

### MQTT Connection Issues
```bash
# Check mosquitto status
sudo systemctl status mosquitto

# Check mosquitto logs
sudo journalctl -u mosquitto -f

# Test MQTT connection
mosquitto_pub -h localhost -t test -m "hello"
mosquitto_sub -h localhost -t test
```

### Models Not Found
Ensure model weights exist:
```bash
ls -la models/unified_models/pineapple_ripe/weights/best.pt
ls -la models/unified_models/pineapple_defect/weights/best.pt
```

## Production Deployment

For production use without preview windows:

1. Edit `mqtt_camera_publisher.py`:
   ```python
   RUN_MODE = 'headless'
   ```

2. Run as background services (optional):
   ```bash
   # Start detection system
   nohup python unified_detector_mqtt_2view.py > detector.log 2>&1 &
   
   # Start camera publisher
   nohup python mqtt_camera_publisher.py > camera.log 2>&1 &
   ```

3. Monitor logs:
   ```bash
   tail -f detector.log
   tail -f camera.log
   ```

## Integration Examples

### Python Client
```python
import paho.mqtt.client as mqtt
import json

def on_message(client, userdata, msg):
    result = json.loads(msg.payload.decode())
    grade = result['result']  # 'good' or 'bad'
    print(f"Grade: {grade}")

client = mqtt.Client()
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.subscribe("pineapple/result_simple")
client.loop_forever()
```

### Hardware Trigger Integration
Connect to PLC/Arduino to send triggers:
```python
# When conveyor sensor detects pineapple
import paho.mqtt.publish as publish
publish.single("pineapple/capture_trigger", 
               payload='{"id":"' + sensor_id + '","trigger":true}',
               hostname="localhost")
```
