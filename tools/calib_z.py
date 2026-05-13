#!/usr/bin/env python3
import sys
import time
from pathlib import Path
import cv2
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_cfg():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("  ✗ picamera2 not installed. Run on RPi5.")
        sys.exit(1)

    W = cfg["cameras"]["stereo_left"]["width"]
    H = cfg["cameras"]["stereo_left"]["height"]
    idx = cfg["cameras"]["stereo_left"]["index"]

    print(f"Opening LEFT stereo camera (index {idx})...")
    cam = Picamera2(idx)
    config = cam.create_preview_configuration(
        main={"size": (2304, 1296), "format": "RGB888"}
    )
    cam.configure(config)
    cam.start()
    
    # Force full sensor crop to match detection FOV
    props = cam.camera_properties
    full_size = props.get("PixelArraySize", None)
    if full_size is not None and len(full_size) == 2:
        cam.set_controls({"ScalerCrop": (0, 0, int(full_size[0]), int(full_size[1]))})

    print("Warming up camera...")
    time.sleep(2.0)
    
    print("\n" + "="*50)
    print("Z-CALIBRATION TOOL")
    print("="*50)
    print("1. Place a measuring tape vertically at a known depth (e.g. 500mm or 1000mm from the camera lens).")
    print("2. Look at the camera feed on the screen.")
    print("3. Note down the height (from the wooden base) at the VERY TOP line.")
    print("4. Note down the height (from the wooden base) at the VERY BOTTOM line.")
    print("Press 'q' in the image window to quit.")
    print("="*50 + "\n")

    cv2.namedWindow("Z-Calibration", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            frame = cam.capture_array()
            frame = cv2.resize(frame, (W, H))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Draw center crosshairs
            cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 255, 0), 2)
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 0), 2)
            
            # Draw top and bottom reference lines
            cv2.line(frame, (W // 2 - 50, 5), (W // 2 + 50, 5), (0, 0, 255), 3)
            cv2.line(frame, (W // 2 - 50, H - 5), (W // 2 + 50, H - 5), (0, 0, 255), 3)
            
            # Labels
            cv2.putText(frame, "Top (y=0)", (W//2 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Center (y={H//2})", (W//2 + 20, H//2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Bottom (y={H})", (W//2 + 20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Z-Calibration", frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    finally:
        print("Closing camera...")
        cam.stop()
        cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
