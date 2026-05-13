#!/usr/bin/env python3
"""
tools/test_yolo_only.py
=======================
Standalone script to test grape detection using ONLY the YOLO model.
Bypasses the OpenCV fallback and the 3-frame confirmation window.
"""

import sys
import argparse
from pathlib import Path
import time

import cv2
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.detector import YOLODetector

def load_cfg():
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Test YOLO detection without OpenCV fallback.")
    parser.add_argument("--image", type=str, help="Path to image file (e.g. test10.jpg)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    # Load config and override confidence if provided
    cfg = load_cfg()
    cfg["detection"]["yolo_conf_thres"] = args.conf
    
    print("\n" + "=" * 50)
    print("  YOLO-ONLY DETECTION TEST")
    print("=" * 50)
    
    # Initialize only the YOLO detector
    detector = YOLODetector(cfg)

    if args.image:
        print(f"  Processing image: {args.image}")
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"  ✗ Error: Could not read image '{args.image}'")
            return
            
        t0 = time.perf_counter()
        res = detector.detect(frame)
        ms = (time.perf_counter() - t0) * 1000
        
        vis = frame.copy()
        if res.success:
            print(f"  ✓ Found {len(res.boxes)} clusters. ({ms:.1f}ms)")
            print(f"  ★ POI Confidence: {res.best_score:.2f}")
            
            # Draw all merged boxes in grey
            for (x1, y1, x2, y2), score in zip(res.boxes, res.scores):
                cv2.rectangle(vis, (x1, y1), (x2, y2), (160, 160, 160), 2)
                cv2.putText(vis, f"{score:.2f}", (x1, y1-4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
                
            # Draw the chosen POI (largest area) in bright green
            if res.best_box:
                x1, y1, x2, y2 = res.best_box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(vis, f"POI conf={res.best_score:.2f}", (x1, max(y1-8, 12)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print(f"  ✗ No detections found. ({ms:.1f}ms)")
            
        cv2.imshow("YOLO Only Test", vis)
        print("  Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Live Camera Mode
        print("  Starting PiCamera2 (Left Stereo)...")
        from core.camera_manager import CameraManager
        cam = CameraManager(cfg)
        cam.open_all()
        
        print("  Running live YOLO inference. Press 'q' to quit.")
        try:
            while True:
                ok, L, R = cam.read_stereo()
                if not ok:
                    continue
                    
                t0 = time.perf_counter()
                res = detector.detect(L)
                ms = (time.perf_counter() - t0) * 1000
                
                vis = L.copy()
                if res.success:
                    # Draw all boxes
                    for (x1, y1, x2, y2), score in zip(res.boxes, res.scores):
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (160, 160, 160), 2)
                        
                    # Highlight POI
                    if res.best_box:
                        x1, y1, x2, y2 = res.best_box
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(vis, f"POI {res.best_score:.2f} ({ms:.0f}ms)", 
                                    (x1, max(y1-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, f"Scanning... ({ms:.0f}ms)", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                    
                cv2.imshow("YOLO Only Live", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cam.close_all()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
