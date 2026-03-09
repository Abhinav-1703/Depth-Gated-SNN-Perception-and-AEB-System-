"""
Depth-Gated SpikeYOLO with Robust Distance-Based Braking + UDP + Logging
CPU-ready experimental version
"""

import sys
import os
import time
import socket
import csv
import torch
import cv2
import numpy as np
import pyrealsense2 as rs

# ---------------- CONFIG ---------------- #

CONFIG = {
    "WEIGHTS_PATH": "69M_best.pt",
    "CONF_THRESH": 0.15,
    "IOU_THRESH": 0.45,

    "DEPTH_GATE_METERS": 3.0,   # gating region
    "BRAKE_DISTANCE": 3.5,      # increase for testing first

    "IMG_SIZE": (416, 416),
    "DEVICE": "cpu",

    "UDP_IP": "192.168.7.5",
    "UDP_PORT": 5005,

    "LOG_FILE": "run_log.csv",

    "FONT": cv2.FONT_HERSHEY_SIMPLEX
}

# ---------------- REGISTER SPIKING MODULE ---------------- #

def register_modules():
    sys.path.append(os.getcwd())
    from ultralytics.nn.modules import yolo_spikformer
    sys.modules['yolo_spikformer'] = yolo_spikformer

register_modules()
from ultralytics.nn.modules import yolo_spikformer


# ---------------- LOAD MODEL ---------------- #

def load_model(path, device):
    print("[INFO] Loading model...")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ckpt["model"]
    model.to(device).float().eval()
    return model


# ---------------- DEPTH GATING ---------------- #

def apply_depth_gating(img, depth, depth_scale, gate_distance, size):

    img_r = cv2.resize(img, size)
    depth_r = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)

    depth_m = depth_r.astype(float) * depth_scale
    mask = (depth_m > 0) & (depth_m < gate_distance)

    active = np.count_nonzero(mask)
    total = size[0] * size[1]
    sparsity = 1.0 - (active / total)

    gated_img = img_r.copy()
    gated_img[~mask] = [0, 0, 0]

    tensor = torch.from_numpy(gated_img).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return tensor, gated_img, sparsity


# ---------------- ROBUST DEPTH SAMPLING ---------------- #

def get_depth_distance(depth_np, cx, cy, depth_scale):
    """
    Uses median of 5x5 window instead of single pixel (critical fix).
    """

    h, w = depth_np.shape

    x1 = max(cx - 2, 0)
    x2 = min(cx + 3, w)
    y1 = max(cy - 2, 0)
    y2 = min(cy + 3, h)

    window = depth_np[y1:y2, x1:x2]
    valid = window[window > 0]

    if len(valid) == 0:
        return None

    return np.median(valid) * depth_scale


# ---------------- MAIN ---------------- #

def main():

    model = load_model(CONFIG["WEIGHTS_PATH"], CONFIG["DEVICE"])

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_target = (CONFIG["UDP_IP"], CONFIG["UDP_PORT"])

    log_file = open(CONFIG["LOG_FILE"], "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["frame", "sparsity_pct", "latency_ms", "closest_m", "brake"])

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(cfg)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    print("[INFO] Running... Press ESC to quit.")
    frame_id = 0

    try:
        while True:

            start = time.time()

            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color = frames.get_color_frame()
            depth = frames.get_depth_frame()

            if not color or not depth:
                continue

            img = np.asanyarray(color.get_data())
            depth_np = np.asanyarray(depth.get_data())

            input_tensor, gated_view, sparsity = apply_depth_gating(
                img, depth_np, depth_scale,
                CONFIG["DEPTH_GATE_METERS"],
                CONFIG["IMG_SIZE"]
            )

            with torch.no_grad():
                preds = model(input_tensor)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]

            from ultralytics.utils.ops import non_max_suppression, scale_boxes
            preds = non_max_suppression(preds, CONFIG["CONF_THRESH"], CONFIG["IOU_THRESH"])

            latency = (time.time() - start) * 1000

            det = preds[0]

            closest = None
            brake = False

            if len(det):
                det[:, :4] = scale_boxes(input_tensor.shape[2:], det[:, :4], img.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    dist = get_depth_distance(depth_np, cx, cy, depth_scale)

                    if dist is not None:
                        if closest is None or dist < closest:
                            closest = dist

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if closest is not None and closest < CONFIG["BRAKE_DISTANCE"]:
                brake = True

            cmd = "Y" if brake else "N"
            message = f"{cmd},{latency:.2f}"
            sock.sendto(message.encode(), udp_target)

            print("UDP SENT:", message)  # debug confirmation

            logger.writerow([
                frame_id,
                sparsity * 100,
                latency,
                closest if closest else -1,
                cmd
            ])
            frame_id += 1

            gated_display = cv2.resize(gated_view, (img.shape[1], img.shape[0]))
            combined = np.hstack((img, gated_display))

            if closest is not None:
                cv2.putText(combined, f"Closest: {closest:.2f} m", (20,120),
                            CONFIG["FONT"], 0.7, (0,200,255), 2)

            cv2.putText(combined, f"Sparsity: {sparsity*100:.1f}%", (20,60),
                        CONFIG["FONT"], 0.7, (0,255,0), 2)
            cv2.putText(combined, f"Latency: {latency:.1f} ms", (20,90),
                        CONFIG["FONT"], 0.7, (255,255,0), 2)

            cv2.imshow("Original | Depth-Gated", combined)

            if cv2.waitKey(1) == 27:
                break

    finally:
        pipeline.stop()
        log_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
