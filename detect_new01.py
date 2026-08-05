#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import time

import cv2
import numpy as np

try:
    import serial
except ImportError:
    serial = None


def main():
    parser = argparse.ArgumentParser(description="H problem ball pixel detector")
    parser.add_argument("--debug", action="store_true", help="show image, mask, and trackbars")
    parser.add_argument("--run", action="store_true", help="headless serial mode")
    parser.add_argument("--show", action="store_true", help="show windows without debug trackbars")
    parser.add_argument("--camera", default="/dev/video10")
    parser.add_argument("--serial", default="/dev/ttyS3")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    # ==================== adjustable parameters ====================
    CAMERA = args.camera
    FRAME_W = 640
    FRAME_H = 480
    FPS = 120

    ROI_X1 = 0
    ROI_X2 = 630
    ROI_Y1 = 195
    ROI_Y2 = 250

    LOCAL_ROI_HALF_W = 85
    LOCAL_ROI_HALF_H = 24
    LOST_RESET_FRAMES = 5
    LOCAL_ROI_GROW_PER_LOST = 25

    TRACK_Y = 228
    TRACK_BAND = 42

    USE_SEGMENT_THRESHOLD = True
    MASK_MODE = 2
    SEGMENT_COUNT = 3
    SEGMENT_OVERLAP = 35
    THRESH_PERCENTILE = 50
    THRESH_OFFSET = 20
    THRESH_MIN = 65
    THRESH_MAX = 190
    MANUAL_THRESHOLD = 145
    USE_MANUAL_RESCUE = True
    LOCAL_BG_KSIZE = 41
    LOCAL_DIFF_TH = 18
    USE_HOUGH_RESCUE = True
    HOUGH_PARAM2 = 14
    HOUGH_MIN_RADIUS = 5
    HOUGH_MAX_RADIUS = 18

    BLUR_KSIZE = 3
    OPEN_ITER = 1
    CLOSE_ITER = 1

    MIN_AREA = 180
    MAX_AREA = 650
    MIN_W = 4
    MAX_W = 36
    MIN_H = 4
    MAX_H = 36
    MAX_ASPECT = 1.9
    MAX_JUMP_PX = 95

    # Kalman is used only to predict the next search ROI.
    # Serial output still uses only the current frame's real connected component.
    USE_KALMAN_FOR_SEARCH = True
    PROCESS_NOISE = 12.0
    MEASURE_NOISE = 20.0

    CENTER_X = 320
    PRINT_EVERY = 15
    USE_SERIAL = True
    UART_PORT = args.serial
    UART_BAUD = args.baud
    PACKET_HEAD = "<"
    PACKET_TAIL = ">"

    # Default is equivalent to:
    # python3 detect_new01.py --debug --camera /dev/video10 --serial /dev/ttyS3
    # Use --run --show for boot/autostart when the board has a local screen.
    DEBUG_MODE = not args.run
    SHOW_WINDOWS_IN_RUN = args.show

    # ==================== open camera ====================
    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera: {CAMERA}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    real_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    real_fourcc_text = "".join(chr((real_fourcc >> 8 * i) & 0xFF) for i in range(4))
    print(
        "camera actual:",
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
        f"fps={cap.get(cv2.CAP_PROP_FPS):.1f}",
        f"fourcc={real_fourcc_text}",
        flush=True,
    )

    # ==================== open serial ====================
    uart = None
    if USE_SERIAL:
        if serial is None:
            print("pyserial not installed, serial disabled. install: pip3 install pyserial", flush=True)
        else:
            try:
                uart = serial.Serial(UART_PORT, UART_BAUD, timeout=0, write_timeout=0.02)
                print(f"serial actual: {UART_PORT} {UART_BAUD}", flush=True)
            except Exception as exc:
                print(f"serial open failed, serial disabled: {exc}", flush=True)

    # ==================== debug trackbars ====================
    if DEBUG_MODE:
        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Controls", 430, 620)
        cv2.createTrackbar("show", "Controls", 1, 1, lambda _v: None)
        cv2.createTrackbar("mode", "Controls", MASK_MODE, 4, lambda _v: None)
        cv2.createTrackbar("seg_th", "Controls", 1, 1, lambda _v: None)
        cv2.createTrackbar("segments", "Controls", SEGMENT_COUNT, 6, lambda _v: None)
        cv2.createTrackbar("overlap", "Controls", SEGMENT_OVERLAP, 100, lambda _v: None)
        cv2.createTrackbar("percent", "Controls", THRESH_PERCENTILE, 100, lambda _v: None)
        cv2.createTrackbar("offset", "Controls", THRESH_OFFSET, 80, lambda _v: None)
        cv2.createTrackbar("diff_th", "Controls", LOCAL_DIFF_TH, 80, lambda _v: None)
        cv2.createTrackbar("bg_k", "Controls", LOCAL_BG_KSIZE, 99, lambda _v: None)
        cv2.createTrackbar("th_min", "Controls", THRESH_MIN, 255, lambda _v: None)
        cv2.createTrackbar("th_max", "Controls", THRESH_MAX, 255, lambda _v: None)
        cv2.createTrackbar("manual_th", "Controls", MANUAL_THRESHOLD, 255, lambda _v: None)
        cv2.createTrackbar("rescue", "Controls", 1 if USE_MANUAL_RESCUE else 0, 1, lambda _v: None)
        cv2.createTrackbar("hough", "Controls", 1 if USE_HOUGH_RESCUE else 0, 1, lambda _v: None)
        cv2.createTrackbar("min_area", "Controls", MIN_AREA, 1000, lambda _v: None)
        cv2.createTrackbar("max_area", "Controls", MAX_AREA, 1500, lambda _v: None)
        cv2.createTrackbar("roi_y1", "Controls", ROI_Y1, FRAME_H - 1, lambda _v: None)
        cv2.createTrackbar("roi_y2", "Controls", ROI_Y2, FRAME_H - 1, lambda _v: None)
        cv2.createTrackbar("track_y", "Controls", TRACK_Y, FRAME_H - 1, lambda _v: None)
        cv2.createTrackbar("track_band", "Controls", TRACK_BAND, 140, lambda _v: None)
        cv2.createTrackbar("max_jump", "Controls", MAX_JUMP_PX, 250, lambda _v: None)
        cv2.createTrackbar("close_iter", "Controls", CLOSE_ITER, 4, lambda _v: None)

    # ==================== Kalman init ====================
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        np.float32,
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASURE_NOISE
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf_ready = False

    last_measured_center = None
    search_center = None
    lost_count = 0
    seq = 0
    fps_smooth = 0.0
    last_time = time.monotonic()
    kernel = np.ones((3, 3), np.uint8)

    # ==================== main loop ====================
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("camera read failed", flush=True)
            break

        # 1. Calculate frame time.
        now = time.monotonic()
        dt = max(now - last_time, 1e-3)
        last_time = now
        fps_now = 1.0 / dt
        fps_smooth = fps_now if fps_smooth <= 0 else fps_smooth * 0.8 + fps_now * 0.2

        # 2. Predict search center only. This prediction is not sent to the MCU.
        if USE_KALMAN_FOR_SEARCH and kf_ready:
            kf.transitionMatrix[0, 2] = dt
            kf.transitionMatrix[1, 3] = dt
            prediction = kf.predict()
            search_center = (float(prediction[0, 0]), float(prediction[1, 0]))
        elif last_measured_center is not None:
            search_center = last_measured_center

        # 3. Read debug controls.
        if DEBUG_MODE:
            show_windows = cv2.getTrackbarPos("show", "Controls") == 1
            MASK_MODE = cv2.getTrackbarPos("mode", "Controls")
            USE_SEGMENT_THRESHOLD = cv2.getTrackbarPos("seg_th", "Controls") == 1
            SEGMENT_COUNT = max(1, cv2.getTrackbarPos("segments", "Controls"))
            SEGMENT_OVERLAP = cv2.getTrackbarPos("overlap", "Controls")
            THRESH_PERCENTILE = cv2.getTrackbarPos("percent", "Controls")
            THRESH_OFFSET = cv2.getTrackbarPos("offset", "Controls")
            LOCAL_DIFF_TH = cv2.getTrackbarPos("diff_th", "Controls")
            LOCAL_BG_KSIZE = cv2.getTrackbarPos("bg_k", "Controls")
            if LOCAL_BG_KSIZE % 2 == 0:
                LOCAL_BG_KSIZE += 1
            LOCAL_BG_KSIZE = max(9, LOCAL_BG_KSIZE)
            THRESH_MIN = cv2.getTrackbarPos("th_min", "Controls")
            THRESH_MAX = cv2.getTrackbarPos("th_max", "Controls")
            if THRESH_MAX <= THRESH_MIN:
                THRESH_MAX = min(255, THRESH_MIN + 1)
            MANUAL_THRESHOLD = cv2.getTrackbarPos("manual_th", "Controls")
            USE_MANUAL_RESCUE = cv2.getTrackbarPos("rescue", "Controls") == 1
            USE_HOUGH_RESCUE = cv2.getTrackbarPos("hough", "Controls") == 1
            MIN_AREA = max(1, cv2.getTrackbarPos("min_area", "Controls"))
            MAX_AREA = max(MIN_AREA + 1, cv2.getTrackbarPos("max_area", "Controls"))
            ROI_Y1 = cv2.getTrackbarPos("roi_y1", "Controls")
            ROI_Y2 = cv2.getTrackbarPos("roi_y2", "Controls")
            if ROI_Y2 <= ROI_Y1 + 5:
                ROI_Y2 = min(FRAME_H, ROI_Y1 + 6)
            TRACK_Y = cv2.getTrackbarPos("track_y", "Controls")
            TRACK_BAND = max(8, cv2.getTrackbarPos("track_band", "Controls"))
            MAX_JUMP_PX = max(10, cv2.getTrackbarPos("max_jump", "Controls"))
            CLOSE_ITER = cv2.getTrackbarPos("close_iter", "Controls")
        else:
            show_windows = SHOW_WINDOWS_IN_RUN

        # 4. Decide this frame's search ROI.
        using_local_roi = search_center is not None and lost_count < LOST_RESET_FRAMES
        if using_local_roi:
            grow = lost_count * LOCAL_ROI_GROW_PER_LOST
            half_w = LOCAL_ROI_HALF_W + grow
            half_h = LOCAL_ROI_HALF_H + grow
            search_x1 = max(ROI_X1, int(search_center[0] - half_w))
            search_x2 = min(ROI_X2, int(search_center[0] + half_w))
            search_y1 = max(ROI_Y1, int(search_center[1] - half_h))
            search_y2 = min(ROI_Y2, int(search_center[1] + half_h))
        else:
            search_x1 = ROI_X1
            search_x2 = ROI_X2
            search_y1 = ROI_Y1
            search_y2 = ROI_Y2

        # 5. Crop ROI and convert to gray.
        if frame.ndim == 2:
            roi_gray = frame[search_y1:search_y2, search_x1:search_x2]
            view = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if show_windows else None
        else:
            roi_frame = frame[search_y1:search_y2, search_x1:search_x2]
            roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            view = frame.copy() if show_windows else None

        # 6. Denoise.
        if BLUR_KSIZE % 2 == 0:
            BLUR_KSIZE += 1
        roi_gray = cv2.medianBlur(roi_gray, BLUR_KSIZE)

        # 7. Build binary mask.
        # mode 0: manual fixed dark threshold.
        # mode 1: overlapped segment threshold.
        # mode 2: local dark contrast, recommended for uneven lighting.
        # mode 3: local bright contrast.
        # mode 4: local absolute contrast.
        if MASK_MODE == 0:
            _, mask = cv2.threshold(roi_gray, MANUAL_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
            used_thresholds = [MANUAL_THRESHOLD]
            debug_view = mask.copy()
        elif MASK_MODE == 1 and USE_SEGMENT_THRESHOLD:
            roi_h, roi_w = roi_gray.shape[:2]
            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            used_thresholds = []
            step = float(roi_w) / float(SEGMENT_COUNT)

            for i in range(SEGMENT_COUNT):
                base_x1 = int(round(i * step))
                base_x2 = int(round((i + 1) * step))
                x1 = max(0, base_x1 - SEGMENT_OVERLAP)
                x2 = min(roi_w, base_x2 + SEGMENT_OVERLAP)
                if x2 <= x1:
                    continue

                part_gray = roi_gray[:, x1:x2]
                base_gray = float(np.percentile(part_gray, THRESH_PERCENTILE))
                threshold_value = int(base_gray - THRESH_OFFSET)
                threshold_value = max(THRESH_MIN, min(THRESH_MAX, threshold_value))
                used_thresholds.append(threshold_value)

                _, part_mask = cv2.threshold(
                    part_gray,
                    threshold_value,
                    255,
                    cv2.THRESH_BINARY_INV,
                )
                mask[:, x1:x2] = cv2.bitwise_or(mask[:, x1:x2], part_mask)

            # Fixed-threshold rescue is useful when one segment's automatic
            # threshold misses the ball, especially near the bright/dark middle.
            if USE_MANUAL_RESCUE:
                _, rescue_mask = cv2.threshold(
                    roi_gray,
                    MANUAL_THRESHOLD,
                    255,
                    cv2.THRESH_BINARY_INV,
                )
                mask = cv2.bitwise_or(mask, rescue_mask)
                used_thresholds.append(MANUAL_THRESHOLD)
            debug_view = mask.copy()
        else:
            bg = cv2.GaussianBlur(roi_gray, (LOCAL_BG_KSIZE, LOCAL_BG_KSIZE), 0)
            if MASK_MODE == 3:
                diff = cv2.subtract(roi_gray, bg)
            elif MASK_MODE == 4:
                diff = cv2.absdiff(roi_gray, bg)
            else:
                diff = cv2.subtract(bg, roi_gray)

            _, mask = cv2.threshold(diff, LOCAL_DIFF_TH, 255, cv2.THRESH_BINARY)
            used_thresholds = [LOCAL_DIFF_TH]
            debug_view = diff

        # 8. Morphology after full mask merge, so boundary-split balls can reconnect.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITER)

        # 9. Connected components on the complete ROI mask.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

        target_center = None
        target_box = None
        target_score = -1.0
        count_raw = max(0, num_labels - 1)
        count_area = 0
        count_size = 0
        count_band = 0
        count_aspect = 0
        count_jump = 0
        track_y_min = TRACK_Y - TRACK_BAND / 2.0
        track_y_max = TRACK_Y + TRACK_BAND / 2.0

        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])

            if area < MIN_AREA or area > MAX_AREA:
                continue
            count_area += 1

            if w < MIN_W or w > MAX_W or h < MIN_H or h > MAX_H:
                continue
            count_size += 1

            cx = search_x1 + float(centroids[label][0])
            cy = search_y1 + float(centroids[label][1])
            if cy < track_y_min or cy > track_y_max:
                continue
            count_band += 1

            aspect = max(w / max(h, 1), h / max(w, 1))
            if aspect > MAX_ASPECT:
                continue
            count_aspect += 1

            if last_measured_center is not None:
                dist = math.hypot(cx - last_measured_center[0], cy - last_measured_center[1])
                if dist > MAX_JUMP_PX:
                    continue
            else:
                dist = 0.0
            count_jump += 1

            y_penalty = abs(cy - TRACK_Y) * 0.5
            score = area / aspect - dist * 0.25 - y_penalty
            if score > target_score:
                target_score = score
                target_center = (cx, cy)
                target_box = (search_x1 + x, search_y1 + y, w, h)

        # 10. Hough rescue. It is slower, so use it only after component detection fails.
        if target_center is None and USE_HOUGH_RESCUE:
            circles = cv2.HoughCircles(
                roi_gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=24,
                param1=80,
                param2=HOUGH_PARAM2,
                minRadius=HOUGH_MIN_RADIUS,
                maxRadius=HOUGH_MAX_RADIUS,
            )
            if circles is not None:
                circles = np.round(circles[0]).astype(int)
                for x, y, r in circles:
                    cx = search_x1 + float(x)
                    cy = search_y1 + float(y)
                    if cy < track_y_min or cy > track_y_max:
                        continue

                    if last_measured_center is not None:
                        dist = math.hypot(cx - last_measured_center[0], cy - last_measured_center[1])
                        if dist > MAX_JUMP_PX:
                            continue
                    else:
                        dist = 0.0

                    x1 = max(0, x - r)
                    x2 = min(roi_gray.shape[1], x + r + 1)
                    y1 = max(0, y - r)
                    y2 = min(roi_gray.shape[0], y + r + 1)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    mean_inside = cv2.mean(roi_gray[y1:y2, x1:x2])[0]
                    ring_r = int(r * 1.8)
                    rx1 = max(0, x - ring_r)
                    rx2 = min(roi_gray.shape[1], x + ring_r + 1)
                    ry1 = max(0, y - ring_r)
                    ry2 = min(roi_gray.shape[0], y + ring_r + 1)
                    mean_ring = cv2.mean(roi_gray[ry1:ry2, rx1:rx2])[0]
                    contrast = abs(mean_ring - mean_inside)
                    if contrast < 5:
                        continue

                    score = 80.0 + contrast * 3.0 + r * 2.0 - dist * 0.3 - abs(cy - TRACK_Y) * 0.5
                    if score > target_score:
                        target_score = score
                        target_center = (cx, cy)
                        target_box = (int(cx - r), int(cy - r), int(2 * r), int(2 * r))

        # 11. Update Kalman with real detection and prepare serial packet.
        if target_center is not None:
            if USE_KALMAN_FOR_SEARCH:
                measurement = np.array(
                    [[np.float32(target_center[0])], [np.float32(target_center[1])]]
                )
                if not kf_ready:
                    kf.statePost = np.array(
                        [[np.float32(target_center[0])], [np.float32(target_center[1])], [0], [0]],
                        np.float32,
                    )
                    kf_ready = True
                else:
                    kf.correct(measurement)

            last_measured_center = target_center
            lost_count = 0
            cx_send = int(round(target_center[0]))
            cy_send = int(round(target_center[1]))
            packet = f"{PACKET_HEAD}Target,{cx_send},{cy_send}{PACKET_TAIL}\n"
        else:
            lost_count += 1
            if lost_count >= LOST_RESET_FRAMES:
                last_measured_center = None
                search_center = None
                kf_ready = False
            packet = f"{PACKET_HEAD}Lost,0,0{PACKET_TAIL}\n"

        # 12. Send only real current-frame result. No predicted Target is sent.
        if uart is not None:
            try:
                uart.write(packet.encode("ascii"))
            except Exception as exc:
                print(f"serial write failed, serial disabled: {exc}", flush=True)
                try:
                    uart.close()
                except Exception:
                    pass
                uart = None

        # 13. Print low-frequency debug log.
        if seq % PRINT_EVERY == 0:
            th_text = "/".join(str(v) for v in used_thresholds)
            count_text = f"R{count_raw}>A{count_area}>S{count_size}>Y{count_band}>P{count_aspect}>J{count_jump}"
            if target_center is not None:
                print(
                    f"B,{seq},1,cx={target_center[0]:.1f},cy={target_center[1]:.1f},"
                    f"score={target_score:.1f},fps={fps_smooth:.1f},"
                    f"roi={'local' if using_local_roi else 'full'},mode={MASK_MODE},th={th_text},{count_text}",
                    flush=True,
                )
            else:
                print(
                    f"B,{seq},0,lost,fps={fps_smooth:.1f},"
                    f"roi={'local' if using_local_roi else 'full'},mode={MASK_MODE},th={th_text},{count_text}",
                    flush=True,
                )

        # 14. Optional local display. Keep it off in --run mode.
        if show_windows:
            cv2.rectangle(view, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 255, 0), 1)
            if using_local_roi:
                cv2.rectangle(view, (search_x1, search_y1), (search_x2, search_y2), (255, 0, 255), 1)
            cv2.line(view, (CENTER_X, 0), (CENTER_X, FRAME_H - 1), (255, 0, 0), 1)
            cv2.line(view, (0, int(track_y_min)), (FRAME_W - 1, int(track_y_min)), (0, 255, 255), 1)
            cv2.line(view, (0, int(track_y_max)), (FRAME_W - 1, int(track_y_max)), (0, 255, 255), 1)

            if target_box is not None:
                x, y, w, h = target_box
                cx, cy = target_center
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(view, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                show_text = f"TRACK x={cx:.0f} y={cy:.0f} fps={fps_smooth:.1f}"
                show_color = (0, 255, 0)
            else:
                show_text = f"LOST fps={fps_smooth:.1f}"
                show_color = (0, 0, 255)

            cv2.putText(view, show_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, show_color, 2)
            cv2.imshow("ball_vision", view)
            cv2.imshow("roi_gray", roi_gray)
            cv2.imshow("debug_view", debug_view)
            cv2.imshow("mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        elif DEBUG_MODE:
            cv2.waitKey(1)

        seq += 1

    cap.release()
    if uart is not None:
        uart.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
