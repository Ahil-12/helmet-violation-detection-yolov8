# helmet_tracker.py
# Helmet Violation Detection and Tracking System

from ultralytics import YOLO
import cv2
import pandas as pd
import time
import os

from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "best.pt"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)

print("Model classes:", model.names)


# ============================================================
# SELECT VIDEO
# ============================================================

video_name = input("Enter video name (with extension): ").strip()

VIDEO_PATH = os.path.join("videos", video_name)

base_name = os.path.splitext(video_name)[0]

print("Opening:", VIDEO_PATH)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO_PATH}")


# ============================================================
# VIDEO PROPERTIES
# ============================================================

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)

if fps_video <= 0:
    fps_video = 30


# ============================================================
# OUTPUT VIDEO
# ============================================================

output_video_path = os.path.join(
    OUTPUT_DIR,
    f"{base_name}_output.mp4"
)

writer = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_video,
    (width, height)
)


# ============================================================
# TRACKING VARIABLES
# ============================================================

# All unique rider IDs detected during the video
all_ids = set()

# IDs currently/last classified as wearing a helmet
helmet_ids = set()

# IDs currently/last classified as without a helmet
violation_ids = set()

# Store detection information for CSV
rows = []

frames = 0

start_time = time.time()
previous_time = start_time


# ============================================================
# MAIN VIDEO LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frames += 1

    # --------------------------------------------------------
    # YOLO DETECTION + BYTE TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.15,
        imgsz=1280,
        verbose=False
    )

    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        if result.boxes.id is None:
            continue

        for box, track_id_tensor in zip(
            result.boxes,
            result.boxes.id
        ):

            # Get tracking ID
            track_id = int(track_id_tensor.item())

            # Get class
            cls = int(box.cls[0])

            # Get confidence
            confidence = float(box.conf[0])

            # Get class label
            label = model.names[cls]

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # ------------------------------------------------
            # REGISTER UNIQUE RIDER
            # ------------------------------------------------

            all_ids.add(track_id)

            # ------------------------------------------------
            # CLASSIFICATION
            # ------------------------------------------------

            if label == "With Helmet":

                helmet_ids.add(track_id)

                # If previously classified as violation,
                # update its latest status.
                violation_ids.discard(track_id)

                color = (0, 255, 0)
                status = "Helmet"

            elif label == "Without Helmet":

                violation_ids.add(track_id)

                # If previously classified as helmet,
                # update its latest status.
                helmet_ids.discard(track_id)

                color = (0, 0, 255)
                status = "No Helmet"

            else:
                # Ignore unexpected classes
                continue

            # ------------------------------------------------
            # SAVE DETECTION DATA
            # ------------------------------------------------

            rows.append({
                "Track ID": track_id,
                "Status": status,
                "Confidence": round(confidence, 3),
                "Frame": frames
            })

            # ------------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # ------------------------------------------------
            # DRAW LABEL
            # ------------------------------------------------

            text = (
                f"ID:{track_id} "
                f"{label} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # ========================================================
    # CURRENT FRAME COUNTS
    # ========================================================

    current_helmet = 0
    current_violations = 0

    # Count detections visible in the current frame
    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "With Helmet":
                current_helmet += 1

            elif label == "Without Helmet":
                current_violations += 1


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    elapsed = current_time - previous_time

    if elapsed > 0:
        live_fps = 1 / elapsed
    else:
        live_fps = 0

    previous_time = current_time


    # ========================================================
    # DASHBOARD
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (430, 215),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        frame,
        "HELMET VIOLATION DETECTION",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Current Helmet     : {current_helmet}",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Current Violations : {current_violations}",
        (20, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total Riders       : {len(all_ids)}",
        (20, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Helmet Riders      : {len(helmet_ids)}",
        (20, 162),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"No Helmet Riders   : {len(violation_ids)}",
        (20, 192),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )


    # ========================================================
    # SAVE FRAME
    # ========================================================

    writer.write(frame)


    # ========================================================
    # DISPLAY FRAME
    # ========================================================

    cv2.imshow(
        "Helmet Violation Detection",
        frame
    )


    # ========================================================
    # EXIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:

        print("Stopping...")

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()
cv2.destroyAllWindows()


# ============================================================
# FINAL STATISTICS
# ============================================================

total_riders = len(all_ids)

helmet_riders = len(helmet_ids)

no_helmet_riders = len(violation_ids)


# ============================================================
# COMPLIANCE CALCULATION
# ============================================================

if total_riders > 0:

    compliance = (
        helmet_riders /
        total_riders
    ) * 100

    violation_rate = (
        no_helmet_riders /
        total_riders
    ) * 100

else:

    compliance = 0

    violation_rate = 0


# ============================================================
# CSV REPORT
# ============================================================

if rows:

    df = pd.DataFrame(rows)

    # Keep the latest known status for each tracked rider
    df = df.drop_duplicates(
        subset=["Track ID"],
        keep="last"
    )

else:

    df = pd.DataFrame(
        columns=[
            "Track ID",
            "Status",
            "Confidence",
            "Frame"
        ]
    )


csv_path = os.path.join(
    OUTPUT_DIR,
    f"{base_name}_report.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# PDF REPORT
# ============================================================

pdf_path = os.path.join(
    OUTPUT_DIR,
    f"{base_name}_report.pdf"
)

styles = getSampleStyleSheet()

doc = SimpleDocTemplate(
    pdf_path
)

table_data = [
    ["Metric", "Value"],
    ["Total Riders", total_riders],
    ["Helmet Riders", helmet_riders],
    ["No Helmet Riders", no_helmet_riders],
    ["Compliance (%)", f"{compliance:.2f}"],
    ["Violation Rate (%)", f"{violation_rate:.2f}"],
    ["Frames Processed", frames]
]

table = Table(table_data)

table.setStyle(
    TableStyle([
        (
            "GRID",
            (0, 0),
            (-1, -1),
            1,
            colors.black
        ),
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.grey
        ),
        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            colors.whitesmoke
        ),
        (
            "ALIGN",
            (0, 0),
            (-1, -1),
            "CENTER"
        )
    ])
)

title = Paragraph(
    "Helmet Violation Detection Report",
    styles["Title"]
)

doc.build([
    title,
    table
])


# ============================================================
# FINAL TERMINAL REPORT
# ============================================================

processing_time = time.time() - start_time

print()
print("==========================================")
print("       HELMET VIOLATION REPORT")
print("==========================================")

print(f"Video             : {video_name}")
print(f"Frames Processed  : {frames}")
print(f"Total Riders      : {total_riders}")
print(f"Helmet Riders     : {helmet_riders}")
print(f"No Helmet Riders  : {no_helmet_riders}")
print(f"Compliance        : {compliance:.2f}%")
print(f"Violation Rate    : {violation_rate:.2f}%")
print(f"Processing Time   : {processing_time:.2f} seconds")

print("------------------------------------------")

print(f"CSV Saved         : {csv_path}")
print(f"PDF Saved         : {pdf_path}")
print(f"Video Saved       : {output_video_path}")

print("==========================================")
print("             PROCESS COMPLETE")
print("==========================================")