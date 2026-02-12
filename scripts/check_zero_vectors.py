"""Check which faces have zero vector embeddings and why."""

import io
import json
import sqlite3

import numpy as np

db_path = r"C:\Users\Jonathan Hexner\.sim_bench\sim_bench.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get face embeddings
cursor.execute(
    "SELECT image_path, data_blob FROM universal_cache WHERE feature_type = 'face_embedding'"
)
embedding_rows = cursor.fetchall()

# Get InsightFace detections for comparison
cursor.execute(
    "SELECT image_path, data_blob FROM universal_cache WHERE feature_type = 'insightface_detection'"
)
detection_rows = cursor.fetchall()

# Build detection lookup
detections = {}
for row in detection_rows:
    path = row[0]
    data = json.loads(row[1].decode("utf-8") if isinstance(row[1], bytes) else row[1])
    detections[path] = data

print("=== ZERO VECTOR ANALYSIS ===\n")

zero_faces = []
valid_faces = []

for row in embedding_rows:
    cache_key = row[0]  # This is like "D:/path/image.jpg:face_0"
    data = row[1]
    arr = np.load(io.BytesIO(data), allow_pickle=False)

    # Parse cache key
    parts = cache_key.rsplit(":face_", 1)
    if len(parts) == 2:
        image_path = parts[0]
        face_index = int(parts[1])
    else:
        image_path = cache_key
        face_index = 0

    is_zero = np.allclose(arr, 0)

    if is_zero:
        zero_faces.append((image_path, face_index, cache_key))
    else:
        valid_faces.append((image_path, face_index, cache_key))

print(f"Total embeddings: {len(embedding_rows)}")
print(f"Zero vectors: {len(zero_faces)} ({100*len(zero_faces)/len(embedding_rows):.1f}%)")
print(f"Valid vectors: {len(valid_faces)} ({100*len(valid_faces)/len(embedding_rows):.1f}%)")

# Check a few zero vector faces - do they have valid detection data?
print("\n=== SAMPLE ZERO VECTOR FACES ===\n")
for image_path, face_index, cache_key in zero_faces[:5]:
    print(f"Cache key: {cache_key[-60:]}")

    # Check if we have detection data for this image
    if image_path in detections:
        det = detections[image_path]
        faces = det.get("faces", [])
        print(f"  Detection found: {len(faces)} faces")

        # Find this specific face
        for face in faces:
            if face.get("face_index") == face_index:
                bbox = face.get("bbox", {})
                print(f"  Face {face_index} bbox:")
                print(f"    Pixel: x_px={bbox.get('x_px')}, y_px={bbox.get('y_px')}, w_px={bbox.get('w_px')}, h_px={bbox.get('h_px')}")
                print(f"    Relative: x={bbox.get('x'):.3f}, y={bbox.get('y'):.3f}, w={bbox.get('w'):.3f}, h={bbox.get('h'):.3f}")
                break
        else:
            print(f"  Face {face_index} NOT FOUND in detection data!")
    else:
        print(f"  NO detection data for this image!")
    print()

# Check valid faces too
print("\n=== SAMPLE VALID VECTOR FACES ===\n")
for image_path, face_index, cache_key in valid_faces[:3]:
    print(f"Cache key: {cache_key[-60:]}")

    if image_path in detections:
        det = detections[image_path]
        faces = det.get("faces", [])
        print(f"  Detection found: {len(faces)} faces")

        for face in faces:
            if face.get("face_index") == face_index:
                bbox = face.get("bbox", {})
                print(f"  Face {face_index} bbox:")
                print(f"    Pixel: x_px={bbox.get('x_px')}, y_px={bbox.get('y_px')}, w_px={bbox.get('w_px')}, h_px={bbox.get('h_px')}")
                break
    print()

conn.close()
