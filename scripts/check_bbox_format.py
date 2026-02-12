"""Check bbox format in InsightFace detection cache."""

import json
import sqlite3

db_path = r"C:\Users\Jonathan Hexner\.sim_bench\sim_bench.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get InsightFace detection data
cursor.execute(
    "SELECT image_path, data_blob FROM universal_cache WHERE feature_type = 'insightface_detection' LIMIT 3"
)
rows = cursor.fetchall()

print("=== INSIGHTFACE DETECTION BBOX FORMAT ===\n")

for row in rows:
    path = row[0]
    data = row[1]

    # Deserialize - try JSON first, then pickle
    if isinstance(data, bytes):
        try:
            detection = json.loads(data.decode('utf-8'))
        except:
            import pickle
            detection = pickle.loads(data)
    else:
        detection = json.loads(data)

    print(f"Image: ...{path[-50:]}")
    print(f"Detection type: {type(detection)}")

    if isinstance(detection, dict):
        print(f"Keys: {list(detection.keys())}")
        faces = detection.get("faces", [])
        print(f"Number of faces: {len(faces)}")

        for i, face in enumerate(faces[:2]):
            print(f"\n  Face {i}:")
            if isinstance(face, dict):
                for key, value in face.items():
                    print(f"    {key}: {value}")
            else:
                print(f"    {face}")
    print("\n" + "=" * 60 + "\n")

conn.close()
