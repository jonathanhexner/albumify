"""Diagnostic to understand what's happening in the notebook."""

print("="*60)
print("JUPYTER NOTEBOOK DIAGNOSTIC")
print("="*60)

# Check 1: Can we import?
try:
    from face_cluster import FaceRecord
    print("\n[OK] FaceRecord imported successfully")
except Exception as e:
    print(f"\n[ERROR] Import failed: {e}")
    exit(1)

# Check 2: What fields does it have?
# spec-033 P-C C-2: FaceRecord is a Pydantic BaseModel; introspect via model_fields.
fields = list(FaceRecord.model_fields.keys())
print(f"\nFaceRecord has {len(fields)} fields:")
for f in fields:
    print(f"  - {f}")

# Check 3: Are the new fields there?
if 'image_path' in fields and 'face_index' in fields:
    print("\n[OK] New fields present: image_path, face_index")
else:
    print("\n[ERROR] New fields MISSING!")
    print("  Expected: image_path, face_index")
    print(f"  Got: {fields}")
    exit(1)

# Check 4: Can we create an instance with new fields?
import numpy as np

try:
    face = FaceRecord(
        face_id=0,
        image_id="test.jpg",
        bbox=(0, 0, 112, 112),
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
        embedding=np.zeros(512),
        embedding_normalized=np.zeros(512),
        pose=(0.0, 0.0, 0.0),
        blur_score=100.0,
        area=12544.0,
        is_core=False,
        image_path="/path/to/test.jpg",
        face_index=0
    )
    print("\n[OK] FaceRecord created with new fields")
    print(f"  image_path: {face.image_path}")
    print(f"  face_index: {face.face_index}")
except TypeError as e:
    print(f"\n[ERROR] Creation failed: {e}")
    exit(1)

# Check 5: Module location
import face_cluster.types
print(f"\nModule location: {face_cluster.types.__file__}")

print("\n" + "="*60)
print("ALL CHECKS PASSED")
print("="*60)
print("\nIf you're still getting errors in the notebook:")
print("1. Make sure you ran this AFTER restarting kernel")
print("2. Check your notebook is using the correct Python environment")
print("3. Share the full error traceback")
