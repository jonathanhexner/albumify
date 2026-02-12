"""Clear face embedding cache from database."""

import sqlite3

db_path = r"C:\Users\Jonathan Hexner\.sim_bench\sim_bench.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Delete face embeddings
cursor.execute("DELETE FROM universal_cache WHERE feature_type = 'face_embedding'")
deleted = cursor.rowcount
conn.commit()

print(f"Deleted {deleted} cached face embeddings")

# Also clear people table since it was built from bad embeddings
cursor.execute("DELETE FROM people")
deleted_people = cursor.rowcount
conn.commit()

print(f"Deleted {deleted_people} people records")

conn.close()
print("Done. Re-run the pipeline to regenerate embeddings and people clusters.")
