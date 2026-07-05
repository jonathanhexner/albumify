"""ut for occlusion review data layer (corrections round-trip, disagreements)."""
import csv, os
from app.occlusion_review import data as D


def _mini_dataset(tmp_path):
    root = str(tmp_path)
    with open(os.path.join(root, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","label","level","source_dataset","source_path","sha1","split","hard_negative","notes","group_id"])
        w.writerow(["a.jpg","1","","occl","x","s1","train","False","","s1"])
        w.writerow(["b.jpg","0","","alb","y","s2","train","False","","s2"])
        w.writerow(["c.jpg","0","","alb","z","s3","test","False","","s3"])
    with open(os.path.join(root, "haiku_labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","user_label","haiku_occluded","haiku_level","haiku_reason"])
        w.writerow(["a.jpg","1","0","0","clean says haiku"])   # miss
        w.writerow(["b.jpg","0","1","2","blob lower left"])    # flag
        w.writerow(["c.jpg","0","0","0","clean"])              # agree
    return root


def test_disagreements_flags_first(tmp_path):
    root = _mini_dataset(tmp_path)
    recs = D.load_all(root)
    work = D.disagreements(recs)
    assert [r["id"] for r in work] == ["b.jpg", "a.jpg"]  # flagged negative before missed positive


def test_corrections_roundtrip_and_effective_label(tmp_path):
    root = _mini_dataset(tmp_path)
    recs = {r["id"]: r for r in D.load_all(root)}
    D.save_correction("b.jpg", "0", "occluded_l2", "real finger", root)
    D.save_correction("a.jpg", "1", "clean", "", root)
    D.save_correction("c.jpg", "0", "foreground_object", "branch", root)
    corr = D.load_corrections(root)
    assert D.effective_label(recs["b.jpg"], corr) == "1"   # promoted to positive
    assert D.effective_label(recs["a.jpg"], corr) == "0"   # demoted to clean
    assert D.effective_label(recs["c.jpg"], corr) == "fg"  # its own class
