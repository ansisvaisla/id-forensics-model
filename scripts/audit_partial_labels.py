"""Audit corners dataset: find partial images with bad (full-frame) labels."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
tasks = json.loads((PROJECT_ROOT / "data/labels/label_studio_export.json").read_text(encoding="utf-8"))

partial_fullframe = []
partial_realcorners = []
total_partial = 0

for t in tasks:
    anns = [a for a in (t.get("annotations") or []) if not a.get("was_cancelled")]
    if not anns:
        continue
    quality: set = set()
    polygon = None
    for ann in anns:
        for r in ann.get("result") or []:
            if r.get("type") == "choices" and r.get("from_name") == "quality":
                quality.update(r["value"].get("choices", []))
            if r.get("type") == "polygonlabels" and len(r["value"]["points"]) == 4:
                polygon = r["value"]["points"]
    if "partial" not in quality or polygon is None:
        continue
    total_partial += 1
    fu = t.get("file_upload", "")
    stem = fu.split("-", 1)[-1].replace(".jpg", "").replace(".jpeg", "")
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    # Full-frame placeholder: all corners near image border
    is_fullframe = (min(xs) < 5 and max(xs) > 90 and min(ys) < 10 and max(ys) > 85)
    entry = {"stem": stem, "xs": [round(x, 1) for x in xs], "ys": [round(y, 1) for y in ys]}
    if is_fullframe:
        partial_fullframe.append(entry)
    else:
        partial_realcorners.append(entry)

print(f"Total partial with polygon: {total_partial}")
print(f"  Full-frame (bad labels):  {len(partial_fullframe)}")
print(f"  Real corners (keep):      {len(partial_realcorners)}")

print("\n--- BAD (full-frame) ---")
for e in partial_fullframe:
    print(f"  {e['stem']}")

print("\n--- GOOD (real corners, keep) ---")
for e in partial_realcorners[:10]:
    print(f"  {e['stem']}  xs={e['xs']} ys={e['ys']}")

bad_stems = set(e["stem"] for e in partial_fullframe)
out = PROJECT_ROOT / "data" / "eval" / "partial_bad_labels.txt"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(sorted(bad_stems)), encoding="utf-8")
print(f"\nBad stems saved: {out}")
