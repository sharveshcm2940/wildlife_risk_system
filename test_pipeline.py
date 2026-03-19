import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data.realtime_extractor import RealtimeDataExtractor

e = RealtimeDataExtractor()
df = e.extract_all(11.66, 76.63)  # Bandipur, Karnataka
print(f"Features extracted: {len(df.columns)}")
print()
for l in e.get_extraction_log():
    st = l["status"]
    tp = l["source_type"]
    nm = l["source"][:42]
    ms = l["response_ms"]
    ft = ", ".join(str(f)[:30] for f in l["features"][:3])
    print(f"  [{st:^8}] ({tp:^11}) {nm:<42} {ms:>6}ms | {ft}")
print()
s = e.get_extraction_summary()
print(f"Total sources: {s['total_sources']} | Success: {s['success_count']} | Fallback: {s['fallback_count']} | Time: {s['total_time_ms']}ms")
