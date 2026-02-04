"""Quick diagnostic to check if Qdrant has data and if search works."""
from config import get_qdrant_client
from fastembed import TextEmbedding

client = get_qdrant_client()
text_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("=" * 50)
print("AEGIS DIAGNOSTIC")
print("=" * 50)

# Check collections
for coll in ['visual_memory', 'audio_memory', 'tactical_memory']:
    if client.collection_exists(coll):
        count = client.count(collection_name=coll).count
        print(f"\n📦 {coll}: {count} records")
        
        if count > 0:
            # Get sample
            sample = client.scroll(collection_name=coll, limit=2, with_payload=True)[0]
            for p in sample:
                disaster = p.payload.get("detected_disaster", "N/A")
                loc = p.payload.get("location")
                if isinstance(loc, dict):
                    loc_name = loc.get("name", str(loc))
                else:
                    loc_name = str(loc)
                print(f"   - {disaster} | {loc_name[:30]}")
    else:
        print(f"\n❌ {coll}: DOES NOT EXIST")

# Test search
print("\n" + "=" * 50)
print("TESTING SEARCH: 'flood'")
print("=" * 50)

vec = list(text_model.embed(["flood"]))[0].tolist()

for coll in ['tactical_memory', 'audio_memory']:
    if client.collection_exists(coll):
        try:
            res = client.query_points(collection_name=coll, query=vec, limit=3, with_payload=True)
            print(f"\n{coll}:")
            for hit in res.points:
                print(f"   Score: {hit.score:.3f} | {hit.payload.get('source', 'Unknown')}")
        except Exception as e:
            print(f"   ERROR: {e}")

print("\n✅ Diagnostic complete!")
