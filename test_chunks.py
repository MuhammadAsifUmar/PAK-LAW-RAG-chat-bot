from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

all_points = []
next_offset = None
while True:
    results, next_offset = qdrant_client.scroll(
        collection_name="ppc_crpc_legal_docs",
        limit=500,
        offset=next_offset,
        with_payload=True
    )
    all_points.extend(results)
    if next_offset is None:
        break

for point in all_points:
    meta = point.payload.get('metadata', {})
    if meta.get('section') == '378' and meta.get('book') == 'PPC' and meta.get('level') == 1:
        print(point.payload.get('page_content')[:300])