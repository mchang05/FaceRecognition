from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import time

client = QdrantClient(url="http://localhost:6334", api_key="123456")

# client.create_collection(
#     collection_name="test_collection2",
#     vectors_config=VectorParams(size=4, distance=Distance.DOT),
# )

# operation_info = client.upsert(
#     collection_name="test_collection2",
#     wait=True,
#     points=[
#         PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
#         PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
#         PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
#         PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
#         PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
#         PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
#     ],
# )


for i in range(5):
    db_search_start = time.time()

    search_result = client.query_points(
        collection_name="test_collection2",
        query=[0.2, 0.1, 0.9, 0.7],
        with_payload=False,
        limit=3
    ).points

    db_search_time = time.time() - db_search_start
    print(f"Loop {i+1}: {db_search_time}")
    print(search_result)

# print(operation_info)


# def clear_all_collections():
#     """
#     Delete all collections from the Qdrant database
#     """
#     try:
#         # Get all collections
#         collections = client.get_collections()
        
#         if not collections.collections:
#             print("📭 No collections found to delete")
#             return True
            
#         print(f"🗑️  Found {len(collections.collections)} collection(s) to delete:")
        
#         deleted_count = 0
#         for collection in collections.collections:
#             collection_name = collection.name
#             try:
#                 # Delete the collection
#                 client.delete_collection(collection_name)
#                 print(f"   ✅ Deleted collection: {collection_name}")
#                 deleted_count += 1
#             except Exception as e:
#                 print(f"   ❌ Failed to delete collection {collection_name}: {e}")
        
#         print(f"\n🎉 Successfully deleted {deleted_count} collection(s)")
#         return True
    
# clear_all_collections()
    