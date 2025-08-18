import os
import cv2
import uuid
import time
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
from insightface.app import FaceAnalysis

# Initialize Qdrant client with your API key
client = QdrantClient(
    url="http://localhost:6333", 
    api_key="123456",  # Use your actual API key,
    prefer_grpc=True
)

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0)

def get_face_app():
    """Initialize the face analysis app"""
    
    return app

def create_qdrant_collection():
    """Create Qdrant collection for face embeddings"""
    try:
        # Check if collection already exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if "face_embeddings" not in collection_names:
            client.create_collection(
                collection_name="face_embeddings",
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        always_ram=True,
                    ),
                ),
            )
            print("✅ Created 'face_embeddings' collection")
        else:
            print("✅ Collection 'face_embeddings' already exists")
            
        return True
    except Exception as e:
        print(f"❌ Failed to create collection: {e}")
        return False

def store_gdc_profile_embeddings(profile_folder, batch_size=100):
    """
    Optimized version with configurable batch size and better memory management
    """
    app = get_face_app()
    
    if not create_qdrant_collection():
        print("Failed to create/verify collection. Exiting.")
        return
    
    total_start_time = time.time()
    processed_files = []
    batch_points = []
    
    print(f"🚀 Starting OPTIMIZED profile embedding conversion")
    print(f"📁 Folder: {profile_folder} | Batch size: {batch_size}")
    print("=" * 90)
    
    # First pass: collect all valid files
    valid_files = [f for f in os.listdir(profile_folder) 
                   if f.lower().endswith(('.jpg', '.png'))]
    
    print(f"📋 Found {len(valid_files)} image files to process")
    
    for i, fname in enumerate(valid_files, 1):
        img_path = os.path.join(profile_folder, fname)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ [{i:3d}/{len(valid_files)}] {fname} - Could not read image")
            continue
        
        faces = app.get(img)
        if not faces:
            print(f"❌ [{i:3d}/{len(valid_files)}] {fname} - No face detected")
            continue
        
        # Process largest face
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        point_id = str(uuid.uuid4())
        person_name = os.path.splitext(fname)[0]
        
        point = PointStruct(
            id=point_id,
            vector=largest_face.embedding.tolist(),
            payload={
                "person_name": person_name,
                "filename": fname,
                "bbox": largest_face.bbox.tolist(),
                "image_path": img_path,
                "confidence": float(largest_face.det_score) if hasattr(largest_face, 'det_score') else 1.0
            }
        )
        
        batch_points.append(point)
        print(f"✅ [{i:3d}/{len(valid_files)}] {fname} - Prepared for batch")
        
        # Insert batch when full
        if len(batch_points) >= batch_size:
            db_start = time.time()
            try:
                client.upsert(collection_name="face_embeddings", points=batch_points)
                db_time = time.time() - db_start
                print(f"💾 Batch {len(processed_files)//batch_size + 1}: Inserted {len(batch_points)} embeddings in {db_time:.3f}s")
                processed_files.extend([p.payload["filename"] for p in batch_points])
                batch_points = []
                
            except Exception as e:
                print(f"❌ Batch insert failed: {e}")
                batch_points = []
    
    # Final batch
    if batch_points:
        db_start = time.time()
        try:
            client.upsert(collection_name="face_embeddings", points=batch_points)
            db_time = time.time() - db_start
            print(f"💾 Final batch: Inserted {len(batch_points)} embeddings in {db_time:.3f}s")
            processed_files.extend([p.payload["filename"] for p in batch_points])
            
        except Exception as e:
            print(f"❌ Final batch insert failed: {e}")
    
    total_time = time.time() - total_start_time
    
    print("=" * 90)
    print(f"🎯 OPTIMIZED SUMMARY")
    print("=" * 90)
    print(f"Successfully processed: {len(processed_files)}/{len(valid_files)} files")
    print(f"Total time: {total_time:.3f}s")
    print(f"Speed: {len(processed_files)/total_time:.2f} embeddings/second")
    print(f"Batch size used: {batch_size}")

def search_similar_faces(query_image_path, limit=5, score_threshold=0.3):
    """
    Search for similar faces in the Qdrant database
    """
    search_start_time = time.time()
    
    app = get_face_app()
    
    # Load and process query image
    load_start = time.time()
    img = cv2.imread(query_image_path)
    if img is None:
        print(f"Could not read query image: {query_image_path}")
        return []
    load_time = time.time() - load_start
        
    # Face detection
    detection_start = time.time()
    faces = app.get(img)
    if not faces:
        print("No face detected in query image")
        return []
    detection_time = time.time() - detection_start
    
    # Use the largest face for query
    processing_start = time.time()
    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    query_embedding = largest_face.embedding.tolist()
    processing_time = time.time() - processing_start
    
    try:
        # Search in Qdrant
        db_search_start = time.time()
        search_results = client.query_points(
            collection_name="face_embeddings",
            query=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            
        ).points
        db_search_time = time.time() - db_search_start
        
        results = []
        for hit in search_results:
            results.append({
                "id": hit.id,
                "person_name": hit.payload.get("person_name"),
                "filename": hit.payload.get("filename"),
                "similarity_score": float(hit.score),
                "bbox": hit.payload.get("bbox"),
                "image_path": hit.payload.get("image_path"),
                "confidence": hit.payload.get("confidence")
            })
        
        total_search_time = time.time() - search_start_time
        
        # print(f"⏱️  Search Performance:")
        # print(f"   Load image: {load_time:.3f}s")
        # print(f"   Face detection: {detection_time:.3f}s") 
        # print(f"   Processing: {processing_time:.3f}s")
        # print(f"   Database search: {db_search_time:.3f}s")
        # print(f"   Total search time: {total_search_time:.3f}s")
        # print(f"   Found {len(results)} matches")
            
        return results
        
    except Exception as e:
        total_search_time = time.time() - search_start_time
        print(f"❌ Error searching faces: {e}")
        print(f"   Total time before error: {total_search_time:.3f}s")
        return []
        
    except Exception as e:
        print(f"❌ Error searching faces: {e}")
        return []

def get_collection_stats():
    """Get statistics about the face embeddings collection"""
    try:
        info = client.get_collection("face_embeddings")
        print(f"\n📊 Collection Statistics:")
        print(f"   • Total points: {info.points_count}")
        print(f"   • Indexed vectors: {info.indexed_vectors_count}")
        print(f"   • Status: {info.status}")
        return info
    except Exception as e:
        print(f"❌ Error getting collection stats: {e}")
        return None
    
def clear_all_collections():
    """
    Delete all collections from the Qdrant database
    """
    try:
        # Get all collections
        collections = client.get_collections()
        
        if not collections.collections:
            print("📭 No collections found to delete")
            return True
            
        print(f"🗑️  Found {len(collections.collections)} collection(s) to delete:")
        
        deleted_count = 0
        for collection in collections.collections:
            collection_name = collection.name
            try:
                # Delete the collection
                client.delete_collection(collection_name)
                print(f"   ✅ Deleted collection: {collection_name}")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Failed to delete collection {collection_name}: {e}")
        
        print(f"\n🎉 Successfully deleted {deleted_count} collection(s)")
        return True
        
    except Exception as e:
        print(f"❌ Error clearing collections: {e}")
        return False

def clear_face_embeddings_collection():
    """
    Clear only the face_embeddings collection (delete all points but keep collection structure)
    """
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if "face_embeddings" not in collection_names:
            print("📭 Collection 'face_embeddings' does not exist")
            return True
        
        # Get collection info before deletion
        info = client.get_collection("face_embeddings")
        points_count = info.points_count
        
        if points_count == 0:
            print("📭 Collection 'face_embeddings' is already empty")
            return True
        
        # Delete all points from the collection
        client.delete(
            collection_name="face_embeddings",
            points_selector=True  # This deletes all points
        )
        
        print(f"🗑️  Cleared {points_count} points from 'face_embeddings' collection")
        print("✅ Collection structure preserved, all embeddings removed")
        return True
        
    except Exception as e:
        print(f"❌ Error clearing face_embeddings collection: {e}")
        return False

def reset_database():
    """
    Complete database reset - delete all collections and recreate face_embeddings
    """
    print("🔄 Resetting Qdrant database...")
    
    # Clear all collections
    if clear_all_collections():
        # Recreate the face_embeddings collection
        if create_qdrant_collection():
            print("✅ Database reset complete - ready for new embeddings")
            return True
        else:
            print("❌ Failed to recreate face_embeddings collection")
            return False
    else:
        print("❌ Failed to clear collections")
        return False

def list_all_points(collection_name="face_embeddings", limit=None, offset=0):
    """
    List all points in the collection with their metadata
    
    Args:
        collection_name: Name of the collection to query
        limit: Maximum number of points to retrieve (None for all)
        offset: Number of points to skip
    
    Returns:
        List of points with their metadata
    """
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            print(f"📭 Collection '{collection_name}' does not exist")
            return []
        
        # Get collection info first
        info = client.get_collection(collection_name)
        total_points = info.points_count
        
        if total_points == 0:
            print(f"📭 Collection '{collection_name}' is empty")
            return []
        
        # Set limit to total points if not specified
        if limit is None:
            limit = total_points
        
        print(f"📋 Listing points from collection '{collection_name}'")
        print(f"   Total points in collection: {total_points}")
        print(f"   Retrieving: {min(limit, total_points - offset)} points (offset: {offset})")
        print("=" * 80)
        
        # Scroll through all points
        points = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False  # Set to True if you want to see the vectors too
        )
        
        point_list = []
        
        for i, point in enumerate(points[0], 1):
            point_data = {
                "id": point.id,
                "payload": point.payload
            }
            point_list.append(point_data)
            
            # Print point information
            payload = point.payload
            print(f"{i:3d}. ID: {point.id}")
            print(f"     Person: {payload.get('person_name', 'Unknown')}")
            print(f"     File: {payload.get('filename', 'Unknown')}")
            print(f"     Confidence: {payload.get('confidence', 'N/A')}")
            print(f"     BBox: {payload.get('bbox', 'N/A')}")
            print(f"     Path: {payload.get('image_path', 'N/A')}")
            print("-" * 80)
        
        print(f"📊 Retrieved {len(point_list)} points")
        return point_list
        
    except Exception as e:
        print(f"❌ Error listing points: {e}")
        return []
# Example usage
if __name__ == "__main__":
    
    # Test connection first
    try:
        collections = client.get_collections()
        print(f"✅ Connected to Qdrant! Found {len(collections.collections)} collections")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        exit(1)
    
    # Example: Store embeddings from a profile folder
    profile_folder = "test"  # Update this path
    
    if os.path.exists(profile_folder):
        # clear_all_collections()
        
        # print(f"🔄 Processing images from: {profile_folder}")
        # store_gdc_profile_embeddings(profile_folder)
        
        # Get collection statistics
        # get_collection_stats()
        
        # list_all_points()
        
        # Example: Search for similar faces
        query_image = "test3.jpg"  # Update this path
        if os.path.exists(query_image):
            print(f"\n🔍 Searching for similar faces to: {query_image}")
            similar_faces = search_similar_faces(query_image, limit=1, score_threshold=0.6)
            
            if similar_faces:
                print("🎯 Found similar faces:")
                for i, face in enumerate(similar_faces, 1):
                    print(face)
                    print(f"   {i}. {face['person_name']} (Score: {1-face['similarity_score']})")
            else:
                print("😞 No similar faces found")
    else:
        print(f"❌ Profile folder not found: {profile_folder}")
        print("Please update the 'profile_folder' path in the script")