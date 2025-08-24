import os
import time
import json
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List

from qdrant import (
    create_collection,
    get_embedding,
    store_db,
    list_all_points,
    clear_all_collections,
)
from qdrant_client.models import PointStruct

# ----------------------------
# Configuration
# ----------------------------
API_URL = "https://aia-eventapp.com/events/y25/aia_lf/api/"
ORIGIN_HEADER = {"Origin": "https://aia-eventapp.com"}

COLLECTION_NAME = "aia_lf_2025"
TEMP_PHOTO_DIR = "temp_photos"
REQUEST_TIMEOUT = 30  # seconds
BATCH_SIZE = 100
DEFAULT_SINCE = "1970-01-01 00:00:00"


# ----------------------------
# Utilities
# ----------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def chunked(seq: List[Any], size: int) -> List[List[Any]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


# ----------------------------
# Network/API
# ----------------------------
def get_guest_photo_qr_list(event_id: int, time_update: Optional[str] = None) -> Dict[str, Any]:
    """
    Get Guest Photo and QrCode from AIA Event API
    """
    payload = {
        "action": "at_guest_photo_qr_list",
        "event_id": event_id,
        "time_update": time_update or "",
    }

    try:
        print(f"🔄 Calling API with payload: {payload}")
        response = requests.post(
            API_URL,
            json=payload,
            headers=ORIGIN_HEADER,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()
        print("✅ API call successful")
        return result

    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return {"error": "Request timed out", "status": "timeout"}

    except requests.exceptions.ConnectionError:
        print("❌ Connection error")
        return {"error": "Connection error", "status": "connection_error"}

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error: {e}")
        return {"error": f"HTTP error: {e}", "status": "http_error", "status_code": response.status_code}

    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return {"error": f"Request error: {e}", "status": "request_error"}

    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return {"error": f"Invalid JSON response: {e}", "status": "json_error", "raw_response": response.text}

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {"error": f"Unexpected error: {e}", "status": "unknown_error"}


# ----------------------------
# Media and embedding
# ----------------------------
def download_photo(photo_path: str, save_basename: str) -> Optional[str]:
    """
    Download photo from remote path to temp_photos/<save_basename>.jpg
    Returns local file path or None on failure.
    """
    os.makedirs(TEMP_PHOTO_DIR, exist_ok=True)
    photo_url = f"https://space.klicker.app{photo_path}"
    photo_filename = os.path.join(TEMP_PHOTO_DIR, f"{save_basename}.jpg")

    try:
        photo_response = requests.get(photo_url, timeout=15)
        photo_response.raise_for_status()
        with open(photo_filename, "wb") as f:
            f.write(photo_response.content)
        print(f"      ⬇️  Downloaded to {photo_filename}")
        return photo_filename
    except Exception as e:
        print(f"      ⚠️  Failed to download photo: {e}")
        return None


def point_from_item(item: Dict[str, Any], timestamp: str) -> Optional[PointStruct]:
    """
    Build a PointStruct from an API item by downloading the photo and extracting an embedding.
    """
    photo_filename = download_photo(item["ProfilePhoto"], str(item["Name"]))
    if not photo_filename:
        return None

    largest_face = get_embedding(photo_filename)
    if not largest_face:
        print(f"      ⚠️  No face detected for {item.get('Name', 'Unknown')}")
        return None

    return PointStruct(
        id=item["GuestId"],
        vector=largest_face.embedding.tolist(),
        payload={
            "name": item["Name"],
            "bbox": largest_face.bbox.tolist(),
            "photo": item["ProfilePhoto"],
            "email": item.get("Email", ""),
            "qr_code": item.get("QrCode", ""),
            "agent_id": item.get("AgentId", ""),
            "date": timestamp,
        },
    )


def items_to_points(items: List[Dict[str, Any]], timestamp: str) -> List[PointStruct]:
    points: List[PointStruct] = []
    for i, item in enumerate(items, 1):
        print(f"   {i}. {item.get('Name', 'Unknown')} (GuestId: {item.get('GuestId')})")
        p = point_from_item(item, timestamp)
        if p:
            points.append(p)
    return points


# ----------------------------
# Qdrant helpers
# ----------------------------
def ensure_collection(collection_name: str) -> None:
    """
    Create the collection if it does not exist.
    """
    try:
        create_collection(collection_name)
        print(f"✅ Created collection '{collection_name}'")
    except Exception as e:
        msg = str(e).lower()
        if "already exists" in msg or "exists" in msg or "409" in msg:
            print(f"ℹ️ Collection '{collection_name}' already exists")
        else:
            print(f"❌ Failed to create collection '{collection_name}': {e}")

def update_points_in_collection(collection_name: str, points: List[PointStruct]) -> None:
    """
    Upsert (update/insert) points in Qdrant in batches.
    """
    if not points:
        print("⚠️ No points to update")
        return

    total = 0
    for batch in chunked(points, BATCH_SIZE):
        store_db(collection_name, batch)  # store_db should upsert
        total += len(batch)
    print(f"✅ Upserted {len(points)} point(s) into '{collection_name}'")


def latest_date_in_collection(collection_name: str) -> Optional[str]:
    """
    Scan collection and return the latest 'date' payload string, if any.
    """
    latest: Optional[str] = None
    latest_item = None
    points = list_all_points(collection_name)
    for p in points:
        date_str = (p.get("payload") or {}).get("date")
        if date_str and (latest is None or date_str > latest):
            latest = date_str
            latest_item = p

    if latest_item:
        print(f"🕒 Latest item date in '{collection_name}': {latest}")
    else:
        print(f"⚠️ No items with a 'date' found in '{collection_name}'")
    return latest


# ----------------------------
# Initialization
# ----------------------------
def init_db(event_id: int, since: Optional[str] = None) -> None:
    """
    Seed collection with initial data from API.
    """
    ensure_collection(COLLECTION_NAME)
    result = get_guest_photo_qr_list(event_id=event_id, time_update=since or DEFAULT_SINCE)

    if "Result" not in result:
        print(f"⚠️ Unexpected response: {result}")
        return

    items = result.get("Result", [])
    print(f"📊 Found {len(items)} items")
    ts = now_str()
    points = items_to_points(items, ts)
    update_points_in_collection(COLLECTION_NAME, points)


# ----------------------------
# Monitoring loop
# ----------------------------
def run_continuous_monitoring(event_id: int, interval_seconds: int = 2) -> None:
    """
    Run continuous monitoring of the API and upsert into Qdrant.
    """
    ensure_collection(COLLECTION_NAME)
    print(f"🚀 Starting continuous monitoring for event {event_id}")
    print(f"📡 Checking every {interval_seconds} seconds")
    print("Press Ctrl+C to stop\n")

    call_count = 0
    latest_date = latest_date_in_collection(COLLECTION_NAME) or DEFAULT_SINCE

    try:
        while True:
            call_count += 1
            print("\n" + "=" * 60)
            print(f"📞 API Call #{call_count} since: {latest_date}")
            print("=" * 60)

            result = get_guest_photo_qr_list(event_id=event_id, time_update=latest_date)

            if "Result" in result:
                items = result.get("Result", [])
                print(f"📊 Found {len(items)} items")
                ts = now_str()
                points = items_to_points(items, ts)
                update_points_in_collection(COLLECTION_NAME, points)
                # Advance the watermark after processing
                latest_date = ts
            else:
                print(f"⚠️ Unexpected response: {result}")

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
        print(f"📊 Total API calls made: {call_count}")
    except Exception as e:
        print(f"\n❌ Monitoring stopped due to error: {e}")
        print(f"📊 Total API calls made: {call_count}")


# ----------------------------
# Script entry
# ----------------------------
if __name__ == "__main__":
    # Example one-time setup:
    # clear_all_collections()
    # init_db(event_id=29, since=DEFAULT_SINCE)

    # Run continuous monitoring
    run_continuous_monitoring(event_id=29, interval_seconds=10)

