import requests
import time
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_e2e_flow():
    print("Starting End-to-End User Flow Test...")
    print("-" * 50)
    
    # 1. Fetch initial recommendations
    print("1. Fetching initial recommendations...")
    res = requests.post(f"{BASE_URL}/recommend", json={"user_id": 1, "limit": 6})
    if res.status_code != 200:
        print(f"FAILED to fetch recommendations: {res.text}")
        return
    tracks = res.json()["tracks"]
    print(f"SUCCESS: Received {len(tracks)} tracks:")
    for t in tracks:
        print(f"   - {t['title']} (Energy: {t['energy']:.2f})")
    
    track_id = tracks[0]["id"]
    print("-" * 50)
    
    # 2. Check current Vibe State
    print("2. Checking initial Redis Vibe State...")
    redis_res = requests.get(f"{BASE_URL}/admin/redis")
    initial_vibe = None
    for r in redis_res.json():
        if r["key"] == "vibe:1":
            initial_vibe = r["data"]
            print(f"   Current Vibe for User 1: {initial_vibe}")
            
    print("-" * 50)
    
    # 3. Simulate UI 'Play' Event
    print(f"3. Simulating 'play' event on Track {track_id} (UI interaction)...")
    payload = {
        "user_id": 1,
        "track_id": track_id,
        "event_type": "play",
        "duration_ms": 1500,
        "listened_percentage": 0.05,
        "is_rapid_skip": False,
        "context": {}
    }
    telemetry_res = requests.post(f"{BASE_URL}/telemetry", json=payload)
    print(f"SUCCESS: Telemetry Event queued: {telemetry_res.json()}")
    
    # Wait for async Redis update
    time.sleep(1)
    
    print("-" * 50)
    
    # 4. Simulate UI 'Save' Event (Aggressive Vibe Shift)
    print(f"4. Simulating 'save' event on Track {track_id} (Aggressive Vibe Shift)...")
    payload["event_type"] = "save"
    requests.post(f"{BASE_URL}/telemetry", json=payload)
    
    # Wait for async Redis update
    time.sleep(1)
    
    print("-" * 50)
    
    # 5. Check New Vibe State
    print("5. Validating Vibe State changes in Redis...")
    redis_res = requests.get(f"{BASE_URL}/admin/redis")
    new_vibe = None
    for r in redis_res.json():
        if r["key"] == "vibe:1":
            new_vibe = r["data"]
            print(f"   New Vibe for User 1: {new_vibe}")
    
    if str(initial_vibe) != str(new_vibe):
        print("SUCCESS: Redis Vibe State successfully updated!")
    else:
        print("FAILED: Redis Vibe State did not change.")

    print("-" * 50)
    
    # 6. Fetch next round of recommendations
    print("6. Fetching NEW recommendations based on updated vibe...")
    res = requests.post(f"{BASE_URL}/recommend", json={"user_id": 1, "limit": 6})
    tracks = res.json()["tracks"]
    print(f"SUCCESS: Received 6 new tracks:")
    for t in tracks:
        print(f"   - {t['title']} (Energy: {t['energy']:.2f})")
    print("End-to-End Flow Complete.")

if __name__ == "__main__":
    test_e2e_flow()
