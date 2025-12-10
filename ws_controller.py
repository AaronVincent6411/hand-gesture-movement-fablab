import asyncio
import websockets
from main import HandTracker

# --- CONFIGURATION ---
# Use localhost for testing with mock_server.py
ESP32_IP = "localhost" 
WS_PORT = 8765
WS_URL = f"ws://{ESP32_IP}:{WS_PORT}/ws"

async def send_loop(ws, tracker):
    """Reads camera and sends to WebSocket continuously."""
    print("[*] Streaming Hand Data...")
    
    try:
        while True:
            # 1. Get One Frame of Data
            data = tracker.get_coordinates()
            if data is None: # User pressed ESC
                print("\n[!] Exiting...")
                break

            z, r, p, y = data

            # 2. Format Command
            cmd = f"MOV POSCURVANG {z} {r} {p} {y}"
            
            # 3. Send to WebSocket
            await ws.send(cmd)
            
            # 4. Pace the loop (approx 30 FPS)
            # This is crucial to let the asyncio loop handle network traffic
            await asyncio.sleep(0.03)

    except Exception as e:
        print(f"[!] Error in loop: {e}")

async def main():
    tracker = HandTracker()
    print(f"[*] Connecting to {WS_URL}...")
    
    try:
        async with websockets.connect(WS_URL) as ws:
            print("[+] Connected!")
            
            # Wait for welcome message
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2)
                print(f"[RX] {msg}")
            except asyncio.TimeoutError:
                pass

            # Start streaming
            await send_loop(ws, tracker)

    except ConnectionRefusedError:
        print("[-] Connection Refused. Is 'mock_server.py' running?")
    except Exception as e:
        print(f"[-] Error: {e}")
    finally:
        tracker.release()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass