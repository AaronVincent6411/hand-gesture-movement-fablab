import asyncio
import websockets
from main import HandTracker

# --- CONFIGURATION ---
# Use localhost for testing with mock_server.py
ESP32_IP = "192.168.0.100" 
WS_PORT = 80
WS_URL = f"ws://{ESP32_IP}:{WS_PORT}/ws"

async def send_loop(ws, tracker):
    """Reads camera and sends to WebSocket continuously."""
    print("[*] Streaming Hand Data...")
    
    import time
    last_sent = 0
    
    try:
        while True:
            # 1. Get One Frame of Data
            data = tracker.get_coordinates()

            print(data)

            if data is None: # User pressed ESC
                print("\n[!] Exiting...")
                break
            z = 0
            # Only send data every 1 second
            # if time.time() - last_sent >= 1.0:
            z, r, p, y = data

            if z != 0:
                await ws.send("ANIMSTOP")
                await asyncio.sleep(0.5)
                
                if p == 60 or p == -60:
                    p = 0
                # 2. Format Command
                cmd = f"MOV POSCURVANG {z} {r} {p} {y}"
                
                # 3. Send to WebSocket
                await ws.send(cmd + "\n")
                await asyncio.sleep(0.5)
                    
                # last_sent = time.time()



                
            
            # 4. Pace the loop (approx 30 FPS)
            # This is crucial to let the asyncio loop handle network traffic
            await asyncio.sleep(0.03)

    except Exception as e:
        print(f"[!] Error in loop: {e}")

async def main():
    tracker = HandTracker()
    print(f"[*] Connecting to {WS_URL}...")
    
    try:
        async with websockets.connect(WS_URL, ping_interval=None) as ws:
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