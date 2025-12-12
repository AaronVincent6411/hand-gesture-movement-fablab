import asyncio
import websockets
from datetime import datetime

# Configuration
HOST = "localhost"
PORT = 8765

async def handler(websocket):
    """
    Simulates the ESP32.
    1. Accepts connection.
    2. Prints incoming commands.
    3. Sends dummy responses.
    """
    client_addr = websocket.remote_address
    print(f"\n[+] New connection from {client_addr}")

    try:
        # 1. Send Welcome Message (Your client expects this)
        await websocket.send("ESP32_MOCK_TERMINAL_READY")
        
        async for message in websocket:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] [RX] {message}")

            # 2. Simulate Responses
            response = None
            if "POSCURVANG" in message:
                response = "MOV_OK"
            elif "ANIMSTOP" in message:
                response = "STOP_OK"
            
            # Echo back to keep connection alive
            if response:
                await websocket.send(response)

    except websockets.exceptions.ConnectionClosedError:
        print(f"[-] Connection closed abruptly by {client_addr}")
    except websockets.exceptions.ConnectionClosedOK:
        print(f"[-] Connection closed normally by {client_addr}")
    finally:
        print(f"[-] Client disconnected")

async def main():
    print(f"[*] Mock ESP32 Server listening on ws://{HOST}:{PORT}/ws")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")