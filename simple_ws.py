#!/usr/bin/env python3
"""ESP32 WebSocket terminal with paced MOV POSCURVANG helper."""
import asyncio
import math
import websockets
from main import video_capture_coordinates


# ============================================================================
# CONFIGURATION
# ============================================================================
ESP32_IP = "192.168.0.100"
WS_URL = f"ws://{ESP32_IP}:80/ws"
RECONNECT_DELAY = 1
RECONNECT_TIMEOUT = 2
VERBOSE_MODE = False

BUSY_RETRY_DELAY = 0.5
BUSY_MAX_RETRIES = 3
SEND_MIN_INTERVAL = 0.25
IDLE_STOP_TIMEOUT = 10.0


# ============================================================================
# GLOBAL STATE
# ============================================================================
_last_send_ts = 0.0
_last_move_ts = 0.0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def clamp(value, min_v, max_v):
    """Clamp value between min and max."""
    return max(min_v, min(max_v, value))


def is_connected(ws):
    """Check if WebSocket is connected."""
    return ws is not None and not ws.closed


# ============================================================================
# COMMAND SENDING
# ============================================================================
async def send_command(ws, cmd):
    """Send command with pacing and retry for busy/backpressure handling."""
    global _last_send_ts
    loop = asyncio.get_event_loop()

    now = loop.time()
    delta = now - _last_send_ts
    if delta < SEND_MIN_INTERVAL:
        await asyncio.sleep(SEND_MIN_INTERVAL - delta)

    for retry in range(BUSY_MAX_RETRIES):
        try:
            if VERBOSE_MODE:
                print(f"[TX] {cmd}")
            await ws.send(cmd + "\n")
            _last_send_ts = loop.time()
            return True
        except Exception as exc:
            if retry >= BUSY_MAX_RETRIES - 1:
                print(f"[!] Send failed after {BUSY_MAX_RETRIES} retries: {exc}")
                return False
            await asyncio.sleep(BUSY_RETRY_DELAY)
    return False


async def send_commad(ws, mean_height, radious, x_angle, y_angle):
    """
    Send MOV POSCURVANG command with clamping, idle-stop, and 500ms pacing.
    
    Enforces minimum 500ms between calls (max 2 commands/second).
    """
    global _last_move_ts
    loop = asyncio.get_event_loop()

    # Clamp to safe ranges
    mh = clamp(mean_height, 100, 2100)
    radius = clamp(radious, -10000, 10000)
    xa = clamp(x_angle, -60, 60)
    ya = clamp(y_angle, -60, 60)

    now = loop.time()

    # Enforce 500ms minimum gap
    if _last_move_ts > 0:
        elapsed = now - _last_move_ts
        if elapsed < 1.0:
            await asyncio.sleep(1.0 - elapsed)
        if elapsed >= IDLE_STOP_TIMEOUT:
            await send_command(ws, "ANIMSTOP")

    cmd = f"MOV POSCURVANG {int(mh)} {int(radius)} {int(xa)} {int(ya)}"
    ok = await send_command(ws, cmd)
    if ok:
        _last_move_ts = loop.time()
    return ok


# ============================================================================
# DEMO SEQUENCE
# ============================================================================
async def demo_sequence(ws, height, curv_radius, roll, pitch):
    """Execute choreographed demo using send_commad."""
    print("\n" + "=" * 70)
    print("DEMO SEQUENCE: MOV POSCURVANG")
    print("=" * 70 + "\n")

    await send_command(ws, "ANIMSTOP")
    await asyncio.sleep(1.0)

    try:
        # 1. Height sweep (flat plane)
        print("[DEMO] Height sweep")

        print(height)
        print(roll)
        # ws, height, radius, roll, pitch
        await send_commad(ws, height, curv_radius, roll, pitch)
        # await send_commad(ws, 1000, 10000, 0, 0)
        await asyncio.sleep(6)

        # # 2. Circular tilt pattern
        # print("[DEMO] Circular tilt sweep")
        # for angle_deg in range(0, 361, 30):
        #     angle_rad = math.radians(angle_deg)
        #     x_tilt = round(40 * math.cos(angle_rad))
        #     y_tilt = round(40 * math.sin(angle_rad))
        #     await send_commad(ws, 1000, 10000, x_tilt, y_tilt)
        #     await asyncio.sleep(1)

        # # 3. Radius morph (flat to curved)
        # print("[DEMO] Radius morph 5000 -> 700")
        # for radius in range(5000, 699, -400):
        #     await send_commad(ws, 900, radius, 0, 0)
        #     await asyncio.sleep(1)

        # # 4. Curved surface tilt
        # print("[DEMO] Tilting curved surface (radius=1500)")
        # tilt_angles = [0, 20, 40, 20, 0, -20, -40, -20, 0]
        # for x_angle in tilt_angles:
        #     await send_commad(ws, 1200, 1500, x_angle, 0)
        #     await asyncio.sleep(1)

        # # 5. Sphere up/down sweep
        # print("[DEMO] Sphere up and down sweep")
        # await send_commad(ws, 600, 0, 0, 0)
        # await asyncio.sleep(6)
        # for z in range(600, 1400, 200):
        #     await send_commad(ws, z, 0, 0, 0)
        # for z in range(1400, 600, -200):
        #     await send_commad(ws, z, 0, 0, 0)

        print("\n[+] Demo sequence complete!\n")
    except Exception as e:
        print(f"[!] Demo error: {e}")


# ============================================================================
# ASYNC IO LOOPS
# ============================================================================
async def recv_loop(ws):
    """Receive and display ESP32 messages."""
    try:
        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=10)
                msg_str = message.strip()
                if msg_str:
                    print(f"\n[RX] {msg_str}")
                    print("ESP32> ", end='', flush=True)
            except asyncio.TimeoutError:
                continue
    except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed,
            ConnectionResetError, Exception):
        pass


async def send_loop(ws):
    """Read user input and send commands to ESP32."""
    loop = asyncio.get_event_loop()
    try:
        while True:
            height, curv_radius, roll, pitch = video_capture_coordinates()
            try:
                cmd = await loop.run_in_executor(None, input, "ESP32> ")
                cmd = cmd.strip()

                if not cmd:
                    continue

                if cmd.lower() == "demo":

                    await demo_sequence(ws, height, curv_radius, roll, pitch)
                    continue

                if cmd.lower() in ['exit', 'quit']:
                    print("\n[*] Disconnecting...")
                    return False

                if cmd.lower().startswith("poscurv"):
                    parts = cmd.split()
                    if len(parts) == 5:
                        try:
                            mh, rad, xa, ya = map(float, parts[1:5])
                            await send_commad(ws, mh, rad, xa, ya)
                            continue
                        except ValueError:
                            pass
                    print("[!] Usage: poscurv <mean_height> <radius> <x_angle> <y_angle>")
                    continue

                await send_command(ws, cmd)

            except EOFError:
                return False
            except (websockets.exceptions.ConnectionClosed, ConnectionResetError):
                raise
            except Exception as e:
                print(f"\n[!] Send error: {e}")
                return False
    except asyncio.CancelledError:
        pass

    return True


# ============================================================================
# CONNECTION MANAGEMENT
# ============================================================================
async def maintain_connection():
    """Main loop with auto-reconnect."""
    # while True:
    #     video_capture_coordinates()
    ws = None
    
    while True:
        try:
            if ws is None or not is_connected(ws):
                print(f"[*] Connecting to {WS_URL}...")
                try:
                    ws = await asyncio.wait_for(
                        websockets.connect(WS_URL),
                        timeout=RECONNECT_TIMEOUT
                    )
                    print("[+] Connected!\n")

                    try:
                        welcome = await asyncio.wait_for(ws.recv(), timeout=3)
                        print(f"[WELCOME] {welcome}\n")
                    except Exception:
                        pass

                except asyncio.TimeoutError:
                    if VERBOSE_MODE:
                        print(f"[-] Timeout, retrying in {RECONNECT_DELAY}s...")
                    else:
                        print(".", end='', flush=True)
                    await asyncio.sleep(RECONNECT_DELAY)
                    continue
                except Exception as e:
                    if VERBOSE_MODE:
                        print(f"[-] Failed: {e}")
                    print(f"[*] Retrying in {RECONNECT_DELAY}s...")
                    await asyncio.sleep(RECONNECT_DELAY)
                    continue

            recv_task = asyncio.create_task(recv_loop(ws))
            send_task = asyncio.create_task(send_loop(ws))

            done, pending = await asyncio.wait(
                [recv_task, send_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            if send_task in done and send_task.result() is False:
                print("[+] Disconnected")
                return

            print(f"\n[!] Connection lost")
            print(f"[*] Reconnecting in {RECONNECT_DELAY}s...\n")

            try:
                if ws:
                    await ws.close()
            except Exception:
                pass

            ws = None
            await asyncio.sleep(RECONNECT_DELAY)

        except KeyboardInterrupt:
            print("\n[*] Interrupted")
            try:
                if ws:
                    await ws.close()
            except Exception:
                pass
            print("[+] Disconnected")
            return
        except Exception as e:
            print(f"[-] Error: {e}")
            await asyncio.sleep(RECONNECT_DELAY)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":

    print(f"\nESP32 WebSocket Terminal - {ESP32_IP}")
    print("Commands: Type any ESP32 command, 'exit' to quit\n")
    print("=" * 70)
    print("Shortcuts:")
    print("  demo             - Run choreographed movement sequence")
    print("  poscurv z r x y  - Send MOV POSCURVANG (with clamping & pacing)")
    print("  H                - ESP32 help")
    print("=" * 70 + "\n")

    try:
        asyncio.run(maintain_connection())
    except KeyboardInterrupt:
        print("\n[+] Exiting")

