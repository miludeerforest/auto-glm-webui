"""
Open-AutoGLM Multi-Device Web Launcher

Supports controlling multiple Android devices simultaneously over LAN.
"""

import json
import os
import subprocess
import threading
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Open-AutoGLM Multi-Device Launcher")

# Configuration paths
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "launcher" / "config.json"
DEVICES_PATH = BASE_DIR / "launcher" / "devices.json"  # Remembered remote devices
STATIC_DIR = BASE_DIR / "launcher" / "static"

# ============ Data Models ============

@dataclass
class DeviceInfo:
    device_id: str
    status: str  # "online", "offline", "running"
    model: Optional[str] = None
    current_task: Optional[str] = None

class Settings(BaseModel):
    base_url: str
    api_key: str
    model: str
    max_steps: int = 100

class RunRequest(BaseModel):
    task: str

class ConnectRequest(BaseModel):
    address: str  # IP:port format

# ============ Global State ============

# Device management
devices: dict[str, DeviceInfo] = {}

# Process per device
agent_processes: dict[str, subprocess.Popen] = {}

# Log queue per device
log_queues: dict[str, asyncio.Queue] = {}

# WebSocket clients per device
log_clients: dict[str, list[WebSocket]] = {}

# ============ Settings Management ============

def load_settings() -> Settings:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return Settings(**data)
        except Exception:
            pass
    return Settings(
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key="",
        model="autoglm-phone",
        max_steps=100
    )

def save_settings(settings: Settings):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(settings.model_dump(), f, indent=2)

# ============ Remembered Devices ============

def load_remembered_devices() -> list[str]:
    """Load list of remembered remote device addresses."""
    if DEVICES_PATH.exists():
        try:
            with open(DEVICES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("remote_devices", [])
        except Exception:
            pass
    return []

def save_remembered_devices(addresses: list[str]):
    """Save list of remembered remote device addresses."""
    DEVICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEVICES_PATH, "w", encoding="utf-8") as f:
        json.dump({"remote_devices": addresses}, f, indent=2)

def add_remembered_device(address: str):
    """Add a device to remembered list."""
    devices_list = load_remembered_devices()
    if address not in devices_list:
        devices_list.append(address)
        save_remembered_devices(devices_list)

def remove_remembered_device(address: str):
    """Remove a device from remembered list."""
    devices_list = load_remembered_devices()
    if address in devices_list:
        devices_list.remove(address)
        save_remembered_devices(devices_list)

# ============ Task Templates ============

TEMPLATES_PATH = BASE_DIR / "launcher" / "templates.json"

@dataclass
class TaskTemplate:
    id: str
    name: str
    task: str

def load_templates() -> list[dict]:
    """Load saved task templates."""
    if TEMPLATES_PATH.exists():
        try:
            with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Default templates
    return [
        {
            "id": "1",
            "name": "抖音刷视频点赞",
            "task": "打开抖音，自动浏览视频并随机在5秒内点击喜爱，然后上滑浏览下一个视频并随机5秒内点击喜爱，循环20次"
        },
        {
            "id": "2", 
            "name": "微信发消息",
            "task": "打开微信，给文件传输助手发送一条消息：测试消息"
        }
    ]

def save_templates(templates: list[dict]):
    """Save task templates."""
    TEMPLATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEMPLATES_PATH, "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)

# ============ Device Discovery ============

def discover_devices() -> list[DeviceInfo]:
    """Discover all connected ADB devices."""
    global devices
    
    try:
        result = subprocess.run(
            ["adb", "devices", "-l"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace"
        )
        
        new_devices = {}
        for line in result.stdout.strip().split("\n")[1:]:
            if not line.strip():
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                device_id = parts[0]
                status = parts[1]
                
                # Parse model info if available
                model = None
                for part in parts[2:]:
                    if part.startswith("model:"):
                        model = part.split(":", 1)[1]
                        break
                
                # Preserve running status if device was already known
                if device_id in devices and devices[device_id].status == "running":
                    new_devices[device_id] = devices[device_id]
                else:
                    new_devices[device_id] = DeviceInfo(
                        device_id=device_id,
                        status="online" if status == "device" else "offline",
                        model=model
                    )
        
        devices = new_devices
        return list(devices.values())
    
    except Exception as e:
        print(f"Error discovering devices: {e}")
        return list(devices.values())

def pair_device(address: str, pairing_code: str) -> tuple[bool, str]:
    """
    Pair with Android 11+ device using wireless debugging pairing code.
    
    Args:
        address: IP:port shown in "Use pairing code" dialog (e.g., 192.168.1.100:37123)
        pairing_code: 6-digit pairing code
    
    Returns:
        (success, message)
    """
    try:
        # Use adb pair command
        result = subprocess.run(
            ["adb", "pair", address, pairing_code],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace"
        )
        
        output = result.stdout + result.stderr
        
        if "Successfully paired" in output or "successfully" in output.lower():
            # After pairing, we need to connect to the main wireless debugging port
            # The pairing port is different from the connection port
            # Let user know they now need to connect using the main port
            return True, f"配对成功！请在手机的'IP地址和端口'处查看连接地址，然后在上方输入框中输入该地址进行连接。"
        elif "already paired" in output.lower():
            return True, "设备已经配对过了。请使用上方输入框输入'IP地址和端口'进行连接。"
        else:
            return False, f"配对失败: {output.strip()}"
    
    except subprocess.TimeoutExpired:
        return False, "配对超时，请检查配对码是否正确"
    except Exception as e:
        return False, str(e)

def connect_remote_device(address: str, remember: bool = True) -> tuple[bool, str]:
    """Connect to a remote device via TCP/IP."""
    if ":" not in address:
        address = f"{address}:5555"
    
    try:
        result = subprocess.run(
            ["adb", "connect", address],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace"
        )
        
        output = result.stdout + result.stderr
        if "connected" in output.lower():
            if remember:
                add_remembered_device(address)  # Remember this device
            discover_devices()  # Refresh device list
            return True, f"Connected to {address}"
        else:
            return False, output.strip()
    
    except Exception as e:
        return False, str(e)

def disconnect_device(device_id: str) -> tuple[bool, str]:
    """Disconnect from a remote device."""
    try:
        result = subprocess.run(
            ["adb", "disconnect", device_id],
            capture_output=True,
            text=True,
            timeout=5,
            encoding="utf-8",
            errors="replace"
        )
        
        discover_devices()  # Refresh device list
        return True, "Disconnected"
    
    except Exception as e:
        return False, str(e)

def enable_wifi_debug(device_id: str, port: int = 5555) -> tuple[bool, str, str | None]:
    """
    Enable WiFi debugging on a USB-connected device.
    
    Returns:
        (success, message, wifi_address or None)
    """
    import re
    import time
    
    try:
        # Step 1: Enable TCP/IP mode
        result = subprocess.run(
            ["adb", "-s", device_id, "tcpip", str(port)],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace"
        )
        
        if result.returncode != 0:
            return False, f"Failed to enable TCP/IP: {result.stderr}", None
        
        # Wait for device to restart ADB in TCP/IP mode
        time.sleep(2)
        
        # Step 2: Get device WiFi IP address (prioritize wlan0)
        ip_address = None
        
        # Method 1: Try wlan0 directly first (most reliable for WiFi)
        result = subprocess.run(
            ["adb", "-s", device_id, "shell", "ip", "addr", "show", "wlan0"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace"
        )
        
        match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', result.stdout)
        if match:
            ip_address = match.group(1)
        
        # Method 2: Try dumpsys wifi if wlan0 failed
        if not ip_address:
            result = subprocess.run(
                ["adb", "-s", device_id, "shell", "dumpsys", "wifi", "|", "grep", "ip_address"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
                errors="replace",
                shell=True
            )
            match = re.search(r'ip_address=(\d+\.\d+\.\d+\.\d+)', result.stdout)
            if match:
                ip_address = match.group(1)
        
        # Method 3: Try ip route but filter for wlan0 specifically
        if not ip_address:
            result = subprocess.run(
                ["adb", "-s", device_id, "shell", "ip", "route"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
                errors="replace"
            )
            
            for line in result.stdout.split("\n"):
                # Only use lines with wlan0 interface
                if "wlan0" in line and "src" in line:
                    parts = line.split("src")
                    if len(parts) > 1:
                        candidate = parts[1].strip().split()[0]
                        # Filter out virtual/internal addresses
                        if not candidate.startswith(("172.", "10.0.", "127.")):
                            ip_address = candidate
                            break
        
        # Validate the IP address
        if ip_address:
            # Skip common virtual network ranges
            if ip_address.startswith(("172.17.", "172.18.", "172.19.", "172.20.", "10.0.2.")):
                return True, f"TCP/IP enabled. Detected IP {ip_address} appears to be a virtual network. Please check your WiFi connection and try connecting manually.", None
        
        if not ip_address:
            return True, "TCP/IP enabled but could not get WiFi IP. Please ensure device is connected to WiFi and try connecting manually with the device's IP address.", None
        
        wifi_address = f"{ip_address}:{port}"
        
        # Step 3: Auto-connect to the WiFi address
        success, msg = connect_remote_device(wifi_address, remember=True)
        
        if success:
            return True, f"WiFi debugging enabled! Connected to {wifi_address}", wifi_address
        else:
            return True, f"TCP/IP enabled. Address: {wifi_address}. Auto-connect failed: {msg}", wifi_address
    
    except Exception as e:
        return False, str(e), None

# ============ Log Broadcasting ============

async def log_broadcaster(device_id: str):
    """Broadcast logs for a specific device."""
    if device_id not in log_queues:
        log_queues[device_id] = asyncio.Queue()
    
    queue = log_queues[device_id]
    
    while True:
        try:
            line = await asyncio.wait_for(queue.get(), timeout=0.5)
            if line is None:
                # Process ended signal
                print(f"Log broadcaster for {device_id}: received end signal")
                break
            
            # Broadcast to all clients for this device
            if device_id in log_clients:
                to_remove = []
                for client in log_clients[device_id]:
                    try:
                        await client.send_text(line)
                    except Exception:
                        to_remove.append(client)
                
                for c in to_remove:
                    if c in log_clients[device_id]:
                        log_clients[device_id].remove(c)
        except asyncio.TimeoutError:
            # Just continue waiting, don't check process status here
            # The output_reader thread will send None when process ends
            continue
        except Exception as e:
            print(f"Log broadcaster error for {device_id}: {e}")
            break
    
    # Update device status when broadcaster ends
    if device_id in devices:
        devices[device_id].status = "online"
        devices[device_id].current_task = None

def output_reader(pipe, device_id: str, loop):
    """Thread target to read output and push to device-specific queue."""
    try:
        for line in iter(pipe.readline, ""):
            if line:
                if device_id not in log_queues:
                    log_queues[device_id] = asyncio.Queue()
                asyncio.run_coroutine_threadsafe(log_queues[device_id].put(line), loop)
            else:
                break
    except Exception as e:
        print(f"Output reader error for {device_id}: {e}")
    finally:
        # Signal end of logs
        if device_id in log_queues:
            asyncio.run_coroutine_threadsafe(log_queues[device_id].put(None), loop)

# ============ API Endpoints ============

@app.on_event("startup")
async def startup_event():
    """Auto-discover devices and reconnect to remembered remote devices."""
    print("🔍 Discovering devices...")
    discover_devices()
    
    # Auto-reconnect to remembered devices
    remembered = load_remembered_devices()
    if remembered:
        print(f"📱 Reconnecting to {len(remembered)} remembered device(s)...")
        for address in remembered:
            success, msg = connect_remote_device(address, remember=False)
            if success:
                print(f"  ✅ {address}")
            else:
                print(f"  ❌ {address}: {msg}")
    
    print(f"📱 Found {len(devices)} device(s)")

@app.get("/api/settings")
def get_settings():
    return load_settings()

@app.post("/api/settings")
def update_settings(settings: Settings):
    save_settings(settings)
    return {"status": "ok"}

# ============ Task Templates API ============

@app.get("/api/templates")
def get_templates():
    """Get all task templates."""
    return {"templates": load_templates()}

class TemplateRequest(BaseModel):
    name: str
    task: str

@app.post("/api/templates")
def add_template(req: TemplateRequest):
    """Add a new task template."""
    templates = load_templates()
    new_id = str(max([int(t.get("id", 0)) for t in templates] + [0]) + 1)
    templates.append({"id": new_id, "name": req.name, "task": req.task})
    save_templates(templates)
    return {"status": "ok", "id": new_id}

@app.delete("/api/templates/{template_id}")
def delete_template(template_id: str):
    """Delete a task template."""
    templates = load_templates()
    templates = [t for t in templates if t.get("id") != template_id]
    save_templates(templates)
    return {"status": "ok"}


@app.get("/api/devices")
def list_devices():
    """List all connected devices."""
    discover_devices()
    return {"devices": [asdict(d) for d in devices.values()]}

@app.post("/api/devices/refresh")
def refresh_devices():
    """Force refresh device list."""
    discover_devices()
    return {"devices": [asdict(d) for d in devices.values()]}

@app.post("/api/devices/connect")
def connect_device(req: ConnectRequest):
    """Connect to a remote device."""
    success, message = connect_remote_device(req.address)
    return {"success": success, "message": message}

class PairRequest(BaseModel):
    address: str
    code: str

@app.post("/api/devices/pair")
def pair(req: PairRequest):
    """Pair with Android 11+ device using wireless debugging."""
    success, message = pair_device(req.address, req.code)
    return {"success": success, "message": message}

@app.post("/api/devices/{device_id}/disconnect")
def disconnect(device_id: str):
    """Disconnect from a device."""
    success, message = disconnect_device(device_id)
    return {"success": success, "message": message}

@app.post("/api/devices/{device_id}/wifi-debug")
def enable_wifi(device_id: str):
    """Enable WiFi debugging on a USB-connected device."""
    success, message, address = enable_wifi_debug(device_id)
    return {"success": success, "message": message, "address": address}

@app.post("/api/run/{device_id}")
async def run_on_device(device_id: str, req: RunRequest):
    """Run task on a specific device."""
    global agent_processes, devices
    
    if device_id not in devices:
        return {"status": "error", "message": f"Device {device_id} not found"}
    
    if device_id in agent_processes and agent_processes[device_id].poll() is None:
        return {"status": "error", "message": f"Task already running on {device_id}"}
    
    settings = load_settings()
    
    # Construct command with device ID
    cmd = [
        "python", "main.py",
        "--base-url", settings.base_url,
        "--apikey", settings.api_key,
        "--model", settings.model,
        "--device", device_id,
        req.task
    ]
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            cwd=str(BASE_DIR),
            env=env
        )
        
        agent_processes[device_id] = process
        devices[device_id].status = "running"
        devices[device_id].current_task = req.task[:50] + "..." if len(req.task) > 50 else req.task
        
        # Initialize log queue
        log_queues[device_id] = asyncio.Queue()
        
        # Start log broadcaster
        asyncio.create_task(log_broadcaster(device_id))
        
        # Start output reader thread
        loop = asyncio.get_running_loop()
        t = threading.Thread(target=output_reader, args=(process.stdout, device_id, loop))
        t.daemon = True
        t.start()
        
        return {"status": "started", "device_id": device_id}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/run-all")
async def run_on_all_devices(req: RunRequest):
    """Run same task on all online devices."""
    results = []
    
    for device_id, device in devices.items():
        if device.status == "online":
            result = await run_on_device(device_id, req)
            results.append({"device_id": device_id, **result})
    
    return {"results": results}

@app.post("/api/stop/{device_id}")
def stop_on_device(device_id: str):
    """Stop task on a specific device."""
    if device_id in agent_processes and agent_processes[device_id].poll() is None:
        agent_processes[device_id].terminate()
        if device_id in devices:
            devices[device_id].status = "online"
            devices[device_id].current_task = None
        return {"status": "stopped", "device_id": device_id}
    return {"status": "not_running", "device_id": device_id}

@app.post("/api/stop-all")
def stop_all_devices():
    """Stop tasks on all devices."""
    results = []
    for device_id in list(agent_processes.keys()):
        result = stop_on_device(device_id)
        results.append(result)
    return {"results": results}

@app.websocket("/ws/logs/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    """WebSocket for device-specific logs."""
    await websocket.accept()
    
    if device_id not in log_clients:
        log_clients[device_id] = []
    log_clients[device_id].append(websocket)
    
    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        if device_id in log_clients and websocket in log_clients[device_id]:
            log_clients[device_id].remove(websocket)

# Serve static files
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("launcher_app:app", host="127.0.0.1", port=8000, reload=True)
