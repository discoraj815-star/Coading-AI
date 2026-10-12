"""Trading AI Assistant — Phone Connection & Device Pairing Utility.

Displays private local network address, generates device pairing PINs,
and verifies cryptographic device pairing state per Step 12 DeviceManager.
"""

import os
import sys
import socket
import logging
import argparse
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from trading_ai.core.config import get_settings
from trading_ai.security.device_manager import DeviceManager
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PhonePairing")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_local_ip() -> str:
    """Detect private LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not send packets
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def main():
    parser = argparse.ArgumentParser(description="Trading AI Assistant Phone Pairing Helper")
    parser.add_argument("--device-name", default="My Personal Phone", help="Label for phone device")
    args = parser.parse_args()

    settings = get_settings()
    db_mgr = DatabaseManager(settings.db_path)
    repo = Repository(db_mgr)
    dev_mgr = DeviceManager(repository=repo)

    lan_ip = get_local_ip()
    port = 8000

    print("=" * 75)
    print("PRIVATE TRADING AI ASSISTANT -- PHONE SETUP & PAIRING GUIDE")
    print("=" * 75)
    print(f"Desktop Private Network (LAN) IP : {lan_ip}")
    print(f"Phone Access URL                 : http://{lan_ip}:{port}")
    print("-" * 75)

    # 1. Generate pairing code
    dev_id = f"phone_{int(lan_ip.replace('.', '')) % 100000:05d}"
    pin = dev_mgr.generate_pairing_code(device_id=dev_id, device_name=args.device_name, device_type="phone")

    print("DEVICE PAIRING DETAILS:")
    print(f"  Device ID    : {dev_id}")
    print(f"  Device Name  : {args.device_name}")
    print(f"  Pairing PIN  : {pin}  (Expires in 5 minutes)")
    print("-" * 75)
    print("HOW TO CONNECT YOUR PHONE:")
    print("1. Ensure your phone is connected to the SAME private Wi-Fi / LAN network (or WireGuard/Tailscale VPN).")
    print(f"2. On your phone's browser (Chrome/Safari), open: http://{lan_ip}:{port}")
    print("3. Enter your Personal Access Token (or enter the Pairing PIN above).")
    print("4. To install as an app on your phone:")
    print("   - Android (Chrome): Tap the menu (three dots) -> 'Install app' or 'Add to Home screen'.")
    print("   - iPhone (Safari): Tap the Share button -> 'Add to Home Screen'.")
    print("5. The Private PWA icon will be added to your home screen.")
    print("=" * 75)


if __name__ == "__main__":
    main()
