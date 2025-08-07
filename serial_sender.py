# serial_sender.py
"""
Send G-code lines to a Marlin-style controller over USB serial.
Requires: python3-serial  (Debian package: python3-serial)
"""

import time
import serial


def send_gcode(
    port: str,
    baudrate: int,
    gcode_text: str,
    wait_for_ok: bool = True,
) -> None:
    with serial.Serial(port, baudrate, timeout=1) as ser:
        for line in gcode_text.splitlines():
            ser.write((line + "\n").encode())
            ser.flush()
            if wait_for_ok:
                # Read lines until we get an 'ok'
                while True:
                    resp = ser.readline().decode(errors="ignore").strip()
                    if resp == "ok" or resp == "":
                        break
            else:
                time.sleep(0.05)  # crude pacing
