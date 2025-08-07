# main.py
"""
Main loop:
1. Capture image
2. OCR title
3. Determine zone
4. Generate and transmit G-code
"""

import os
from capture import capture_frame
from ocr import extract_title
from sorter import determine_zone
from gcode import generate_gcode
from serial_sender import send_gcode


# ----- CONFIGURE YOUR HARDWARE HERE -----
SERIAL_PORT = os.getenv("MPCNC_PORT", "/dev/ttyUSB0")
BAUDRATE     = int(os.getenv("MPCNC_BAUD", "115200"))

def process_one_card() -> None:
    frame = capture_frame(preview=False)
    title = extract_title(frame)
    print("OCR:", repr(title))

    zone_name, coord = determine_zone(title)
    if coord is None:
        print("!!! Unable to classify. Manual intervention required.")
        return

    print(f"Sorting to zone {zone_name} at {coord}")
    gcode = generate_gcode(coord)
    print("Generated G-code:\n", gcode)
    send_gcode(SERIAL_PORT, BAUDRATE, gcode)
    print("Done.\n")


if __name__ == "__main__":
    try:
        while True:
            input("Insert card and press <Enter> (Ctrl-C to quit)… ")
            process_one_card()
    except KeyboardInterrupt:
        print("\nExiting.")
