"""
servo_test.py -- isolate the hardware path from the vision pipeline.

No camera, no OpenCV, no laser detection. This only opens the serial port and
streams angle commands, exactly as the benchmark does. If the gimbal moves
here, the Arduino, wiring, power and pin map are all fine and any "gimbal
never moved" problem is upstream in the vision/state-machine code. If it does
NOT move here, the fault is in the hardware path and the benchmark cannot
possibly work.

Run:  python servo_test.py
"""

import serial
import time

SERIAL_PORT = '/dev/cu.usbmodem14101'   # must match the benchmark script
BAUD_RATE = 115200

# Same protocol as the benchmark: hunterPan,hunterTilt,targetPan,targetTilt\n
# The last two are parsed and discarded by the firmware (virtual target).
TARGET_PAN, TARGET_TILT = 90, 125


def send(ser, pan, tilt):
    pan = max(0, min(180, int(pan)))
    tilt = max(0, min(180, int(tilt)))
    ser.write(f"{pan},{tilt},{TARGET_PAN},{TARGET_TILT}\n".encode('utf-8'))
    return pan, tilt


def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"FAIL: could not open {SERIAL_PORT}: {e}")
        print("Check the cable, and that the Arduino IDE's Serial Monitor is CLOSED")
        print("(it holds the port exclusively and the script will be locked out).")
        return

    print(f"Opened {SERIAL_PORT}. Waiting 2s for the Arduino to finish resetting...")
    time.sleep(2)

    # The firmware prints READY once on boot. Seeing it confirms two-way comms.
    time.sleep(0.3)
    if ser.in_waiting:
        print(f"Arduino says: {ser.read(ser.in_waiting).decode(errors='replace').strip()}")
    else:
        print("(no startup banner seen -- not fatal, the reset may have finished already)")

    print("\n--- TEST 1: TILT sweep (laser should move UP the board, then back) ---")
    print("    170 = ceiling park, lower numbers aim further down the board.")
    for tilt in list(range(170, 119, -5)) + list(range(120, 171, 5)):
        p, t = send(ser, 90, tilt)
        print(f"    sent pan={p} tilt={t}")
        time.sleep(0.15)

    time.sleep(1.0)

    print("\n--- TEST 2: PAN sweep (laser should move LEFT then RIGHT) ---")
    for pan in list(range(90, 61, -3)) + list(range(60, 121, 3)) + list(range(120, 89, -3)):
        p, t = send(ser, pan, 140)
        print(f"    sent pan={p} tilt={t}")
        time.sleep(0.15)

    print("\n--- Returning to the ceiling park position ---")
    send(ser, 90, 170)
    time.sleep(1.0)
    ser.close()

    print("""
RESULT GUIDE
  Both sweeps moved          -> hardware is fine; the problem is in the
                                vision/state machine (likely stuck in the
                                calibration state, never reaching homing).
  Only ONE sweep moved       -> one servo is unplugged, or the pin map is
                                wrong for that axis. Swap HUNTER_PAN_PIN and
                                HUNTER_TILT_PIN in gimbal_receiver.ino.
  Neither moved, port opened -> wrong pins entirely, no servo power, or the
                                sketch is not uploaded. Re-upload
                                gimbal_receiver.ino and check that pan is on
                                D3 and tilt on D5.
  Gimbal twitched then froze -> brown-out. Two servos off the Uno's 5V rail
                                can reset the board. Use a separate 5-6V
                                supply with a common ground.
""")


if __name__ == "__main__":
    main()
