/*
 * gimbal_receiver.ino  --  Arduino Uno
 *
 * Firmware for the Python laser-pursuit benchmark.
 *
 * HARDWARE
 *   Two servos only -- the hunter gimbal carrying the red laser.
 *     Hunter PAN  signal -> D3
 *     Hunter TILT signal -> D5
 *   The "target" is virtual: it is a green circle drawn by the Python
 *   script. Nothing physical corresponds to it.
 *
 * PROTOCOL (matches send_angles() in the Python script)
 *   Baud 115200, one ASCII line per command, terminated with '\n':
 *   (was 9600, which cost 14.6 ms per command -- 23% of a frame at 17 fps.
 *    115200 cuts that to 1.2 ms. The Python BAUD_RATE must match.)
 *
 *       hunterPan,hunterTilt,targetPan,targetTilt\n
 *       e.g.   90,170,90,125\n
 *
 *   The last two fields are parsed and then discarded -- they exist only
 *   so this firmware stays compatible with the Python script as written.
 *   Each value is an integer 0..180 and is clamped here as well, so a
 *   garbled line can never drive a servo past its limit. Lines that do not
 *   contain 4 fields are ignored and the servos hold position.
 *
 * STARTUP POSE
 *   Tilt parks at 170, which aims the laser at the ceiling, matching what
 *   the Python script assumes as its starting position (and what it sends
 *   on shutdown). During homing the script walks the tilt down until the
 *   camera picks the dot up inside the board.
 *
 * POWER
 *   Two small servos will usually run off the Uno's 5V pin, but if you see
 *   the board resetting mid-run -- the laser jumping back to the ceiling
 *   for no reason -- that is a brown-out. Move the servo V+ to a separate
 *   5-6V supply and tie its ground to the Uno's ground.
 */

#include <Servo.h>
#include <stdlib.h>

// ---------- pin map ----------
const byte HUNTER_PAN_PIN  = 3;
const byte HUNTER_TILT_PIN = 5;

// ---------- startup pose (matches the Python defaults) ----------
const int HOME_HUNTER_PAN  = 90;
const int HOME_HUNTER_TILT = 170;   // aims at the ceiling

// ---------- optional mechanical safety limits ----------
// Tighten these if your gimbal binds before it reaches 0 or 180.
const int MIN_ANGLE = 0;
const int MAX_ANGLE = 180;

Servo hunterPan;
Servo hunterTilt;

// ---------- serial line buffer ----------
const byte BUF_LEN = 32;
char  buf[BUF_LEN];
byte  bufIdx = 0;

int clampAngle(long v) {
  if (v < MIN_ANGLE) return MIN_ANGLE;
  if (v > MAX_ANGLE) return MAX_ANGLE;
  return (int)v;
}

void writeHunter(int pan, int tilt) {
  hunterPan.write(clampAngle(pan));
  hunterTilt.write(clampAngle(tilt));
}

void parseLine(char *line) {
  int vals[4];
  byte n = 0;

  char *tok = strtok(line, ",");
  while (tok != NULL && n < 4) {
    while (*tok == ' ') tok++;          // tolerate a stray leading space
    vals[n++] = clampAngle(atol(tok));
    tok = strtok(NULL, ",");
  }

  // Only act on a complete, well-formed 4-field command.
  // vals[2] and vals[3] are the virtual target -- deliberately unused.
  if (n == 4) {
    writeHunter(vals[0], vals[1]);
  }
}

void setup() {
  Serial.begin(115200);

  hunterPan.attach(HUNTER_PAN_PIN);
  hunterTilt.attach(HUNTER_TILT_PIN);

  writeHunter(HOME_HUNTER_PAN, HOME_HUNTER_TILT);

  delay(500);            // let the servos settle before the PC streams

  // Single startup banner. Deliberately NOT echoing every command: the
  // Python side never reads from the port, so a per-command reply would
  // slowly fill the host's input buffer over a long benchmark run.
  Serial.println("READY");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (bufIdx > 0) {
        buf[bufIdx] = '\0';
        parseLine(buf);
        bufIdx = 0;
      }
    } else if (bufIdx < (BUF_LEN - 1)) {
      buf[bufIdx++] = c;
    } else {
      bufIdx = 0;        // overlong garbage: drop it, resync on next newline
    }
  }
}
