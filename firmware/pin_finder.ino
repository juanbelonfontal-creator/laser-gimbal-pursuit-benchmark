/*
 * pin_finder.ino  --  Arduino Uno
 *
 * Diagnostic ONLY. Upload this, open Serial Monitor at 9600 baud, and
 * watch which physical servo moves as each pin is announced.
 *
 * It attaches ONE pin at a time and wiggles it between 70 and 110 degrees,
 * so there is no ambiguity about which servo is responding. Write down the
 * pin number for the hunter pan servo and the hunter tilt servo, then put
 * those numbers into gimbal_receiver.ino.
 *
 * The 70-110 range is deliberately narrow and centred, so a servo cannot
 * slam into a mechanical stop while you are watching.
 */

#include <Servo.h>

// Candidate pins to test. Add or remove as needed.
const byte TEST_PINS[] = {9, 10, 5, 6, 3, 11};
const byte N_PINS = sizeof(TEST_PINS) / sizeof(TEST_PINS[0]);

Servo s;

void setup() {
  Serial.begin(9600);
  delay(1000);
  Serial.println();
  Serial.println("=== SERVO PIN FINDER ===");
  Serial.println("Watch which servo moves as each pin is called out.");
  Serial.println();
}

void loop() {
  for (byte i = 0; i < N_PINS; i++) {
    byte pin = TEST_PINS[i];

    Serial.print(">>> NOW TESTING PIN D");
    Serial.println(pin);

    s.attach(pin);

    // Three slow wiggles so it is easy to spot.
    for (byte rep = 0; rep < 3; rep++) {
      s.write(70);
      delay(600);
      s.write(110);
      delay(600);
    }
    s.write(90);
    delay(400);

    s.detach();          // release before moving to the next pin

    Serial.print("    (done with D");
    Serial.print(pin);
    Serial.println(")");
    Serial.println();
    delay(1500);         // pause so you can note it down
  }

  Serial.println("--- full pass complete, looping again ---");
  Serial.println();
  delay(3000);
}
