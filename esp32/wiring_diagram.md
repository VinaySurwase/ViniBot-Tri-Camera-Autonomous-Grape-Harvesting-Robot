# ViniBot ESP32 Wiring Diagram
# ==============================
# ESP32 → BTS7960B × 2 + HC-SR04 × 3 + RPi5

=================================================================
  UART (RPi5 ↔ ESP32)
=================================================================

  RPi5 Pin 8  (GPIO 14 / TX)  ───────────────► ESP32 RX2 (GPIO 16)
  RPi5 Pin 10 (GPIO 15 / RX)  ◄─────────────── ESP32 TX2 (GPIO 17)
  RPi5 Pin 6  (GND)           ───────────────── ESP32 GND

  NOTE: Both are 3.3V logic. NO level shifter needed for UART.
  Baud: 115200, 8N1

=================================================================
  LEFT BTS7960B BOARD  (Controls: Front-Left + Rear-Left motors)
=================================================================

  ESP32 GPIO 25  ──────────────────────────► LEFT BTS7960B  RPWM
  ESP32 GPIO 26  ──────────────────────────► LEFT BTS7960B  LPWM
  ESP32 GPIO 32  ──────────────────────────► LEFT BTS7960B  R_EN
  ESP32 GPIO 32  ──────────────────────────► LEFT BTS7960B  L_EN  (same pin, tie EN lines)
  ESP32 5V / Vin ──────────────────────────► LEFT BTS7960B  VCC   (logic supply)
  ESP32 GND      ──────────────────────────► LEFT BTS7960B  GND

  Battery (+)    ──────────────────────────► LEFT BTS7960B  B+
  Battery (-)    ──────────────────────────► LEFT BTS7960B  B-

  LEFT BTS7960B  M+  ──────────────────────── Front-Left Motor (+)
  LEFT BTS7960B  M-  ──────────────────────── Front-Left Motor (-)
                     ─── (parallel) ────────── Rear-Left Motor (+)
                                  ─────────── Rear-Left Motor (-)

  ┌─────────────────────────────────────────────────────────────┐
  │ Motor direction logic (BTS7960B):                           │
  │   RPWM HIGH, LPWM=0  → Motor spins FORWARD                 │
  │   LPWM HIGH, RPWM=0  → Motor spins BACKWARD                │
  │   Both LOW            → Motor COASTS (off)                  │
  │   Both HIGH           → NEVER DO THIS (shoot-through!)      │
  └─────────────────────────────────────────────────────────────┘

=================================================================
  RIGHT BTS7960B BOARD  (Controls: Front-Right + Rear-Right motors)
=================================================================

  ESP32 GPIO 27  ──────────────────────────► RIGHT BTS7960B  RPWM
  ESP32 GPIO 14  ──────────────────────────► RIGHT BTS7960B  LPWM
  ESP32 GPIO 33  ──────────────────────────► RIGHT BTS7960B  R_EN
  ESP32 GPIO 33  ──────────────────────────► RIGHT BTS7960B  L_EN  (same pin, tie EN lines)
  ESP32 5V / Vin ──────────────────────────► RIGHT BTS7960B  VCC   (logic supply)
  ESP32 GND      ──────────────────────────► RIGHT BTS7960B  GND

  Battery (+)    ──────────────────────────► RIGHT BTS7960B  B+
  Battery (-)    ──────────────────────────► RIGHT BTS7960B  B-

  RIGHT BTS7960B M+  ─────────────────────── Front-Right Motor (+)
  RIGHT BTS7960B M-  ─────────────────────── Front-Right Motor (-)
                     ─── (parallel) ────────── Rear-Right Motor (+)
                                   ─────────── Rear-Right Motor (-)

=================================================================
  HC-SR04 SENSORS  (via 5V→3.3V Level Shifter on ECHO pin)
=================================================================

  SENSOR         | HC-SR04 Pin | Level Shifter | ESP32 GPIO
  ─────────────────────────────────────────────────────────
  FRONT          | VCC         | ─────────────── 5V
  FRONT          | GND         | ─────────────── GND
  FRONT          | TRIG        | ─────────────── GPIO 4  (3.3V output, direct)
  FRONT          | ECHO        | HV side ──────► GPIO 5  (LV side → ESP32)
  ─────────────────────────────────────────────────────────
  LEFT           | VCC         | ─────────────── 5V
  LEFT           | GND         | ─────────────── GND
  LEFT           | TRIG        | ─────────────── GPIO 18 (3.3V output, direct)
  LEFT           | ECHO        | HV side ──────► GPIO 19 (LV side → ESP32)
  ─────────────────────────────────────────────────────────
  RIGHT          | VCC         | ─────────────── 5V
  RIGHT          | GND         | ─────────────── GND
  RIGHT          | TRIG        | ─────────────── GPIO 21 (3.3V output, direct)
  RIGHT          | ECHO        | HV side ──────► GPIO 22 (LV side → ESP32)

  ┌──────────────────────────────────────────────────────────────┐
  │ IMPORTANT: HC-SR04 ECHO outputs 5V and WILL damage ESP32!   │
  │ Connect ECHO only via the 5V→3.3V side of the level shifter.│
  │ TRIG accepts 3.3V input directly — no level shifter needed. │
  └──────────────────────────────────────────────────────────────┘

  Alternative (no level shifter board): voltage divider on each ECHO:
    ECHO ──── 1kΩ ──── GPIO ──── 2kΩ ──── GND
    (Output is 5V × 2k/(1k+2k) = 3.33V ✓)

=================================================================
  POWER SUPPLY SUMMARY
=================================================================

  Device           | Supply     | Source
  ──────────────────────────────────────────────────────
  ESP32 Dev Board  | 5V / USB   | DC-DC buck converter or USB
  BTS7960B VCC     | 5V         | ESP32 5V / Vin pin
  HC-SR04 VCC      | 5V         | ESP32 5V / Vin pin
  Left Motors ×2   | 12V / 24V  | Main robot battery
  Right Motors ×2  | 12V / 24V  | Main robot battery
  RPi 5            | 5V 5A      | Separate power supply

  WARNING: Never power ESP32 from RPi 5V GPIO pin!
           Use a dedicated power source for ESP32.

=================================================================
  COMPLETE ESP32 PIN ASSIGNMENT SUMMARY
=================================================================

  GPIO  | Function          | Connected to
  ──────────────────────────────────────────────────────────────
  GPIO 4  | FRONT TRIG      | HC-SR04 FRONT TRIG
  GPIO 5  | FRONT ECHO      | Level shifter LV → HC-SR04 FRONT ECHO
  GPIO 14 | RIGHT_BWD_PWM   | RIGHT BTS7960B LPWM
  GPIO 16 | UART2 RX        | RPi5 TX (GPIO 14, Pin 8)
  GPIO 17 | UART2 TX        | RPi5 RX (GPIO 15, Pin 10)
  GPIO 18 | LEFT TRIG       | HC-SR04 LEFT TRIG
  GPIO 19 | LEFT ECHO       | Level shifter LV → HC-SR04 LEFT ECHO
  GPIO 21 | RIGHT TRIG      | HC-SR04 RIGHT TRIG
  GPIO 22 | RIGHT ECHO      | Level shifter LV → HC-SR04 RIGHT ECHO
  GPIO 25 | LEFT_FWD_PWM    | LEFT BTS7960B RPWM
  GPIO 26 | LEFT_BWD_PWM    | LEFT BTS7960B LPWM
  GPIO 27 | RIGHT_FWD_PWM   | RIGHT BTS7960B RPWM
  GPIO 32 | LEFT_EN         | LEFT BTS7960B R_EN + L_EN
  GPIO 33 | RIGHT_EN        | RIGHT BTS7960B R_EN + L_EN
