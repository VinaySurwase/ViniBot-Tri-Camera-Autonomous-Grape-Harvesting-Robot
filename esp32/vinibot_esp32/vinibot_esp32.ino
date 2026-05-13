/*
 * vinibot_esp32.ino — ViniBot Chassis Motor Controller
 * =====================================================
 * Includes "Dual-Ear" JSON Bracket Catcher (Listens to both Pi and PC)
 */

#include <Arduino.h>
#include <ArduinoJson.h>

#define SERVO_PIN 13
#define SCISSOR_PIN 23   // GPIO12 is a strapping pin — use GPIO23 instead

// ── Native Servo Control ─────────────────────────────────────────────────────
// Servo physical range: 0–44 degrees. 22 = horizontal center.
// The pulse mapping uses the full 0–180 scale because the servo hardware
// expects 500–2400us for its entire mechanical range.
int currentServoAngle = 22;

void writeServo(int angle) {
  angle = constrain(angle, 0, 44); // HARDWARE SAFETY LIMIT
  currentServoAngle = angle;
  int pulse_us = map(angle, 0, 180, 500, 2400);
  int duty = (pulse_us * 4095) / 20000;
  ledcWrite(SERVO_PIN, duty);
}

// ── Scissor Servo Control ────────────────────────────────────────────────────
// GPIO 23, LEDC Channel 6.  80° = OPEN (blades apart),  0° = CLOSED (cutting).
#define SCISSOR_OPEN   80
#define SCISSOR_CLOSED 0
#define SCISSOR_STEP_MS 15   // faster step for quick cut action

int currentScissorAngle = SCISSOR_OPEN;

void writeScissor(int angle) {
  angle = constrain(angle, 0, 80);
  currentScissorAngle = angle;
  int pulse_us = map(angle, 0, 180, 500, 2400);
  int duty = (pulse_us * 4095) / 20000;
  ledcWrite(SCISSOR_PIN, duty);
}

void smoothScissor(int target) {
  target = constrain(target, 0, 80);
  while (currentScissorAngle != target) {
    if (currentScissorAngle < target)
      writeScissor(currentScissorAngle + 1);
    else
      writeScissor(currentScissorAngle - 1);
    delay(SCISSOR_STEP_MS);
  }
}

void doCut() {
  Serial.println("[SCISSOR] Cutting...");
  writeScissor(SCISSOR_CLOSED);    // snap shut — full power
  delay(500);                      // hold closed so blades fully engage & cut
  writeScissor(SCISSOR_OPEN);      // reopen
  Serial.println("[SCISSOR] Cut complete.");
}

// ── UART Configuration ───────────────────────────────────────────────────────
#define UART_RX_PIN 16
#define UART_TX_PIN 17
#define UART_BAUD 115200

// ── BTS7960B Motor Driver Pins ───────────────────────────────────────────────
#define LEFT_FWD_PWM 25
#define LEFT_BWD_PWM 26
#define LEFT_EN 32

#define RIGHT_FWD_PWM 27
#define RIGHT_BWD_PWM 14
#define RIGHT_EN 33

#define PWM_FREQ 5000
#define PWM_BITS 8

// ── Ultrasonic Sensors ───────────────────────────────────────────────────────
#define FRONT_TRIG 4
#define FRONT_ECHO 5
#define LEFT_TRIG 18
#define LEFT_ECHO 19
#define RIGHT_TRIG 21
#define RIGHT_ECHO 22

#define OBSTACLE_STOP_MM 200 // Stop if front sensor < 20cm
#define SIDE_CORRECT_MM 150  // Steer correction if side sensor < 15cm

// ── State ────────────────────────────────────────────────────────────────────
enum DriveMode { MANUAL, AUTO };
enum ChassisState {
  IDLE,
  MOVING_FORWARD,
  MOVING_BACKWARD,
  TURNING_LEFT,
  TURNING_RIGHT
};

DriveMode mode = MANUAL; // Starts in Manual
ChassisState state = IDLE;
int currentSpeed = 0;

// Dual Buffers for Dual-Ear Listening
String piBuffer = "";
String pcBuffer = "";

volatile uint16_t distFront = 9999;
volatile uint16_t distLeft = 9999;
volatile uint16_t distRight = 9999;

TaskHandle_t TaskSensorsHandle;

// ── Function prototypes
// ───────────────────────────────────────────────────────
void setupMotors();
void setMotors(int leftPWM, int rightPWM);
void driveForward(int speed);
void driveBackward(int speed);
void turnLeft(int speed);
void turnRight(int speed);
void stopMotors();
uint16_t readUltrasonic(int trigPin, int echoPin);
void updateSensors();
void processCommand(const String &json);
void sendOK(const char *msg);
void sendError(const char *err);
void sendStatus();
void autoLoop();
void writeScissor(int angle);
void smoothScissor(int target);
void doCut();

// ── FreeRTOS Sensor Task (Runs on Core 0) ────────────────────────────────────
void TaskSensors(void *pvParameters) {
  for (;;) {
    distFront = readUltrasonic(FRONT_TRIG, FRONT_ECHO);
    vTaskDelay(pdMS_TO_TICKS(20)); // Yield to watchdog and avoid echo bounce
    distLeft = readUltrasonic(LEFT_TRIG, LEFT_ECHO);
    vTaskDelay(pdMS_TO_TICKS(20));
    distRight = readUltrasonic(RIGHT_TRIG, RIGHT_ECHO);
    vTaskDelay(pdMS_TO_TICKS(20));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.println("[ViniBot ESP32] Booting...");

  Serial2.begin(UART_BAUD, SERIAL_8N1, UART_RX_PIN, UART_TX_PIN);

  // 1. Initialize End-Effector Servo on Channel 0 (Timer 0, 50Hz, 12-bit)
  ledcAttachChannel(SERVO_PIN, 50, 12, 0);
  writeServo(22);

  // 1b. Initialize Scissor Servo on Channel 6 (Timer 3, 50Hz, 12-bit)
  // Uses a separate timer from the end-effector to avoid cross-channel jitter.
  ledcAttachChannel(SCISSOR_PIN, 50, 12, 6);
  writeScissor(SCISSOR_OPEN);  // start with blades open
  delay(500);                  // let servo settle before continuing

  // 2. Initialize Motors
  setupMotors();

  pinMode(FRONT_TRIG, OUTPUT);
  pinMode(FRONT_ECHO, INPUT);
  pinMode(LEFT_TRIG, OUTPUT);
  pinMode(LEFT_ECHO, INPUT);
  pinMode(RIGHT_TRIG, OUTPUT);
  pinMode(RIGHT_ECHO, INPUT);

  Serial.println("[ViniBot ESP32] Hardware pins configured.");

  // 3. START SENSOR THREAD ON CORE 0
  // This prevents pulseIn() from blocking UART communications on Core 1
  xTaskCreatePinnedToCore(TaskSensors,        // Task function
                          "TaskSensors",      // Task name
                          2048,               // Stack size
                          NULL,               // Parameters
                          1,                  // Priority
                          &TaskSensorsHandle, // Task handle
                          0                   // Run on Core 0
  );

  Serial.println(
      "[ViniBot ESP32] Ready. Listening to both Pi (UART2) and PC (USB).");
}

// ─────────────────────────────────────────────────────────────────────────────
void loop() {
  // Note: updateSensors() is no longer called here.
  // It runs completely asynchronously on Core 0!

  // 🚨 GLOBAL EMERGENCY BRAKE
  if (distFront < OBSTACLE_STOP_MM && state == MOVING_FORWARD) {
    stopMotors();
    Serial.println("[WARNING] Emergency Brake Applied!");
  }

  // 🌟 EAR 1: Listen to the Raspberry Pi (Serial2)
  while (Serial2.available()) {
    char c = Serial2.read();
    if (c == '\n' || c == '\r')
      continue;
    piBuffer += c;
    if (c == '}') {
      piBuffer.trim();
      if (piBuffer.startsWith("{"))
        processCommand(piBuffer);
      piBuffer = "";
    }
  }

  // 🌟 EAR 2: Listen to your PC / Arduino IDE (Serial)
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r')
      continue;
    pcBuffer += c;
    if (c == '}') {
      pcBuffer.trim();
      if (pcBuffer.startsWith("{"))
        processCommand(pcBuffer);
      pcBuffer = "";
    }
  }

  if (mode == AUTO) {
    autoLoop();
  }

  // Yield to FreeRTOS watchdog
  vTaskDelay(pdMS_TO_TICKS(10));
}

// ─────────────────────────────────────────────────────────────────────────────
void setupMotors() {
  pinMode(LEFT_EN, OUTPUT);
  digitalWrite(LEFT_EN, HIGH);
  pinMode(RIGHT_EN, OUTPUT);
  digitalWrite(RIGHT_EN, HIGH);
  // Explicitly assign motors to Channels 2, 3, 4, 5 (Timers 1 & 2)
  // This prevents them from sharing Timer 0 with the servo (Channel 0 & 1).
  ledcAttachChannel(LEFT_FWD_PWM, PWM_FREQ, PWM_BITS, 2);
  ledcAttachChannel(LEFT_BWD_PWM, PWM_FREQ, PWM_BITS, 3);
  ledcAttachChannel(RIGHT_FWD_PWM, PWM_FREQ, PWM_BITS, 4);
  ledcAttachChannel(RIGHT_BWD_PWM, PWM_FREQ, PWM_BITS, 5);
  stopMotors();
}

void setMotors(int leftPWM, int rightPWM) {
  if (leftPWM > 0) {
    ledcWrite(LEFT_FWD_PWM, leftPWM);
    ledcWrite(LEFT_BWD_PWM, 0);
  } else if (leftPWM < 0) {
    ledcWrite(LEFT_FWD_PWM, 0);
    ledcWrite(LEFT_BWD_PWM, -leftPWM);
  } else {
    ledcWrite(LEFT_FWD_PWM, 0);
    ledcWrite(LEFT_BWD_PWM, 0);
  }

  if (rightPWM > 0) {
    ledcWrite(RIGHT_FWD_PWM, rightPWM);
    ledcWrite(RIGHT_BWD_PWM, 0);
  } else if (rightPWM < 0) {
    ledcWrite(RIGHT_FWD_PWM, 0);
    ledcWrite(RIGHT_BWD_PWM, -rightPWM);
  } else {
    ledcWrite(RIGHT_FWD_PWM, 0);
    ledcWrite(RIGHT_BWD_PWM, 0);
  }
}

void driveForward(int speed) {
  currentSpeed = speed;
  state = MOVING_FORWARD;
  setMotors(speed, speed);
}
void driveBackward(int speed) {
  currentSpeed = speed;
  state = MOVING_BACKWARD;
  setMotors(-speed, -speed);
}
void turnLeft(int speed) {
  currentSpeed = speed;
  state = TURNING_LEFT;
  setMotors(-speed, speed);
}
void turnRight(int speed) {
  currentSpeed = speed;
  state = TURNING_RIGHT;
  setMotors(speed, -speed);
}
void stopMotors() {
  currentSpeed = 0;
  state = IDLE;
  setMotors(0, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
uint16_t readUltrasonic(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  long duration = pulseIn(echoPin, HIGH, 30000);
  if (duration == 0)
    return 9999;
  return (uint16_t)(duration * 0.1715);
}
// updateSensors() was removed because it is now handled perfectly by
// TaskSensors on Core 0.

// ─────────────────────────────────────────────────────────────────────────────
void autoLoop() {
  if (distFront < OBSTACLE_STOP_MM) {
    stopMotors();
    return;
  }
  // Reduced speeds for 30RPM motors to allow stable grape detection
  int leftPWM = 50;
  int rightPWM = 50;
  if (distLeft < SIDE_CORRECT_MM)
    leftPWM = 30;
  if (distRight < SIDE_CORRECT_MM)
    rightPWM = 30;

  state = MOVING_FORWARD;
  setMotors(leftPWM, rightPWM);
}

// ─────────────────────────────────────────────────────────────────────────────
void processCommand(const String &json) {
  Serial.println("[UART RX] " + json);

  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, json);
  if (err) {
    sendError("invalid_json");
    return;
  }

  const char *cmd = doc["cmd"];
  if (!cmd) {
    sendError("no_cmd_field");
    return;
  }

  if (mode == AUTO) {
    if (strcmp(cmd, "status") != 0 && strcmp(cmd, "mode") != 0) {
      mode = MANUAL;
      stopMotors();
      Serial.println(
          "[MODE] External command received. Switching to MANUAL override.");
    }
  }

  if (strcmp(cmd, "move") == 0) {
    const char *dir = doc["dir"] | "stop";
    // Default to 50 (reduced for 30RPM motors & grape detection) if no speed
    // provided
    int speed = doc["speed"].isNull() ? 50 : doc["speed"].as<int>();
    speed = constrain(speed, 0, 255);

    if (strcmp(dir, "forward") == 0) {
      if (distFront < OBSTACLE_STOP_MM) {
        sendError("obstacle_front");
        return;
      }
      driveForward(speed);
      sendOK("moving_forward");
    } else if (strcmp(dir, "backward") == 0) {
      driveBackward(speed);
      sendOK("moving_backward");
    } else if (strcmp(dir, "left") == 0) {
      turnLeft(speed);
      sendOK("turning_left");
    } else if (strcmp(dir, "right") == 0) {
      turnRight(speed);
      sendOK("turning_right");
    } else if (strcmp(dir, "stop") == 0) {
      stopMotors();
      sendOK("stopped");
    } else {
      sendError("unknown_dir");
    }
  } else if (strcmp(cmd, "stop") == 0) {
    stopMotors();
    sendOK("stopped");
  } else if (strcmp(cmd, "servo") == 0) {
    int angle = doc["angle"] | 22; // Default to horizontal center
    writeServo(angle);
    sendOK("servo_moved");
  } else if (strcmp(cmd, "mode") == 0) {
    const char *val = doc["value"] | "manual";
    if (strcmp(val, "auto") == 0) {
      mode = AUTO;
      sendOK("mode_auto");
      Serial.println("[MODE] AUTO");
    } else {
      mode = MANUAL;
      stopMotors();
      sendOK("mode_manual");
      Serial.println("[MODE] MANUAL");
    }
  } else if (strcmp(cmd, "status") == 0) {
    sendStatus();

  // ── Scissor commands ─────────────────────────────────────────────────────
  } else if (strcmp(cmd, "cut") == 0) {
    doCut();
    sendOK("cut_done");

  } else if (strcmp(cmd, "scissor") == 0) {
    int angle = doc["angle"] | 0;
    smoothScissor(angle);
    Serial.printf("[SCISSOR] Moved to %d°\n", currentScissorAngle);
    sendOK("scissor_moved");

  } else {
    sendError("unknown_cmd");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void sendOK(const char *msg) {
  String reply = "{\"ok\":true,\"msg\":\"";
  reply += msg;
  reply += "\"}";
  Serial2.println(reply);
  Serial.println("[UART TX] " + reply);
}

void sendError(const char *err) {
  String reply = "{\"ok\":false,\"error\":\"";
  reply += err;
  reply += "\"}";
  Serial2.println(reply);
  Serial.println("[UART TX] " + reply);
}

void sendStatus() {
  StaticJsonDocument<256> doc;
  doc["ok"] = true;
  const char *stateStr = "idle";
  switch (state) {
  case MOVING_FORWARD:
    stateStr = "moving_forward";
    break;
  case MOVING_BACKWARD:
    stateStr = "moving_backward";
    break;
  case TURNING_LEFT:
    stateStr = "turning_left";
    break;
  case TURNING_RIGHT:
    stateStr = "turning_right";
    break;
  default:
    stateStr = "idle";
    break;
  }
  doc["status"] = stateStr;
  doc["mode"] = (mode == AUTO) ? "auto" : "manual";
  doc["speed"] = currentSpeed;

  JsonObject sensors = doc.createNestedObject("sensors");
  sensors["front"] = distFront;
  sensors["left"] = distLeft;
  sensors["right"] = distRight;

  String out;
  serializeJson(doc, out);
  Serial2.println(out);
  Serial.println("[UART TX] " + out);
}