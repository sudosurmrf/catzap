#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>
#include <ArduinoJson.h>
#include <esp_system.h>
#include <stdarg.h>

// ===== WiFi credentials — update these =====
const char* ssid = "Rebellious Amish Family";
const char* password = "Lolcats1!";

// ===== Pin definitions =====
#define PAN_SERVO_PIN   18
#define TILT_SERVO_PIN  19
#define TRIGGER_SERVO_PIN 23
#define STATUS_LED_PIN   2

// ===== Trigger servo angles — tune these to your water gun =====
#define TRIGGER_REST_ANGLE  90    // servo position when NOT firing
#define TRIGGER_FIRE_ANGLE  180   // servo position that presses the trigger
#define TRIGGER_HOLD_MS     2000  // how long to hold the trigger pressed before releasing

// ===== Safe boot pose — matches CALIBRATION_HOME_PAN/TILT on the server =====
// IMPORTANT: tilt must NOT default to 90 on this rig — 90° is physically
// "straight down" and slams into a mechanical stop. 45° is the level pose.
// If the ESP32 reboots for any reason (watchdog, WiFi crash, power glitch),
// setup() writes these values to the servos, so the gun always lands here
// after a reset instead of flailing to whatever the previous default was.
#define BOOT_PAN_ANGLE   90
#define BOOT_TILT_ANGLE  45

Servo panServo;
Servo tiltServo;
Servo triggerServo;

// Servo poses are integer degrees end-to-end. The Python client snaps to
// whole degrees before sending, and Servo.write() takes an int, so carrying
// a float through the request handlers served no purpose — it just meant
// two extra conversions and an `(int)` cast at every write site.
int currentPan = BOOT_PAN_ANGLE;
int currentTilt = BOOT_TILT_ANGLE;

static bool g_trigger_active = false;       // true while trigger servo is held in fire position
static unsigned long g_trigger_release_ms = 0;  // millis() at which to release the trigger

// ───── Move-and-kill: zero LEDC duty after idle to stop jitter ─────
// Servos stay permanently attach()'d so the LEDC channel is never torn
// down. To silence jitter we set the channel duty to 0 (pin stays low,
// no pulses). servo.write() resumes correct output instantly — no
// re-init, no glitch pulse, no mechanical snap.
#define SERVO_IDLE_MS 1500

static unsigned long g_last_servo_cmd_ms = 0;
static bool g_servos_quiet = false;
static int g_pan_channel = -1;
static int g_tilt_channel = -1;

// ───── Log ring buffer ──────────────────────────────────────────────
// In-RAM circular buffer of recent log lines, exposed over HTTP at /logs.
// This is how we see what the ESP32 was doing before a crash or anomaly
// without needing a physical serial connection. Every logged event is
// also echoed to Serial so USB debugging still works when a cable is
// attached.
#define LOG_MAX_ENTRIES 64
#define LOG_MAX_LEN     96

struct LogEntry {
    unsigned long timestamp_ms;
    char message[LOG_MAX_LEN];
};

static LogEntry g_log_buffer[LOG_MAX_ENTRIES];
static int g_log_head = 0;     // next slot to write
static int g_log_count = 0;    // how many slots are populated (caps at LOG_MAX_ENTRIES)

static void logLine(const char* fmt, ...) {
    LogEntry& slot = g_log_buffer[g_log_head];
    slot.timestamp_ms = millis();

    va_list args;
    va_start(args, fmt);
    vsnprintf(slot.message, LOG_MAX_LEN, fmt, args);
    va_end(args);

    // Serial echo is gated behind ENABLE_SERIAL_LOG to avoid paying ~3-8ms
    // per logLine call at 115200 baud when no debugger is attached. The ring
    // buffer above is the primary log sink; Serial is only for USB debugging.
    // Re-enable by adding -D ENABLE_SERIAL_LOG to platformio.ini build_flags.
#ifdef ENABLE_SERIAL_LOG
    Serial.print("[");
    Serial.print(slot.timestamp_ms);
    Serial.print("] ");
    Serial.println(slot.message);
#endif

    g_log_head = (g_log_head + 1) % LOG_MAX_ENTRIES;
    if (g_log_count < LOG_MAX_ENTRIES) g_log_count++;
}

// Translate the ESP reset reason enum into a short string we can log.
// This is the most important diagnostic: every boot's first log line
// tells us *why* the ESP32 restarted, which is the key to diagnosing
// the wild-tilt-during-reboot behavior.
static const char* resetReasonStr(esp_reset_reason_t r) {
    switch (r) {
        case ESP_RST_POWERON:   return "POWERON";
        case ESP_RST_EXT:       return "EXT";
        case ESP_RST_SW:        return "SW";
        case ESP_RST_PANIC:     return "PANIC";
        case ESP_RST_INT_WDT:   return "INT_WDT";
        case ESP_RST_TASK_WDT:  return "TASK_WDT";
        case ESP_RST_WDT:       return "WDT";
        case ESP_RST_DEEPSLEEP: return "DEEPSLEEP";
        case ESP_RST_BROWNOUT:  return "BROWNOUT";
        case ESP_RST_SDIO:      return "SDIO";
        default:                return "UNKNOWN";
    }
}

WebServer server(80);

void handleAim() {
    if (server.method() != HTTP_POST) {
        server.send(405, "application/json", "{\"error\":\"Method not allowed\"}");
        return;
    }

    String body = server.arg("plain");
    JsonDocument doc;
    DeserializationError error = deserializeJson(doc, body);

    if (error) {
        logLine("reject: invalid JSON (len=%d)", body.length());
        server.send(400, "application/json", "{\"error\":\"Invalid JSON\"}");
        return;
    }

    // STRICT validation — if either field is missing or unparseable, reject
    // the request rather than silently falling back to currentPan/currentTilt.
    // The old `doc["pan"] | currentPan` pattern could produce 90/90 on any
    // corrupted packet after a reboot, which slams the tilt servo against
    // its mechanical stop. Better to drop the command and let the server retry.
    if (doc["pan"].isNull() || doc["tilt"].isNull()) {
        logLine("reject: missing pan or tilt (body=%s)", body.c_str());
        server.send(400, "application/json", "{\"error\":\"Missing pan or tilt\"}");
        return;
    }

    int pan = doc["pan"].as<int>();
    int tilt = doc["tilt"].as<int>();

    pan = constrain(pan, 0, 180);
    tilt = constrain(tilt, 0, 180);

    panServo.write(pan);
    tiltServo.write(tilt);
    currentPan = pan;
    currentTilt = tilt;
    g_last_servo_cmd_ms = millis();
    g_servos_quiet = false;

    logLine("aim: pan=%d tilt=%d", pan, tilt);

    String response;
    JsonDocument respDoc;
    respDoc["pan"] = pan;
    respDoc["tilt"] = tilt;
    serializeJson(respDoc, response);
    server.send(200, "application/json", response);
}

void handleFire() {
    if (server.method() != HTTP_POST) {
        server.send(405, "application/json", "{\"error\":\"Method not allowed\"}");
        return;
    }

    // Fire hold duration is fixed at TRIGGER_HOLD_MS. Any duration_ms field
    // in the request body is ignored — we always drive the trigger to the
    // fire angle and release it asynchronously from loop().
    logLine("fire: hold=%dms", TRIGGER_HOLD_MS);

    // If a fire is already in-flight, extend the hold rather than ignoring
    // the request. The server fires every ~2 s and we don't want missed shots.
    triggerServo.write(TRIGGER_FIRE_ANGLE);
    digitalWrite(STATUS_LED_PIN, HIGH);
    g_trigger_release_ms = millis() + TRIGGER_HOLD_MS;
    g_trigger_active = true;

    String response;
    JsonDocument respDoc;
    respDoc["fired"] = true;
    respDoc["duration_ms"] = TRIGGER_HOLD_MS;
    serializeJson(respDoc, response);
    server.send(200, "application/json", response);
}

void handleGoto() {
    if (server.method() != HTTP_POST) {
        server.send(405, "application/json", "{\"error\":\"Method not allowed\"}");
        return;
    }

    String body = server.arg("plain");
    JsonDocument doc;
    DeserializationError error = deserializeJson(doc, body);

    if (error) {
        logLine("reject: invalid JSON (len=%d)", body.length());
        server.send(400, "application/json", "{\"error\":\"Invalid JSON\"}");
        return;
    }

    // STRICT validation — if either field is missing or unparseable, reject
    // the request rather than silently falling back to currentPan/currentTilt.
    // The old `doc["pan"] | currentPan` pattern could produce the boot-time
    // defaults (90, 45) on any corrupted packet, which slams the tilt servo
    // against its mechanical stop. Better to drop the command and retry.
    if (doc["pan"].isNull() || doc["tilt"].isNull()) {
        logLine("reject: missing pan or tilt (body=%s)", body.c_str());
        server.send(400, "application/json", "{\"error\":\"Missing pan or tilt\"}");
        return;
    }

    int pan = doc["pan"].as<int>();
    int tilt = doc["tilt"].as<int>();

    pan = constrain(pan, 0, 180);
    tilt = constrain(tilt, 0, 180);

    panServo.write(pan);
    tiltServo.write(tilt);
    currentPan = pan;
    currentTilt = tilt;
    g_last_servo_cmd_ms = millis();
    g_servos_quiet = false;

    logLine("goto: pan=%d tilt=%d", pan, tilt);

    String response;
    JsonDocument respDoc;
    respDoc["pan"] = pan;
    respDoc["tilt"] = tilt;
    serializeJson(respDoc, response);
    server.send(200, "application/json", response);
}

void handleStop() {
    if (g_pan_channel >= 0) ledcWrite(g_pan_channel, 0);
    if (g_tilt_channel >= 0) ledcWrite(g_tilt_channel, 0);
    g_servos_quiet = true;
    logLine("stop: pwm silenced, pan=%d tilt=%d", currentPan, currentTilt);

    String response;
    JsonDocument doc;
    doc["stopped"] = true;
    doc["pan"] = currentPan;
    doc["tilt"] = currentTilt;
    serializeJson(doc, response);
    server.send(200, "application/json", response);
}

void handlePosition() {
    String response;
    JsonDocument doc;
    doc["pan"] = currentPan;
    doc["tilt"] = currentTilt;
    serializeJson(doc, response);
    server.send(200, "application/json", response);
}

void handleLogs() {
    // Emit the ring buffer as a JSON object with entries in chronological
    // order (oldest first). We also include uptime_ms at response time so
    // the caller can compute relative ages without tracking the ESP32's
    // boot epoch separately.
    JsonDocument doc;
    doc["uptime_ms"] = millis();
    doc["count"] = g_log_count;
    JsonArray arr = doc["lines"].to<JsonArray>();

    // Walk from oldest to newest. If we haven't filled the buffer yet,
    // start at 0; otherwise start at the current head (which is the
    // oldest slot, since it's the next one to be overwritten).
    int start_idx = (g_log_count < LOG_MAX_ENTRIES) ? 0 : g_log_head;
    for (int i = 0; i < g_log_count; i++) {
        int idx = (start_idx + i) % LOG_MAX_ENTRIES;
        JsonObject o = arr.add<JsonObject>();
        o["ts"] = g_log_buffer[idx].timestamp_ms;
        o["msg"] = g_log_buffer[idx].message;
    }

    String response;
    serializeJson(doc, response);
    server.send(200, "application/json", response);
}

void handleHealth() {
    String response;
    JsonDocument doc;
    doc["status"] = "ok";
    doc["pan"] = currentPan;
    doc["tilt"] = currentTilt;
    serializeJson(doc, response);
    server.send(200, "application/json", response);
}

void setup() {
    Serial.begin(115200);
    // Give the serial port a moment to settle before the first log
    delay(100);

    // Boot diagnostics use Serial.println directly (bypassing logLine) so
    // they're always visible in the serial monitor without paying the per-
    // command UART cost in steady-state operation. logLine still records
    // these into the ring buffer for HTTP /logs retrieval.
    Serial.println();
    Serial.println("=== ESP32 actuator booting ===");

    esp_reset_reason_t reason = esp_reset_reason();
    Serial.printf("boot: reset_reason=%s\n", resetReasonStr(reason));
    logLine("boot: reset_reason=%s", resetReasonStr(reason));

    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, LOW);

    g_pan_channel = panServo.attach(PAN_SERVO_PIN);
    g_tilt_channel = tiltServo.attach(TILT_SERVO_PIN);
    triggerServo.attach(TRIGGER_SERVO_PIN);
    g_last_servo_cmd_ms = millis();
    panServo.write(BOOT_PAN_ANGLE);
    tiltServo.write(BOOT_TILT_ANGLE);
    triggerServo.write(TRIGGER_REST_ANGLE);
    Serial.printf("servos attached: pan=%d tilt=%d trigger=%d home=(%d,%d)\n",
                  PAN_SERVO_PIN, TILT_SERVO_PIN, TRIGGER_SERVO_PIN,
                  BOOT_PAN_ANGLE, BOOT_TILT_ANGLE);
    logLine("servos attached: pan=%d tilt=%d trigger=%d home=(%d,%d)",
            PAN_SERVO_PIN, TILT_SERVO_PIN, TRIGGER_SERVO_PIN,
            BOOT_PAN_ANGLE, BOOT_TILT_ANGLE);

    // Keep WiFi alive across transient drops so a temporary disconnection
    // doesn't snowball into a watchdog reset.
    WiFi.setAutoReconnect(true);
    WiFi.persistent(true);
    WiFi.begin(ssid, password);
    Serial.printf("wifi connecting to '%s'", ssid);
    logLine("wifi connecting to '%s'", ssid);
    unsigned long wifi_start = millis();
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    IPAddress ip = WiFi.localIP();
    Serial.printf("wifi connected: ip=%d.%d.%d.%d (took %lu ms)\n",
                  ip[0], ip[1], ip[2], ip[3], millis() - wifi_start);
    logLine("wifi connected: ip=%d.%d.%d.%d (took %lu ms)",
            ip[0], ip[1], ip[2], ip[3], millis() - wifi_start);

    server.on("/aim", handleAim);
    server.on("/goto", handleGoto);
    server.on("/fire", handleFire);
    server.on("/stop", handleStop);
    server.on("/position", HTTP_GET, handlePosition);
    server.on("/health", handleHealth);
    server.on("/logs", HTTP_GET, handleLogs);
    server.begin();
    Serial.println("http server started on port 80");
    Serial.println("=== boot complete ===");
    logLine("http server started on port 80");
}

void loop() {
    // If WiFi drops, attempt reconnection every 5 s without blocking the
    // watchdog. Without this, a temporary disconnect would leave the web
    // server in a stuck state and eventually cause a watchdog reset — which
    // in turn would re-run setup() and snap the servos to the boot pose.
    if (WiFi.status() != WL_CONNECTED) {
        static unsigned long lastReconnectAttempt = 0;
        unsigned long now = millis();
        if (now - lastReconnectAttempt > 5000) {
            lastReconnectAttempt = now;
            logLine("wifi disconnected, attempting reconnect");
            WiFi.reconnect();
        }
        delay(100);  // yield to the WiFi task, keep watchdog fed
        return;
    }

    // Non-blocking trigger release: check if the hold period has elapsed.
    // The (long) cast makes the subtraction safe against millis() rollover.
    if (g_trigger_active && (long)(millis() - g_trigger_release_ms) >= 0) {
        triggerServo.write(TRIGGER_REST_ANGLE);
        digitalWrite(STATUS_LED_PIN, LOW);
        g_trigger_active = false;
        logLine("trigger released");
    }

    // Move-and-kill: zero LEDC duty after idle to silence jitter.
    // Servos stay attached — only the PWM output goes quiet. The next
    // servo.write() sets the correct duty instantly, no re-init needed.
    if (!g_servos_quiet &&
        (long)(millis() - g_last_servo_cmd_ms) >= SERVO_IDLE_MS) {
        if (g_pan_channel >= 0) ledcWrite(g_pan_channel, 0);
        if (g_tilt_channel >= 0) ledcWrite(g_tilt_channel, 0);
        g_servos_quiet = true;
        logLine("idle quiet: pan=%d tilt=%d", currentPan, currentTilt);
    }

    server.handleClient();
}
