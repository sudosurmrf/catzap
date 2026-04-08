#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>
#include <ArduinoJson.h>

// ===== WiFi credentials — update these =====
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ===== Pin definitions =====
#define PAN_SERVO_PIN   18
#define TILT_SERVO_PIN  19
#define SOLENOID_PIN    23
#define STATUS_LED_PIN   2

Servo panServo;
Servo tiltServo;

float currentPan = 90.0;
float currentTilt = 90.0;

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
        server.send(400, "application/json", "{\"error\":\"Invalid JSON\"}");
        return;
    }

    float pan = doc["pan"] | currentPan;
    float tilt = doc["tilt"] | currentTilt;

    pan = constrain(pan, 0.0, 180.0);
    tilt = constrain(tilt, 0.0, 180.0);

    panServo.write((int)pan);
    tiltServo.write((int)tilt);
    currentPan = pan;
    currentTilt = tilt;

    Serial.printf("Aimed to pan=%.1f, tilt=%.1f\n", pan, tilt);

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

    String body = server.arg("plain");
    JsonDocument doc;
    deserializeJson(doc, body);

    int duration_ms = doc["duration_ms"] | 200;
    duration_ms = constrain(duration_ms, 50, 2000);

    Serial.printf("FIRE! Duration: %dms\n", duration_ms);

    digitalWrite(SOLENOID_PIN, HIGH);
    digitalWrite(STATUS_LED_PIN, HIGH);
    delay(duration_ms);
    digitalWrite(SOLENOID_PIN, LOW);
    digitalWrite(STATUS_LED_PIN, LOW);

    String response;
    JsonDocument respDoc;
    respDoc["fired"] = true;
    respDoc["duration_ms"] = duration_ms;
    serializeJson(respDoc, response);
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
    Serial.println("CatZap Actuator starting...");

    pinMode(SOLENOID_PIN, OUTPUT);
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(SOLENOID_PIN, LOW);
    digitalWrite(STATUS_LED_PIN, LOW);

    panServo.attach(PAN_SERVO_PIN);
    tiltServo.attach(TILT_SERVO_PIN);
    panServo.write(90);
    tiltServo.write(90);

    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("Connected! IP address: ");
    Serial.println(WiFi.localIP());

    server.on("/aim", handleAim);
    server.on("/fire", handleFire);
    server.on("/health", handleHealth);
    server.begin();
    Serial.println("Actuator server started on port 80");
}

void loop() {
    server.handleClient();
}
