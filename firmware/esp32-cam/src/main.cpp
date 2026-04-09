#include <Arduino.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <WebServer.h>

// ===== WiFi credentials — update these =====
const char* ssid = "Rebellious Amish Family";
const char* password = "Lolcats1!";

// ===== AI-Thinker ESP32-CAM pin definitions =====
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Built-in LED on GPIO 33 (active LOW on AI-Thinker board)
#define LED_PIN           33

WebServer server(81);

void handleStream() {
    WiFiClient client = server.client();

    String response = "HTTP/1.1 200 OK\r\n";
    response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    client.print(response);

    while (client.connected()) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            break;
        }

        String header = "--frame\r\n";
        header += "Content-Type: image/jpeg\r\n";
        header += "Content-Length: " + String(fb->len) + "\r\n\r\n";

        client.print(header);
        client.write(fb->buf, fb->len);
        client.print("\r\n");

        esp_camera_fb_return(fb);

        if (!client.connected()) break;
    }
}

void handleHealth() {
    server.send(200, "application/json", "{\"status\":\"ok\"}");
}

void setup() {
    Serial.begin(115200);
    Serial.println("CatZap ESP32-CAM starting...");

    // LED setup — blink to show we're alive
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);  // ON (active low)

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_VGA;  // 640x480
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        // Fast blink = camera error
        while (true) {
            digitalWrite(LED_PIN, LOW);
            delay(100);
            digitalWrite(LED_PIN, HIGH);
            delay(100);
        }
    }

    // Slow blink = connecting to WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        digitalWrite(LED_PIN, LOW);
        delay(250);
        digitalWrite(LED_PIN, HIGH);
        delay(250);
    }

    // Solid ON = connected and ready
    digitalWrite(LED_PIN, LOW);

    server.on("/stream", HTTP_GET, handleStream);
    server.on("/health", HTTP_GET, handleHealth);
    server.begin();

    // Triple blink = all good, server running
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(150);
        digitalWrite(LED_PIN, LOW);
        delay(150);
    }
    // LED stays ON
    digitalWrite(LED_PIN, LOW);
}

void loop() {
    server.handleClient();
}
