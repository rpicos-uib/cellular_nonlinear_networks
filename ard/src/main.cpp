#include "esp_camera.h"
#include <WiFi.h>

#define CAMERA_MODEL_AI_THINKER

#include "camera_pins.h"


void startCameraServer();

void setup() {
  Serial.begin(115200);
  
 // Serial.begin(921600);
  Serial.setDebugOutput(true);
  Serial.println();



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
  //config.pin_sscb_sda = SIOD_GPIO_NUM;
  //config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.fb_location = CAMERA_FB_IN_PSRAM;


  if(psramFound()){
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } 


#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t* s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);        // flip it back
    s->set_brightness(s, 1);   // up the brightness just a bit
    s->set_saturation(s, -2);  // lower the saturation
  }


#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 0);
#endif


// Network settings. It will connect to the strongest network
const char* ssid1 = "wifi1";
const char* password1 = "pass1";
const char* ssid2 = "wifi2";
const char* password2 = "pass2";


int numNetworks = WiFi.scanNetworks();
int strongestNetwork = -1;
int strongestRSSI = -100;

for (int i = 0; i < numNetworks; i++) {
  String ssid = WiFi.SSID(i);
  int rssi = WiFi.RSSI(i);
  if (ssid == ssid1 || ssid == ssid2) {
    if (rssi > strongestRSSI) {
      strongestNetwork = i;
      strongestRSSI = rssi;
    }
    Serial.print(ssid);
      Serial.print(": ");
      Serial.println(rssi);
  }
}

if (strongestNetwork == -1) {
  Serial.println("Both networks not available");
  return;
} else {
  String ssid = WiFi.SSID(strongestNetwork);
  String password;
  if(ssid == ssid1) password = password1;
  if(ssid == ssid2) password = password2;
  WiFi.begin(ssid.c_str(), password.c_str());

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");
  startCameraServer();
  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");
  Serial.println(__cplusplus);
}

}


void loop() {
  // put your main code here, to run repeatedly:
  delay(1000000);
}