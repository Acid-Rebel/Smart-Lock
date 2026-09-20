#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "img_converters.h"

#include "FS.h"
#include "SD_MMC.h"

#include "mqtt_client.h"
#include "esp_crt_bundle.h"

#include <ArduinoJson.h>
#include <math.h>
#include <time.h>

// ESP-DL
#include "human_face_detect_msr01.hpp"
#include "human_face_detect_mnp01.hpp"
#include "face_recognition_112_v1_s8.hpp"

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>

using namespace dl::detect;

// ============================================================
// OLED
// ============================================================

#define OLED_SDA   14
#define OLED_SCL   21
#define OLED_ADDR  0x3C

Adafruit_SH1106G display(128, 64, &Wire, -1);


// ============================================================
// WIFI
// ============================================================

#define WIFI_SSID       "Free net!!"
#define WIFI_PASSWORD   "Ajeth@123"


// ============================================================
// MQTT
// ============================================================

#define MQTT_URI "wss://iot.coreflux.cloud:443/mqtt"

#define TOPIC_REGISTER_CMD \
  "847291/583104/command/register"

#define TOPIC_DELETE_CMD \
  "847291/583104/command/delete"

#define TOPIC_UNLOCK_CMD \
  "847291/583104/command/unlock"

#define TOPIC_REGISTRATION_EVENT \
  "847291/583104/event/registration"

#define TOPIC_DELETION_EVENT \
  "847291/583104/event/deletion"

#define TOPIC_LOCK_EVENT \
  "847291/583104/event/lock"

#define TOPIC_AUTH_EVENT \
  "847291/583104/event/auth"

#define TOPIC_AUTH_IMAGE \
  "847291/583104/event/auth/image"


esp_mqtt_client_handle_t mqttClient = nullptr;

volatile bool mqttConnected = false;


// ============================================================
// EVENT LOGGING / CLOCK
// ============================================================

// Structured JSON Lines log. One event per line.
// Intruder images are NOT stored on the SD card.
#define EVENT_LOG_PATH "/esp32/events.jsonl"
#define USERS_JSON_PATH "/esp32/registered_users.json"

bool sdReady = false;
bool timeSynchronized = false;


// ============================================================
// BUTTON
// ============================================================

#define BUTTON_PIN 1

#define BUTTON_DEBOUNCE_MS 50

bool lastButtonReading = HIGH;
bool stableButtonState = HIGH;
unsigned long lastButtonChange = 0;


// ============================================================
// COMMON DATABASE CONSTANTS
// ============================================================

#define MAX_USERS                 10
#define MAX_NAME_LENGTH           32
#define EMBEDDING_SIZE            512
#define MAX_EMBEDDINGS_PER_USER   10


// ============================================================
// SOLENOID LOCK
// ============================================================

// LOW  = LOCKED
// HIGH = UNLOCKED
//
// This GPIO drives the MOSFET/transistor driver input.
// Do NOT connect the 12 V solenoid directly to the GPIO.
#define LOCK_PIN 2

#define AUTO_LOCK_TIME_MS 10000UL

bool lockIsUnlocked = false;
unsigned long unlockStartedAt = 0;
int lastUnlockDisplaySecond = -1;
char unlockedUserName[MAX_NAME_LENGTH] = {0};


// ============================================================
// CAMERA PINS
// ============================================================

#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1

#define XCLK_GPIO_NUM    15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5

#define Y9_GPIO_NUM      16
#define Y8_GPIO_NUM      17
#define Y7_GPIO_NUM      18
#define Y6_GPIO_NUM      12
#define Y5_GPIO_NUM      10
#define Y4_GPIO_NUM       8
#define Y3_GPIO_NUM       9
#define Y2_GPIO_NUM      11

#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM    13


// ============================================================
// DATABASE
// ============================================================

#define DB_PATH       "/esp32/face_db.bin"
#define DB_TEMP_PATH  "/esp32/face_db.tmp"

#define DB_MAGIC      0x534C4442
#define DB_VERSION    1
struct FaceUser {

  int32_t id;

  char name[MAX_NAME_LENGTH];

  uint32_t embeddingCount;

  float embeddings[
    MAX_EMBEDDINGS_PER_USER
  ][EMBEDDING_SIZE];
};


struct FaceDatabase {

  uint32_t magic;

  uint32_t version;

  uint32_t userCount;

  int32_t nextID;

  FaceUser users[MAX_USERS];
};


// Database is large, so put it in PSRAM.
FaceDatabase *database = nullptr;

// Authentication embeddings are also large (512 floats each).
// Keep these persistent buffers in PSRAM instead of the loopTask stack.
// This prevents stack-canary resets during second-frame verification.
float *authEmbeddingPSRAM = nullptr;
float *verifyEmbeddingPSRAM = nullptr;

bool cameraInitialized = false;


// ============================================================
// FACE AI
// ============================================================

HumanFaceDetectMSR01 detector1(
  0.1F,
  0.3F,
  10,
  0.4F
);

HumanFaceDetectMNP01 detector2(
  0.1F,
  0.3F,
  10
);

FaceRecognition112V1S8 recognizer;


// Authentication threshold.
#define FACE_THRESHOLD 0.60F


// ============================================================
// COMMAND QUEUE
// ============================================================

enum PendingCommandType {

  CMD_NONE,

  CMD_REGISTER,

  CMD_DELETE,

  CMD_UNLOCK
};


volatile PendingCommandType pendingCommand =
  CMD_NONE;

int32_t pendingID = -1;

char pendingName[MAX_NAME_LENGTH] = {0};


// ============================================================
// BUSY STATE
// ============================================================

volatile bool deviceBusy = false;


// ============================================================
// FORWARD DECLARATIONS USED BY LOCK CONTROL
// ============================================================

bool publishJSON(
  const char *topic,
  JsonDocument &doc
);

void oledMessage(
  const char *line1,
  const char *line2 = nullptr,
  const char *line3 = nullptr
);

bool initCamera();
void stopCamera();

bool syncSystemClock();
const char *getDeviceId();
void addEventMetadata(JsonDocument &doc);
bool appendJsonLog(JsonDocument &doc);
bool appendMqttLog(const char *topic, JsonDocument &payload);
bool saveRegisteredUsersFile();
void logSystemEvent(const char *eventName, const char *detail = nullptr);
void logMqttCommandReceived(const char *topic, const char *payload);


// ============================================================
// EVENT LOGGING / CLOCK HELPERS
// ============================================================

const char *getDeviceId()
{
  static char deviceId[40] = {0};

  if (deviceId[0] == '\0') {
    uint64_t chipid = ESP.getEfuseMac();

    snprintf(
      deviceId,
      sizeof(deviceId),
      "ESP32CAM_%04X%08X",
      (uint16_t)(chipid >> 32),
      (uint32_t)chipid
    );
  }

  return deviceId;
}


void addEventMetadata(JsonDocument &doc)
{
  doc["device_id"] = getDeviceId();

  time_t now = time(nullptr);

  if (now >= 1700000000) {
    struct tm utcTime;
    gmtime_r(&now, &utcTime);

    char timestamp[32];

    if (strftime(
          timestamp,
          sizeof(timestamp),
          "%Y-%m-%dT%H:%M:%SZ",
          &utcTime
        ) > 0) {
      doc["timestamp"] = timestamp;
      doc["timestamp_epoch"] = (uint64_t)now;
      return;
    }
  }

  doc["timestamp"] = nullptr;
  doc["timestamp_epoch"] = 0;
  doc["timestamp_source"] = "unsynchronized";
}


bool appendJsonLog(JsonDocument &doc)
{
  if (!sdReady) {
    return false;
  }

  File logFile = SD_MMC.open(
    EVENT_LOG_PATH,
    FILE_APPEND
  );

  if (!logFile) {
    Serial.println("SD LOG: failed to open event log.");
    return false;
  }

  size_t written = serializeJson(doc, logFile);
  logFile.println();
  logFile.flush();
  logFile.close();

  if (written == 0) {
    Serial.println("SD LOG: failed to serialize event.");
    return false;
  }

  return true;
}


bool appendMqttLog(const char *topic, JsonDocument &payload)
{
  if (!sdReady || !topic) {
    return false;
  }

  File logFile = SD_MMC.open(
    EVENT_LOG_PATH,
    FILE_APPEND
  );

  if (!logFile) {
    Serial.println("SD LOG: failed to open MQTT event log.");
    return false;
  }

  // Known MQTT topics contain no quotation marks, so this simple wrapper
  // is sufficient and avoids another large temporary JSON document.
  logFile.print("{\"topic\":\"");
  logFile.print(topic);
  logFile.print("\",\"payload\":");
  serializeJson(payload, logFile);
  logFile.println("}");
  logFile.flush();
  logFile.close();

  return true;
}



bool saveRegisteredUsersFile()
{
  if (!sdReady || database == nullptr) {
    return false;
  }

  // This file intentionally contains only user metadata.
  // Face embeddings remain in face_db.bin and are never exposed to the frontend.
  StaticJsonDocument<4096> doc;

  doc["device_id"] = getDeviceId();
  doc["user_count"] = database->userCount;

  time_t now = time(nullptr);
  if (now >= 1700000000) {
    struct tm utcTime;
    gmtime_r(&now, &utcTime);
    char timestamp[32];
    if (strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utcTime) > 0) {
      doc["updated_at"] = timestamp;
      doc["updated_at_epoch"] = (uint64_t)now;
    }
  } else {
    doc["updated_at"] = nullptr;
    doc["updated_at_epoch"] = 0;
    doc["timestamp_source"] = "unsynchronized";
  }

  JsonArray users = doc["users"].to<JsonArray>();

  for (uint32_t i = 0; i < database->userCount; ++i) {
    FaceUser &user = database->users[i];

    JsonObject item = users.add<JsonObject>();
    item["id"] = user.id;
    item["name"] = user.name;
    item["embedding_count"] = user.embeddingCount;
  }

  File file = SD_MMC.open(USERS_JSON_PATH, FILE_WRITE);
  if (!file) {
    Serial.println("SD USERS: failed to open registered users file.");
    return false;
  }

  size_t written = serializeJsonPretty(doc, file);
  file.println();
  file.flush();
  file.close();

  if (written == 0) {
    Serial.println("SD USERS: failed to write registered users file.");
    return false;
  }

  Serial.printf("SD USERS: wrote %lu registered user(s) to %s\n",
                (unsigned long)database->userCount,
                USERS_JSON_PATH);
  return true;
}


void logSystemEvent(const char *eventName, const char *detail)
{
  StaticJsonDocument<512> doc;

  addEventMetadata(doc);
  doc["event"] = eventName ? eventName : "system";

  if (detail && detail[0]) {
    doc["detail"] = detail;
  }

  appendJsonLog(doc);
}


void logMqttCommandReceived(
  const char *topic,
  const char *payload
)
{
  StaticJsonDocument<768> doc;

  addEventMetadata(doc);
  doc["event"] = "mqtt_command_received";
  doc["topic"] = topic ? topic : "";
  doc["payload"] = payload ? payload : "";

  appendJsonLog(doc);
}


bool syncSystemClock()
{
  Serial.println("NTP: synchronizing UTC clock...");

  configTime(
    0,
    0,
    "pool.ntp.org",
    "time.google.com",
    "time.cloudflare.com"
  );

  time_t now = time(nullptr);

  for (int attempt = 0; attempt < 20 && now < 1700000000; ++attempt) {
    delay(500);
    now = time(nullptr);
  }

  if (now < 1700000000) {
    timeSynchronized = false;
    Serial.println("NTP: synchronization failed; timestamps will be marked unsynchronized.");
    logSystemEvent("time_sync", "failed");
    return false;
  }

  timeSynchronized = true;

  struct tm utcTime;
  gmtime_r(&now, &utcTime);

  char timestamp[32];
  strftime(
    timestamp,
    sizeof(timestamp),
    "%Y-%m-%dT%H:%M:%SZ",
    &utcTime
  );

  Serial.printf("NTP: synchronized UTC = %s\n", timestamp);
  logSystemEvent("time_sync", "success");

  return true;
}


// ============================================================
// SOLENOID LOCK CONTROL
// ============================================================

void publishLockEvent(const char *status, const char *source)
{
  StaticJsonDocument<192> doc;

  doc["status"] = status;

  if (source && source[0]) {
    doc["source"] = source;
  }

  publishJSON(
    TOPIC_LOCK_EVENT,
    doc
  );
}


void lockDoor(const char *source = "auto")
{
  digitalWrite(LOCK_PIN, LOW);

  lockIsUnlocked = false;
  unlockStartedAt = 0;
  lastUnlockDisplaySecond = -1;
  unlockedUserName[0] = '\0';

  Serial.println();
  Serial.println("LOCK: LOCKED");
  Serial.printf("LOCK: GPIO %d = LOW\n", LOCK_PIN);

  oledMessage("LOCKED");

  publishLockEvent(
    "locked",
    source
  );

  // Resume camera only after the door is locked.
  if (!cameraInitialized) {
    delay(100);
    if (!initCamera()) {
      Serial.println("Camera: failed to restart after locking.");
    }
  }
}


void oledUnlockCountdown(int secondsRemaining)
{
  char line2[24];
  char line3[24];

  if (unlockedUserName[0]) {
    snprintf(line2, sizeof(line2), "%s", unlockedUserName);
  }
  else {
    snprintf(line2, sizeof(line2), "MANUAL OPEN");
  }

  snprintf(line3, sizeof(line3), "LOCK:%d SEC", secondsRemaining);

  oledMessage("UNLOCKED", line2, line3);
}


void unlockDoor(const char *source = "face", const char *userName = nullptr)
{
  digitalWrite(LOCK_PIN, HIGH);

  lockIsUnlocked = true;
  unlockStartedAt = millis();
  lastUnlockDisplaySecond = -1;

  memset(unlockedUserName, 0, sizeof(unlockedUserName));
  if (userName && userName[0]) {
    strncpy(
      unlockedUserName,
      userName,
      MAX_NAME_LENGTH - 1
    );
  }

  Serial.println();
  Serial.println("LOCK: UNLOCKED");
  Serial.printf("LOCK: GPIO %d = HIGH\n", LOCK_PIN);
  Serial.printf(
    "LOCK: Auto-lock in %lu seconds\n",
    AUTO_LOCK_TIME_MS / 1000UL
  );

  oledUnlockCountdown(
    (int)(AUTO_LOCK_TIME_MS / 1000UL)
  );

  publishLockEvent(
    "unlocked",
    source
  );

  // Do not leave the camera DMA running while no code is consuming
  // frames. That was causing EV-VSYNC-OVF and watchdog resets.
  stopCamera();
}


void serviceAutoLock()
{
  if (!lockIsUnlocked) {
    return;
  }

  unsigned long elapsed = millis() - unlockStartedAt;

  if (elapsed >= AUTO_LOCK_TIME_MS) {
    Serial.println(
      "LOCK: 10 seconds elapsed. Auto-locking."
    );

    lockDoor("auto");
    return;
  }

  unsigned long remainingMs =
    AUTO_LOCK_TIME_MS - elapsed;

  // Ceiling division so the display starts at 10 and reaches 1
  // before the lock closes at the end of the 10-second interval.
  int remainingSeconds =
    (int)((remainingMs + 999UL) / 1000UL);

  if (remainingSeconds != lastUnlockDisplaySecond) {
    lastUnlockDisplaySecond = remainingSeconds;

    oledUnlockCountdown(remainingSeconds);

    Serial.printf(
      "LOCK: %d seconds remaining\n",
      remainingSeconds
    );
  }
}

// ============================================================
// DATABASE
// ============================================================

void initializeDatabase()
{
  memset(
    database,
    0,
    sizeof(FaceDatabase)
  );

  database->magic =
    DB_MAGIC;

  database->version =
    DB_VERSION;

  database->userCount =
    0;

  database->nextID =
    1;
}


int findUserIndex(int32_t id)
{
  if (!database) {
    return -1;
  }

  for (
    uint32_t i = 0;
    i < database->userCount;
    i++
  ) {

    if (
      database->users[i].id == id
    ) {

      return (int)i;
    }
  }

  return -1;
}


void printDatabase()
{
  Serial.println();
  Serial.println(
    "========== FACE DATABASE =========="
  );

  Serial.printf(
    "Users: %lu\n",
    (unsigned long)database->userCount
  );

  for (
    uint32_t i = 0;
    i < database->userCount;
    i++
  ) {

    FaceUser &user =
      database->users[i];

    Serial.printf(
      "ID=%ld | Name=%s | Embeddings=%lu\n",
      (long)user.id,
      user.name,
      (unsigned long)user.embeddingCount
    );
  }

  Serial.println(
    "==================================="
  );
}


// ============================================================
// SAVE DATABASE
// ============================================================

bool saveDatabase()
{
  Serial.println(
    "Saving face database..."
  );


  File file =
    SD_MMC.open(
      DB_TEMP_PATH,
      FILE_WRITE
    );


  if (!file) {

    Serial.println(
      "ERROR: Could not open temporary DB file."
    );

    return false;
  }


  size_t written =
    file.write(
      (uint8_t *)database,
      sizeof(FaceDatabase)
    );


  file.flush();
  file.close();


  if (
    written != sizeof(FaceDatabase)
  ) {

    Serial.printf(
      "ERROR: Database write incomplete: %u / %u\n",
      (unsigned)written,
      (unsigned)sizeof(FaceDatabase)
    );

    SD_MMC.remove(
      DB_TEMP_PATH
    );

    return false;
  }


  File verify =
    SD_MMC.open(
      DB_TEMP_PATH,
      FILE_READ
    );


  if (!verify) {

    Serial.println(
      "ERROR: Could not verify DB."
    );

    SD_MMC.remove(
      DB_TEMP_PATH
    );

    return false;
  }


  size_t fileSize =
    verify.size();

  verify.close();


  if (
    fileSize != sizeof(FaceDatabase)
  ) {

    Serial.printf(
      "ERROR: DB size mismatch: %u / %u\n",
      (unsigned)fileSize,
      (unsigned)sizeof(FaceDatabase)
    );

    SD_MMC.remove(
      DB_TEMP_PATH
    );

    return false;
  }


  SD_MMC.remove(
    DB_PATH
  );


  if (
    !SD_MMC.rename(
      DB_TEMP_PATH,
      DB_PATH
    )
  ) {

    Serial.println(
      "ERROR: Could not rename DB."
    );

    return false;
  }


  Serial.println(
    "Database saved successfully."
  );

  if (!saveRegisteredUsersFile()) {
    Serial.println("WARNING: Registered users JSON was not updated.");
  }

  return true;
}


// ============================================================
// LOAD DATABASE
// ============================================================

bool loadDatabase()
{
  File file =
    SD_MMC.open(
      DB_PATH,
      FILE_READ
    );


  if (!file) {

    Serial.println(
      "No database found."
    );

    Serial.println(
      "Creating new database."
    );

    initializeDatabase();

    return saveDatabase();
  }


  if (
    file.size() != sizeof(FaceDatabase)
  ) {

    Serial.println(
      "ERROR: Existing database has wrong size."
    );

    file.close();

    initializeDatabase();

    return saveDatabase();
  }


  size_t readBytes =
    file.read(
      (uint8_t *)database,
      sizeof(FaceDatabase)
    );


  file.close();


  if (
    readBytes != sizeof(FaceDatabase)
  ) {

    Serial.println(
      "ERROR: Could not read complete database."
    );

    initializeDatabase();

    return saveDatabase();
  }


  if (
    database->magic != DB_MAGIC ||
    database->version != DB_VERSION
  ) {

    Serial.println(
      "ERROR: Invalid database header."
    );

    initializeDatabase();

    return saveDatabase();
  }


  if (
    database->userCount > MAX_USERS
  ) {

    Serial.println(
      "ERROR: Invalid user count."
    );

    initializeDatabase();

    return saveDatabase();
  }


  Serial.println(
    "Database loaded successfully."
  );

  if (!saveRegisteredUsersFile()) {
    Serial.println("WARNING: Could not synchronize registered users JSON.");
  }

  printDatabase();

  return true;
}


// ============================================================
// MQTT JSON PUBLISH
// ============================================================

bool publishJSON(
  const char *topic,
  JsonDocument &doc
)
{
  if (!topic) {
    return false;
  }

  // Build a new payload so the caller's document does not need extra space.
  // Every JSON MQTT event receives the same device/timestamp metadata.
  StaticJsonDocument<768> outgoing;

  addEventMetadata(outgoing);

  for (JsonPair kv : doc.as<JsonObject>()) {
    outgoing[kv.key()] = kv.value();
  }

  outgoing["topic"] = topic;

  // Always persist the event locally, even when MQTT is disconnected.
  appendMqttLog(topic, outgoing);

  char buffer[768];

  size_t len = serializeJson(
    outgoing,
    buffer,
    sizeof(buffer)
  );

  if (len == 0 || len >= sizeof(buffer)) {
    Serial.println("MQTT JSON serialization failed or payload is too large.");
    return false;
  }

  if (
    !mqttConnected ||
    mqttClient == nullptr
  ) {
    Serial.println(
      "MQTT not connected; JSON not published. Event kept on SD log."
    );
    return false;
  }

  int result = esp_mqtt_client_publish(
    mqttClient,
    topic,
    buffer,
    (int)len,
    0,
    0
  );

  Serial.printf(
    "MQTT JSON publish result: %d\n",
    result
  );

  return result >= 0;
}

// ============================================================
// CAMERA INITIALIZATION
// ============================================================

bool initCamera()
{
  camera_config_t config;


  config.ledc_channel =
    LEDC_CHANNEL_0;

  config.ledc_timer =
    LEDC_TIMER_0;


  config.pin_d0 =
    Y2_GPIO_NUM;

  config.pin_d1 =
    Y3_GPIO_NUM;

  config.pin_d2 =
    Y4_GPIO_NUM;

  config.pin_d3 =
    Y5_GPIO_NUM;

  config.pin_d4 =
    Y6_GPIO_NUM;

  config.pin_d5 =
    Y7_GPIO_NUM;

  config.pin_d6 =
    Y8_GPIO_NUM;

  config.pin_d7 =
    Y9_GPIO_NUM;


  config.pin_xclk =
    XCLK_GPIO_NUM;

  config.pin_pclk =
    PCLK_GPIO_NUM;

  config.pin_vsync =
    VSYNC_GPIO_NUM;

  config.pin_href =
    HREF_GPIO_NUM;


  config.pin_sccb_sda =
    SIOD_GPIO_NUM;

  config.pin_sccb_scl =
    SIOC_GPIO_NUM;


  config.pin_pwdn =
    PWDN_GPIO_NUM;

  config.pin_reset =
    RESET_GPIO_NUM;


  config.xclk_freq_hz =
    20000000;


  // IMPORTANT:
  // Keep RGB565 because this is the format
  // used by our working ESP-DL pipeline.
  config.pixel_format =
    PIXFORMAT_RGB565;


  config.frame_size =
    FRAMESIZE_QVGA;


  config.jpeg_quality =
    12;


  // IMPORTANT:
  // Two frame buffers + LATEST mode.
  //
  // This prevents us from being stuck processing
  // an old frame when authentication is triggered.
  config.fb_count =
    2;

  config.fb_location =
    CAMERA_FB_IN_PSRAM;

  config.grab_mode =
    CAMERA_GRAB_LATEST;


  esp_err_t err =
    esp_camera_init(
      &config
    );


  if (
    err != ESP_OK
  ) {

    Serial.printf(
      "Camera init failed: 0x%x\n",
      err
    );

    return false;
  }


  cameraInitialized = true;

  Serial.println(
    "Camera initialized successfully."
  );


  sensor_t *sensor =
    esp_camera_sensor_get();


  if (sensor) {

    Serial.printf(
      "Camera sensor PID: 0x%04X\n",
      sensor->id.PID
    );
  }


  return true;
}


// ============================================================
// STOP CAMERA WHILE DOOR IS UNLOCKED
// ============================================================

void stopCamera()
{
  if (!cameraInitialized) {
    return;
  }

  Serial.println("Camera: stopping capture while lock is unlocked...");
  esp_err_t err = esp_camera_deinit();
  if (err == ESP_OK) {
    cameraInitialized = false;
    Serial.println("Camera: capture stopped.");
  } else {
    Serial.printf("Camera: deinit failed: 0x%x\n", err);
  }
}


// ============================================================
// FLUSH STALE FRAMES
// ============================================================

void flushCameraFrames()
{
  Serial.println(
    "Flushing stale camera frames..."
  );


  // With CAMERA_GRAB_LATEST and fb_count=2,
  // discard the currently queued frames.
  for (
    int i = 0;
    i < 2;
    i++
  ) {

    camera_fb_t *fb =
      esp_camera_fb_get();


    if (fb) {

      Serial.printf(
        "Discarded frame %d: "
        "%dx%d len=%u\n",
        i + 1,
        fb->width,
        fb->height,
        (unsigned)fb->len
      );


      esp_camera_fb_return(
        fb
      );
    }


    delay(20);
  }


  // Give the sensor a little time to
  // produce another frame after flushing.
  delay(50);


  Serial.println(
    "Camera ready for fresh frame."
  );
}


// ============================================================
// DETECT FACE
// ============================================================

static bool validateLandmarks(
  const result_t &face,
  int imageWidth,
  int imageHeight
)
{
  if (face.keypoint.size() < 10) {
    return false;
  }

  // MNP01 order used by ESP-DL examples:
  // 0-1  = left eye
  // 2-3  = left mouth corner
  // 4-5  = nose
  // 6-7  = right eye
  // 8-9  = right mouth corner
  for (int i = 0; i < 10; i += 2) {
    const int x = face.keypoint[i];
    const int y = face.keypoint[i + 1];

    if (x < -2 || x > imageWidth + 1 ||
        y < -2 || y > imageHeight + 1) {
      return false;
    }
  }

  const float lex = (float)face.keypoint[0];
  const float ley = (float)face.keypoint[1];
  const float lmx = (float)face.keypoint[2];
  const float lmy = (float)face.keypoint[3];
  const float nx  = (float)face.keypoint[4];
  const float ny  = (float)face.keypoint[5];
  const float rex = (float)face.keypoint[6];
  const float rey = (float)face.keypoint[7];
  const float rmx = (float)face.keypoint[8];
  const float rmy = (float)face.keypoint[9];

  const float eyeDx = rex - lex;
  const float eyeDy = rey - ley;
  const float eyeDist = sqrtf(eyeDx * eyeDx + eyeDy * eyeDy);

  const float mouthDx = rmx - lmx;
  const float mouthDy = rmy - lmy;
  const float mouthDist = sqrtf(mouthDx * mouthDx + mouthDy * mouthDy);

  if (eyeDist < 12.0f || mouthDist < 8.0f) {
    return false;
  }

  // Require the eyes to be ordered left-to-right in the image.
  if (rex <= lex) {
    return false;
  }

  // Sanity check the vertical geometry. These are deliberately loose
  // because the whole point is to support mild head tilt.
  const float eyeMidY = 0.5f * (ley + rey);
  if (ny < eyeMidY - 0.45f * eyeDist ||
      ny > eyeMidY + 1.35f * eyeDist) {
    return false;
  }

  return true;
}

bool detectFace(
  camera_fb_t *fb,
  std::vector<int> &landmarks,
  result_t *bestResult = nullptr,
  bool *hasValidLandmarks = nullptr
)
{
  if (!fb) {
    return false;
  }

  if (hasValidLandmarks) {
    *hasValidLandmarks = false;
  }

  landmarks.clear();

  std::vector<int> shape = {
    (int)fb->height,
    (int)fb->width,
    3
  };

  auto &candidates =
    detector1.infer(
      (uint16_t *)fb->buf,
      shape
    );

  if (candidates.empty()) {
    Serial.println("MSR01: no face.");
    return false;
  }

  Serial.printf(
    "MSR01: %d candidate(s)\n",
    (int)candidates.size()
  );

  // MNP01 is mandatory in this version. There is deliberately NO
  // bounding-box fallback: every accepted face must provide valid
  // five-point landmarks so registration and authentication use the
  // exact same geometric normalization.
  auto &refined =
    detector2.infer(
      (uint16_t *)fb->buf,
      shape,
      candidates
    );

  if (refined.empty()) {
    Serial.println(
      "MNP01: no refined face; retrying frame (5PT required)."
    );
    return false;
  }

  Serial.printf(
    "MNP01: %d refined candidate(s)\n",
    (int)refined.size()
  );

  auto refinedBest = refined.begin();
  for (auto it = refined.begin(); it != refined.end(); ++it) {
    if (it->score > refinedBest->score) {
      refinedBest = it;
    }
  }

  if (!validateLandmarks(
        *refinedBest,
        (int)fb->width,
        (int)fb->height
      )) {
    Serial.println(
      "MNP01: landmarks failed geometry validation; retrying frame (5PT required)."
    );
    return false;
  }

  result_t selected = *refinedBest;

  Serial.printf(
    "LANDMARKS VALID | LE=(%d,%d) LM=(%d,%d) N=(%d,%d) RE=(%d,%d) RM=(%d,%d)\n",
    selected.keypoint[0], selected.keypoint[1],
    selected.keypoint[2], selected.keypoint[3],
    selected.keypoint[4], selected.keypoint[5],
    selected.keypoint[6], selected.keypoint[7],
    selected.keypoint[8], selected.keypoint[9]
  );

  Serial.printf(
    "FACE DETECTED | score=%.3f | box=(%d,%d)-(%d,%d) | alignment=5PT\n",
    selected.score,
    selected.box[0],
    selected.box[1],
    selected.box[2],
    selected.box[3]
  );

  for (int i = 0; i < 10; ++i) {
    landmarks.push_back(selected.keypoint[i]);
  }

  if (bestResult) {
    *bestResult = selected;
  }

  if (hasValidLandmarks) {
    *hasValidLandmarks = true;
  }

  return true;
}


// ============================================================
// CAPTURE FRESH EMBEDDING
// ============================================================

// Reusable PSRAM buffer for the ESP-DL recognition input.
// The camera produces RGB565, while the ESP-DL uint16_t recognition
// overload explicitly expects BGR565.
static uint16_t *bgr565Buffer = nullptr;
static size_t bgr565BufferPixels = 0;

static bool ensureBGR565Buffer(size_t pixelCount)
{
  if (bgr565Buffer != nullptr && bgr565BufferPixels >= pixelCount) {
    return true;
  }

  if (bgr565Buffer != nullptr) {
    free(bgr565Buffer);
    bgr565Buffer = nullptr;
    bgr565BufferPixels = 0;
  }

  bgr565Buffer = (uint16_t *)ps_malloc(pixelCount * sizeof(uint16_t));
  if (bgr565Buffer == nullptr) {
    Serial.printf(
      "ERROR: Failed to allocate BGR565 buffer for %u pixels.\n",
      (unsigned)pixelCount
    );
    return false;
  }

  bgr565BufferPixels = pixelCount;
  Serial.printf(
    "BGR565 buffer allocated in PSRAM: %u bytes\n",
    (unsigned)(pixelCount * sizeof(uint16_t))
  );
  return true;
}

static bool convertRGB565ToBGR565(
  camera_fb_t *fb,
  uint16_t *dst,
  uint32_t *checksum
)
{
  if (!fb || !dst) {
    return false;
  }

  const size_t pixelCount =
    (size_t)fb->width * (size_t)fb->height;

  const uint16_t *src = (const uint16_t *)fb->buf;

  uint32_t hash = 2166136261UL;

  for (size_t i = 0; i < pixelCount; ++i) {
    const uint16_t pixel = src[i];

    const uint16_t r = (pixel >> 11) & 0x1F;
    const uint16_t g = (pixel >> 5)  & 0x3F;
    const uint16_t b =  pixel        & 0x1F;

    // RGB565 -> BGR565: swap the 5-bit red and blue fields.
    const uint16_t bgr565 =
      (uint16_t)((b << 11) | (g << 5) | r);

    dst[i] = bgr565;

    // Lightweight checksum over the converted recognition buffer.
    if ((i & 0x60U) == 0U) {
      hash ^= (uint8_t)(bgr565 & 0xFF);
      hash *= 16777619UL;
      hash ^= (uint8_t)(bgr565 >> 8);
      hash *= 16777619UL;
    }
  }

  if (checksum) {
    *checksum = hash;
  }

  return true;
}

// ============================================================
// BUILD 112x112 BGR888 FACE TENSOR FROM DETECTED FACE BOX
//
// The ESP-DL aligned-face overload on this installed ESP-DL build is
// leaving the caller-provided Tensor unchanged (the checksum stayed
// identical even when source frames changed). To avoid feeding a
// constant/zero tensor to the feature model, build the exact 112x112x3
// BGR888 tensor ourselves from the current camera frame.
//
// The tensor is the size expected by FaceRecognition112V1S8, so the
// previous crash seen with the full 320x240 BGR888 path is avoided.
// ============================================================

static uint8_t rgb565ToR(uint16_t p)
{
  uint8_t r5 = (uint8_t)((p >> 11) & 0x1F);
  return (uint8_t)((r5 * 255U + 15U) / 31U);
}

static uint8_t rgb565ToG(uint16_t p)
{
  uint8_t g6 = (uint8_t)((p >> 5) & 0x3F);
  return (uint8_t)((g6 * 255U + 31U) / 63U);
}

static uint8_t rgb565ToB(uint16_t p)
{
  uint8_t b5 = (uint8_t)(p & 0x1F);
  return (uint8_t)((b5 * 255U + 15U) / 31U);
}

static uint8_t sampleChannelBilinear(
  const uint16_t *image,
  int width,
  int height,
  float fx,
  float fy,
  int channel
)
{
  // Clamp to the valid camera image.
  if (fx < 0.0f) fx = 0.0f;
  if (fy < 0.0f) fy = 0.0f;
  if (fx > (float)(width - 1))  fx = (float)(width - 1);
  if (fy > (float)(height - 1)) fy = (float)(height - 1);

  int x0 = (int)fx;
  int y0 = (int)fy;
  int x1 = (x0 + 1 < width) ? x0 + 1 : x0;
  int y1 = (y0 + 1 < height) ? y0 + 1 : y0;

  float ax = fx - (float)x0;
  float ay = fy - (float)y0;

  const uint16_t p00 = image[y0 * width + x0];
  const uint16_t p10 = image[y0 * width + x1];
  const uint16_t p01 = image[y1 * width + x0];
  const uint16_t p11 = image[y1 * width + x1];

  uint8_t c00, c10, c01, c11;

  if (channel == 0) { // B
    c00 = rgb565ToB(p00); c10 = rgb565ToB(p10);
    c01 = rgb565ToB(p01); c11 = rgb565ToB(p11);
  }
  else if (channel == 1) { // G
    c00 = rgb565ToG(p00); c10 = rgb565ToG(p10);
    c01 = rgb565ToG(p01); c11 = rgb565ToG(p11);
  }
  else { // R
    c00 = rgb565ToR(p00); c10 = rgb565ToR(p10);
    c01 = rgb565ToR(p01); c11 = rgb565ToR(p11);
  }

  float top = (float)c00 + ((float)c10 - (float)c00) * ax;
  float bot = (float)c01 + ((float)c11 - (float)c01) * ax;
  float val = top + (bot - top) * ay;

  if (val < 0.0f) val = 0.0f;
  if (val > 255.0f) val = 255.0f;

  return (uint8_t)(val + 0.5f);
}

// ArcFace / InsightFace canonical 5-point geometry for a 112x112 input.
// Source order from ESP-DL MNP01 is:
//   left eye, left mouth, nose, right eye, right mouth
// Reordered below as:
//   left eye, right eye, nose, left mouth, right mouth
static const float kArcFaceDst[5][2] = {
  {38.2946f, 51.6963f},
  {73.5318f, 51.5014f},
  {56.0252f, 71.7366f},
  {41.5493f, 92.3655f},
  {70.7299f, 92.2041f}
};

static bool allocateAlignedTensor(Tensor<uint8_t> &alignedFace)
{
  alignedFace.set_shape({112, 112, 3});

  if (alignedFace.element == nullptr) {
    if (!alignedFace.calloc_element()) {
      Serial.println(
        "ERROR: Failed to allocate 112x112 BGR888 tensor."
      );
      return false;
    }
  } else {
    memset(
      alignedFace.element,
      0,
      (size_t)alignedFace.get_size()
    );
  }

  return true;
}

// Estimate a 2D similarity transform (scale + rotation + translation)
// mapping 5 source landmarks to the fixed ArcFace landmark template.
// This is the same geometric normalization family used by InsightFace:
// it removes changes in face scale, in-plane rotation and translation
// without introducing the non-uniform stretching caused by resizing a box.
static bool estimateSimilarityTransform5Point(
  const float src[5][2],
  float matrix[2][3]
)
{
  float srcMeanX = 0.0f;
  float srcMeanY = 0.0f;
  float dstMeanX = 0.0f;
  float dstMeanY = 0.0f;

  for (int i = 0; i < 5; ++i) {
    srcMeanX += src[i][0];
    srcMeanY += src[i][1];
    dstMeanX += kArcFaceDst[i][0];
    dstMeanY += kArcFaceDst[i][1];
  }

  srcMeanX /= 5.0f;
  srcMeanY /= 5.0f;
  dstMeanX /= 5.0f;
  dstMeanY /= 5.0f;

  float a = 0.0f;
  float b = 0.0f;
  float srcEnergy = 0.0f;

  for (int i = 0; i < 5; ++i) {
    const float sx = src[i][0] - srcMeanX;
    const float sy = src[i][1] - srcMeanY;
    const float dx = kArcFaceDst[i][0] - dstMeanX;
    const float dy = kArcFaceDst[i][1] - dstMeanY;

    a += sx * dx + sy * dy;
    b += sx * dy - sy * dx;
    srcEnergy += sx * sx + sy * sy;
  }

  const float rotationNorm = sqrtf(a * a + b * b);
  if (srcEnergy < 1.0e-6f || rotationNorm < 1.0e-6f) {
    return false;
  }

  const float scale = rotationNorm / srcEnergy;
  const float c = a / rotationNorm;
  const float sn = b / rotationNorm;

  matrix[0][0] = scale * c;
  matrix[0][1] = -scale * sn;
  matrix[1][0] = scale * sn;
  matrix[1][1] = scale * c;

  matrix[0][2] =
    dstMeanX -
    (matrix[0][0] * srcMeanX + matrix[0][1] * srcMeanY);

  matrix[1][2] =
    dstMeanY -
    (matrix[1][0] * srcMeanX + matrix[1][1] * srcMeanY);

  return true;
}

static bool buildAlignedBGR888FromLandmarks(
  camera_fb_t *fb,
  const result_t &face,
  Tensor<uint8_t> &alignedFace,
  uint32_t *checksum
)
{
  if (!fb || !fb->buf || face.keypoint.size() < 10) {
    return false;
  }

  const float src[5][2] = {
    // MNP01: left eye
    {(float)face.keypoint[0], (float)face.keypoint[1]},
    // MNP01: right eye
    {(float)face.keypoint[6], (float)face.keypoint[7]},
    // MNP01: nose
    {(float)face.keypoint[4], (float)face.keypoint[5]},
    // MNP01: left mouth
    {(float)face.keypoint[2], (float)face.keypoint[3]},
    // MNP01: right mouth
    {(float)face.keypoint[8], (float)face.keypoint[9]}
  };

  float matrix[2][3];
  if (!estimateSimilarityTransform5Point(src, matrix)) {
    return false;
  }

  if (!allocateAlignedTensor(alignedFace)) {
    return false;
  }

  const uint16_t *image = (const uint16_t *)fb->buf;
  const int imageWidth = (int)fb->width;
  const int imageHeight = (int)fb->height;

  // Invert the similarity transform so each output pixel samples from
  // the correct location in the original RGB565 camera frame.
  const float det =
    matrix[0][0] * matrix[1][1] -
    matrix[0][1] * matrix[1][0];

  if (fabsf(det) < 1.0e-8f) {
    return false;
  }

  const float inv00 =  matrix[1][1] / det;
  const float inv01 = -matrix[0][1] / det;
  const float inv10 = -matrix[1][0] / det;
  const float inv11 =  matrix[0][0] / det;

  const float itx =
    -(inv00 * matrix[0][2] + inv01 * matrix[1][2]);
  const float ity =
    -(inv10 * matrix[0][2] + inv11 * matrix[1][2]);

  uint32_t hash = 2166136261UL;

  for (int y = 0; y < 112; ++y) {
    for (int x = 0; x < 112; ++x) {
      // Destination -> source mapping.
      const float sx = inv00 * (float)x + inv01 * (float)y + itx;
      const float sy = inv10 * (float)x + inv11 * (float)y + ity;

      const uint8_t b = sampleChannelBilinear(
        image, imageWidth, imageHeight, sx, sy, 0
      );
      const uint8_t g = sampleChannelBilinear(
        image, imageWidth, imageHeight, sx, sy, 1
      );
      const uint8_t r = sampleChannelBilinear(
        image, imageWidth, imageHeight, sx, sy, 2
      );

      const size_t idx =
        ((size_t)y * 112U + (size_t)x) * 3U;

      alignedFace.element[idx + 0] = b;
      alignedFace.element[idx + 1] = g;
      alignedFace.element[idx + 2] = r;

      if (((x + y * 112) & 0x3F) == 0) {
        hash ^= b; hash *= 16777619UL;
        hash ^= g; hash *= 16777619UL;
        hash ^= r; hash *= 16777619UL;
      }
    }
  }

  if (checksum) {
    *checksum = hash;
  }

  Serial.printf(
    "ALIGNMENT: ArcFace 5-point similarity transform | matrix=[%.5f %.5f %.5f; %.5f %.5f %.5f]\n",
    matrix[0][0], matrix[0][1], matrix[0][2],
    matrix[1][0], matrix[1][1], matrix[1][2]
  );

  return true;
}

static bool buildAlignedBGR888FromBox(
  camera_fb_t *fb,
  const result_t &face,
  Tensor<uint8_t> &alignedFace,
  uint32_t *checksum
)
{
  if (!fb || fb->buf == nullptr) {
    return false;
  }

  const uint16_t *image = (const uint16_t *)fb->buf;
  const int imageWidth = (int)fb->width;
  const int imageHeight = (int)fb->height;

  int left = face.box[0];
  int top = face.box[1];
  int right = face.box[2];
  int bottom = face.box[3];

  if (left < 0) left = 0;
  if (top < 0) top = 0;
  if (right >= imageWidth) right = imageWidth - 1;
  if (bottom >= imageHeight) bottom = imageHeight - 1;

  if (right <= left || bottom <= top) {
    return false;
  }

  float boxW = (float)(right - left + 1);
  float boxH = (float)(bottom - top + 1);
  float side = (boxW > boxH ? boxW : boxH) * 1.30f;

  float cx = ((float)left + (float)right) * 0.5f;
  float cy = ((float)top + (float)bottom) * 0.5f;
  float cropLeft = cx - side * 0.5f;
  float cropTop  = cy - side * 0.5f;

  if (!allocateAlignedTensor(alignedFace)) {
    return false;
  }

  uint32_t hash = 2166136261UL;

  for (int y = 0; y < 112; ++y) {
    float sy = cropTop + ((float)y + 0.5f) * side / 112.0f - 0.5f;

    for (int x = 0; x < 112; ++x) {
      float sx = cropLeft + ((float)x + 0.5f) * side / 112.0f - 0.5f;

      const uint8_t b = sampleChannelBilinear(
        image, imageWidth, imageHeight, sx, sy, 0
      );
      const uint8_t g = sampleChannelBilinear(
        image, imageWidth, imageHeight, sx, sy, 1
      );
      const uint8_t r = sampleChannelBilinear(
        image, imageWidth, imageHeight, sx, sy, 2
      );

      size_t idx = ((size_t)y * 112U + (size_t)x) * 3U;
      alignedFace.element[idx + 0] = b;
      alignedFace.element[idx + 1] = g;
      alignedFace.element[idx + 2] = r;

      if (((x + y * 112) & 0x3F) == 0) {
        hash ^= b; hash *= 16777619UL;
        hash ^= g; hash *= 16777619UL;
        hash ^= r; hash *= 16777619UL;
      }
    }
  }

  if (checksum) {
    *checksum = hash;
  }

  return true;
}

// ============================================================
// GENERATE EMBEDDING FROM A CAMERA FRAME
// ============================================================

bool generateEmbeddingFromFrame(
  camera_fb_t *fb,
  const result_t &detectedFace,
  float *outputEmbedding
)
{
  if (!fb || !outputEmbedding) {
    return false;
  }

  recognizer.set_thresh(0.55F);

  Tensor<uint8_t> alignedFace;
  uint32_t alignedChecksum = 0;

  if (!validateLandmarks(
        detectedFace,
        (int)fb->width,
        (int)fb->height
      )) {
    Serial.println(
      "ERROR: No valid 5-point landmarks; refusing non-5PT embedding."
    );
    return false;
  }

  Serial.println(
    "RECOGNITION PIPELINE: MNP01 5 landmarks -> ArcFace similarity alignment -> 112x112 BGR888"
  );

  if (!buildAlignedBGR888FromLandmarks(
        fb,
        detectedFace,
        alignedFace,
        &alignedChecksum
      )) {
    Serial.println(
      "ERROR: 5-point alignment failed; refusing frame."
    );
    return false;
  }

  Serial.printf(
    "ALIGNED BGR888 CHECKSUM: 0x%08lX | mode=5PT\n",
    (unsigned long)alignedChecksum
  );

  if (alignedFace.element == nullptr ||
      alignedFace.get_size() != (112 * 112 * 3)) {
    Serial.printf(
      "ERROR: Invalid aligned tensor size: %d\n",
      (int)alignedFace.get_size()
    );
    return false;
  }

  uint8_t minPixel = 255;
  uint8_t maxPixel = 0;
  uint32_t pixelSum = 0;

  for (size_t i = 0; i < (size_t)alignedFace.get_size(); i += 31) {
    uint8_t v = alignedFace.element[i];
    if (v < minPixel) minPixel = v;
    if (v > maxPixel) maxPixel = v;
    pixelSum += v;
  }

  Serial.printf(
    "ALIGNED BGR888 STATS: min=%u max=%u sampleSum=%lu\n",
    minPixel,
    maxPixel,
    (unsigned long)pixelSum
  );

  face_info_t info = recognizer.recognize(
    alignedFace
  );
  (void)info;

  Tensor<float> &embedding =
    recognizer.get_face_emb(-1);

  if (embedding.element == nullptr ||
      embedding.get_size() != EMBEDDING_SIZE) {
    Serial.printf(
      "ERROR: Wrong/empty embedding size: %d\n",
      (int)embedding.get_size()
    );
    return false;
  }

  memcpy(
    outputEmbedding,
    embedding.element,
    EMBEDDING_SIZE * sizeof(float)
  );

  double embNorm = 0.0;
  double embSum = 0.0;
  for (int i = 0; i < EMBEDDING_SIZE; ++i) {
    embSum += outputEmbedding[i];
    embNorm += (double)outputEmbedding[i] * outputEmbedding[i];
  }

  Serial.printf(
    "EMBEDDING FIRST VALUES: %.7f %.7f\n",
    outputEmbedding[0],
    outputEmbedding[1]
  );
  Serial.printf(
    "EMBEDDING STATS: norm=%.7f sum=%.7f\n",
    sqrt(embNorm),
    embSum
  );

  return true;
}

bool captureEmbedding(
  float *outputEmbedding,
  camera_fb_t **capturedFrame = nullptr,
  result_t *capturedFace = nullptr
)
{
  if (!outputEmbedding) {
    return false;
  }

  // Use the same global recognizer instance used by the original implementation.

  // ----------------------------------------------------------
  // CRITICAL FIX:
  //
  // Throw away frames that were already sitting in the
  // camera buffer before this authentication/registration
  // operation began.
  // ----------------------------------------------------------

  flushCameraFrames();


  camera_fb_t *fb =
    esp_camera_fb_get();


  if (!fb) {

    Serial.println(
      "ERROR: Fresh camera capture failed."
    );

    return false;
  }


  Serial.printf(
    "PROCESSING FRESH FRAME | "
    "ptr=%p | len=%u | time=%lu\n",
    fb->buf,
    (unsigned)fb->len,
    millis()
  );


  std::vector<int> landmarks;
  result_t detectedFace;


  bool faceFound =
    detectFace(
      fb,
      landmarks,
      &detectedFace
    );


  if (!faceFound) {

    Serial.println(
      "No face detected in fresh frame."
    );


    if (capturedFrame) {

      // Caller explicitly wants the frame,
      // even though face detection failed.
      *capturedFrame =
        fb;

    } else {

      esp_camera_fb_return(
        fb
      );
    }


    return false;
  }


  if (!generateEmbeddingFromFrame(
        fb,
        detectedFace,
        outputEmbedding
      )) {

    if (capturedFrame) {
      *capturedFrame = fb;
    } else {
      esp_camera_fb_return(fb);
    }

    return false;
  }


  Serial.println(
    "Embedding generated successfully."
  );

  if (capturedFace) {
    *capturedFace = detectedFace;
  }


  if (capturedFrame) {

    *capturedFrame =
      fb;

  } else {

    esp_camera_fb_return(
      fb
    );
  }


  return true;
}


// ============================================================
// OLED HELPERS
// ============================================================

void oledPrintCentered(const char *text, int y)
{
  if (!text) {
    return;
  }

  int maxChars = display.width() / 6;
  int len = strlen(text);

  if (len > maxChars) {
    len = maxChars;
  }

  int x = max(0, (display.width() - len * 6) / 2);

  display.setCursor(x, y);

  for (int i = 0; i < len; i++) {
    display.write(text[i]);
  }
}


void oledMessage(
  const char *line1,
  const char *line2,
  const char *line3
)
{
  // Always redraw the COMPLETE OLED buffer.
  // fillScreen() is used instead of relying only on clearDisplay()
  // so no pixels from the previous screen can remain.
  display.fillScreen(SH110X_BLACK);

  display.setTextColor(SH110X_WHITE);
  display.setTextWrap(false);
  display.setTextSize(1);

  const int lineHeight = 12;
  int lines = 0;

  if (line1) lines++;
  if (line2) lines++;
  if (line3) lines++;

  int y = (display.height() - lines * lineHeight) / 2;

  if (y < 0) {
    y = 0;
  }

  const char *linesToDraw[3] = {
    line1,
    line2,
    line3
  };

  for (int i = 0; i < 3; i++) {
    if (!linesToDraw[i]) {
      continue;
    }

    const char *text = linesToDraw[i];

    int textWidth = strlen(text) * 6;
    int x = (display.width() - textWidth) / 2;

    if (x < 0) {
      x = 0;
    }

    display.setCursor(x, y);
    display.print(text);

    y += lineHeight;
  }

  // Push the complete framebuffer to the OLED.
  display.display();

  // Give the I2C/display driver a chance to finish.
  yield();
}

void oledReady()
{
  oledMessage("READY", "PRESS BUTTON");
}

void oledSearching()
{
  oledMessage("SEARCHING", "LOOK AT CAMERA");
}

void oledNoFace()
{
  oledMessage("NO FACE", "TRY AGAIN");
}

void oledAuthenticating()
{
  oledMessage("FACE FOUND", "AUTHENTICATING");
}

void oledAuthorized(const char *name)
{
  oledMessage("AUTHORIZED", name);
}

void oledUnknown()
{
  oledMessage("UNKNOWN", "ACCESS DENIED");
}

void oledRegistering(const char *name, int number = 0)
{
  char line2[24];

  if (number > 0) {
    snprintf(line2, sizeof(line2), "%s %d/%d", name, number, MAX_EMBEDDINGS_PER_USER);
    oledMessage("REGISTERING", line2);
  } else {
    oledMessage("REGISTERING", name);
  }
}

void oledRegisterSuccess(const char *name)
{
  oledMessage("REGISTERED", name);
}

void oledRegisterFailed()
{
  oledMessage("REGISTER FAILED", "TRY AGAIN");
}

void oledDeleteSuccess(const char *name = nullptr)
{
  if (name && name[0]) {
    oledMessage("USER DELETED", name);
  } else {
    oledMessage("USER DELETED");
  }
}

void oledDeleteFailed(const char *reason = nullptr)
{
  if (reason && reason[0]) {
    oledMessage("DELETE FAILED", reason);
  } else {
    oledMessage("DELETE FAILED");
  }
}

// ============================================================
// DISPLAY DETECTED FACE
// ============================================================

void displayFace(
  uint16_t *image,
  int imageWidth,
  int imageHeight,
  int left,
  int top,
  int right,
  int bottom
)
{
  if (!image || imageWidth <= 0 || imageHeight <= 0) {
    return;
  }

  // Add padding around the detector box so the displayed image
  // contains some of the face/head instead of being tightly cropped.
  int boxW = right - left;
  int boxH = bottom - top;

  if (boxW <= 2 || boxH <= 2) {
    Serial.println("OLED FACE PREVIEW: invalid face box.");
    return;
  }

  int padX = max(8, boxW / 6);
  int padY = max(8, boxH / 6);

  left   = constrain(left  - padX, 0, imageWidth  - 1);
  top    = constrain(top   - padY, 0, imageHeight - 1);
  right  = constrain(right + padX, 0, imageWidth  - 1);
  bottom = constrain(bottom + padY, 0, imageHeight - 1);

  int cropWidth  = right - left + 1;
  int cropHeight = bottom - top + 1;

  if (cropWidth <= 1 || cropHeight <= 1) {
    return;
  }

  // Explicitly clear the complete OLED framebuffer first.
  display.fillScreen(SH110X_BLACK);

  const int oledWidth  = display.width();
  const int oledHeight = display.height();

  float scaleX = (float)oledWidth  / cropWidth;
  float scaleY = (float)oledHeight / cropHeight;
  float scale  = min(scaleX, scaleY);

  int drawWidth  = max(1, (int)(cropWidth  * scale));
  int drawHeight = max(1, (int)(cropHeight * scale));

  int offsetX = (oledWidth  - drawWidth)  / 2;
  int offsetY = (oledHeight - drawHeight) / 2;

  // Use a lower threshold than before. A threshold of 110 could
  // make a normally exposed face appear almost completely blank.
  const int GRAY_THRESHOLD = 75;

  for (int y = 0; y < drawHeight; y++) {
    int srcY = top + ((long)y * cropHeight) / drawHeight;
    srcY = constrain(srcY, 0, imageHeight - 1);

    for (int x = 0; x < drawWidth; x++) {
      int srcX = left + ((long)x * cropWidth) / drawWidth;
      srcX = constrain(srcX, 0, imageWidth - 1);

      uint16_t pixel = image[srcY * imageWidth + srcX];

      uint8_t r = ((pixel >> 11) & 0x1F) << 3;
      uint8_t g = ((pixel >> 5)  & 0x3F) << 2;
      uint8_t b = (pixel & 0x1F) << 3;

      uint8_t gray =
        (uint16_t)(r * 30 + g * 59 + b * 11) / 100;

      if (gray >= GRAY_THRESHOLD) {
        display.drawPixel(
          offsetX + x,
          offsetY + y,
          SH110X_WHITE
        );
      }
    }
  }

  display.display();
  yield();

  Serial.printf(
    "OLED FACE PREVIEW: box=(%d,%d)-(%d,%d), crop=%dx%d, display=%dx%d\n",
    left, top, right, bottom,
    cropWidth, cropHeight,
    drawWidth, drawHeight
  );
}

// ============================================================
// AUTHENTICATION SEARCH - 10 SECOND WINDOW
// ============================================================

bool captureAuthenticationEmbedding(
  float *outputEmbedding,
  camera_fb_t **capturedFrame,
  result_t *capturedFace
)
{
  if (!outputEmbedding || !capturedFrame || !capturedFace) {
    return false;
  }

  *capturedFrame = nullptr;

  // Clear frames that existed before the button press.
  flushCameraFrames();

  const unsigned long SEARCH_TIMEOUT_MS = 10000;
  const unsigned long start = millis();

  oledSearching();

  while (millis() - start < SEARCH_TIMEOUT_MS) {
    camera_fb_t *fb = esp_camera_fb_get();

    if (!fb) {
      delay(30);
      continue;
    }

    std::vector<int> landmarks;
    result_t face;

    bool found = detectFace(
      fb,
      landmarks,
      &face
    );

    if (found) {
      // Show the exact face frame that will be authenticated.
      displayFace(
        (uint16_t *)fb->buf,
        fb->width,
        fb->height,
        face.box[0],
        face.box[1],
        face.box[2],
        face.box[3]
      );

      // Keep the exact detected face frame visible for at least
      // 3 seconds. Do not draw another message over it during
      // this interval.
      const unsigned long previewStart = millis();

      while (millis() - previewStart < 3000UL) {
        delay(10);
        yield();
      }

      // Clear the preview before switching to text.
      display.fillScreen(SH110X_BLACK);
      display.display();
      yield();

      *capturedFrame = fb;
      *capturedFace = face;

      oledAuthenticating();

      if (!generateEmbeddingFromFrame(
            fb,
            face,
            outputEmbedding
          )) {
        return false;
      }

      double embSum = 0.0;
      double embSq = 0.0;
      for (int i = 0; i < EMBEDDING_SIZE; i++) {
        embSum += outputEmbedding[i];
        embSq += (double)outputEmbedding[i] * outputEmbedding[i];
      }
      Serial.printf(
        "AUTH EMBEDDING FINGERPRINT | first=%.7f second=%.7f norm=%.7f sum=%.7f\n",
        outputEmbedding[0],
        outputEmbedding[1],
        sqrt(embSq),
        embSum
      );

      Serial.println("Authentication embedding generated.");
      return true;
    }

    esp_camera_fb_return(fb);
    delay(30);
  }

  Serial.println("10-second face search expired: no face found.");
  oledNoFace();
  return false;
}


// ============================================================
// COSINE SIMILARITY
// ============================================================

float cosineSimilarity(
  const float *a,
  const float *b
)
{
  double dot = 0.0;

  double normA = 0.0;

  double normB = 0.0;


  for (
    int i = 0;
    i < EMBEDDING_SIZE;
    i++
  ) {

    dot +=
      (double)a[i] *
      (double)b[i];

    normA +=
      (double)a[i] *
      (double)a[i];

    normB +=
      (double)b[i] *
      (double)b[i];
  }


  if (
    normA <= 0.0 ||
    normB <= 0.0
  ) {

    return -1.0F;
  }


  return (float)(
    dot /
    (
      sqrt(normA) *
      sqrt(normB)
    )
  );
}


// ============================================================
// REGISTRATION EVENT
// ============================================================

void publishRegistrationProgress(
  int32_t id,
  const char *name,
  const char *status,
  int embeddingNumber = 0
)
{
  StaticJsonDocument<256> doc;


  doc["id"] =
    id;

  doc["name"] =
    name;

  doc["status"] =
    status;


  if (
    embeddingNumber > 0
  ) {

    doc["embedding"] =
      embeddingNumber;

    doc["total"] =
      MAX_EMBEDDINGS_PER_USER;
  }


  publishJSON(
    TOPIC_REGISTRATION_EVENT,
    doc
  );
}


// ============================================================
// PERFORM REGISTRATION
// ============================================================

void performRegistration(
  int32_t id,
  const char *name
)
{
  deviceBusy = true;
  oledRegistering(name);


  Serial.println();
  Serial.println(
    "========================================"
  );

  Serial.println(
    "STARTING FACE REGISTRATION"
  );

  Serial.println(
    "========================================"
  );


  Serial.printf(
    "ID   : %ld\n",
    (long)id
  );

  Serial.printf(
    "NAME : %s\n",
    name
  );


  // ----------------------------------------------------------
  // Find existing ID.
  // ----------------------------------------------------------

  int existingIndex =
    findUserIndex(id);


  if (
    existingIndex >= 0
  ) {

    FaceUser &existing =
      database->users[
        existingIndex
      ];


    // Existing populated user.
    if (
      existing.embeddingCount > 0
    ) {

      StaticJsonDocument<256> doc;

      doc["id"] =
        id;

      doc["name"] =
        name;

      doc["status"] =
        "failed";

      doc["reason"] =
        "id_already_registered";


      publishJSON(
        TOPIC_REGISTRATION_EVENT,
        doc
      );

      oledRegisterFailed();

      deviceBusy = false;

      return;
    }


  } else {

    // --------------------------------------------------------
    // New user.
    // --------------------------------------------------------

    if (
      database->userCount >=
      MAX_USERS
    ) {

      StaticJsonDocument<256> doc;

      doc["id"] =
        id;

      doc["name"] =
        name;

      doc["status"] =
        "failed";

      doc["reason"] =
        "database_full";


      publishJSON(
        TOPIC_REGISTRATION_EVENT,
        doc
      );

      oledRegisterFailed();

      deviceBusy = false;

      return;
    }


    existingIndex =
      database->userCount++;


    database->users[
      existingIndex
    ].id = id;


    memset(
      database->users[
        existingIndex
      ].name,
      0,
      MAX_NAME_LENGTH
    );


    strncpy(
      database->users[
        existingIndex
      ].name,
      name,
      MAX_NAME_LENGTH - 1
    );


    database->users[
      existingIndex
    ].embeddingCount = 0;
  }


  FaceUser &user =
    database->users[
      existingIndex
    ];


  // Update name in case this is an
  // empty placeholder record.
  memset(
    user.name,
    0,
    MAX_NAME_LENGTH
  );


  strncpy(
    user.name,
    name,
    MAX_NAME_LENGTH - 1
  );


  publishRegistrationProgress(
    id,
    name,
    "capturing"
  );


  // ----------------------------------------------------------
  // Capture 10 embeddings.
  // ----------------------------------------------------------

  for (
    int n = 0;
    n < MAX_EMBEDDINGS_PER_USER;
    n++
  ) {

    bool captured =
      false;


    Serial.println();
    Serial.printf(
      "Waiting for embedding %d/%d...\n",
      n + 1,
      MAX_EMBEDDINGS_PER_USER
    );


    unsigned long start =
      millis();


    // 15 second timeout per embedding.
    while (
      !captured &&
      millis() - start < 15000
    ) {

      // Reuse the PSRAM authentication buffer for registration.
      // Registration and authentication are mutually exclusive, so this
      // buffer can safely be shared and avoids another large stack object.
      float *embedding = authEmbeddingPSRAM;

      camera_fb_t *registrationFrame = nullptr;
      result_t registrationFace;

      // Tell the user which sample is currently being captured.
      oledRegistering(name, n + 1);

      if (
        captureEmbedding(
          embedding,
          &registrationFrame,
          &registrationFace
        )
      ) {

        // -----------------------------------------------
        // Face was found and embedded. Show the exact
        // face crop that is being added to the database.
        // -----------------------------------------------
        if (registrationFrame) {
          displayFace(
            (uint16_t *)registrationFrame->buf,
            registrationFrame->width,
            registrationFrame->height,
            registrationFace.box[0],
            registrationFace.box[1],
            registrationFace.box[2],
            registrationFace.box[3]
          );

          Serial.printf(
            "REGISTRATION PREVIEW | embedding %d/%d | "
            "face score=%.3f\n",
            n + 1,
            MAX_EMBEDDINGS_PER_USER,
            registrationFace.score
          );

          delay(700);
        }

        memcpy(
          user.embeddings[n],
          embedding,
          EMBEDDING_SIZE *
          sizeof(float)
        );

        double regSum = 0.0;
        double regSq = 0.0;
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
          regSum += embedding[i];
          regSq += (double)embedding[i] * embedding[i];
        }
        Serial.printf(
          "REG EMBEDDING FINGERPRINT | sample=%d first=%.7f second=%.7f norm=%.7f sum=%.7f\n",
          n + 1, embedding[0], embedding[1], sqrt(regSq), regSum
        );

        user.embeddingCount =
          n + 1;

        captured = true;

        Serial.printf(
          "Embedding %d/%d captured and added for ID=%ld Name=%s.\n",
          n + 1,
          MAX_EMBEDDINGS_PER_USER,
          (long)id,
          name
        );

        publishRegistrationProgress(
          id,
          name,
          "capturing",
          n + 1
        );

        // Return the frame after the preview is shown.
        if (registrationFrame) {
          esp_camera_fb_return(registrationFrame);
          registrationFrame = nullptr;
        }

        oledRegistering(name, n + 1);

        // Allow the person to move slightly
        // before the next sample.
        delay(500);

      } else {

        // No face in this frame. captureEmbedding() may
        // still return the frame when capturedFrame is used.
        if (registrationFrame) {
          esp_camera_fb_return(registrationFrame);
          registrationFrame = nullptr;
        }

        // Keep the display in registration/search state.
        oledRegistering(name, n + 1);

        delay(100);
      }
    }


    // --------------------------------------------------------
    // Failed to capture this embedding.
    // --------------------------------------------------------

    if (!captured) {

      Serial.printf(
        "Registration timed out while capturing embedding %d/%d for ID=%ld Name=%s.\n",
        n + 1,
        MAX_EMBEDDINGS_PER_USER,
        (long)id,
        name
      );


      StaticJsonDocument<256> doc;

      doc["id"] =
        id;

      doc["name"] =
        name;

      doc["status"] =
        "failed";

      doc["reason"] =
        "face_not_detected";

      doc["embeddings"] =
        user.embeddingCount;


      publishJSON(
        TOPIC_REGISTRATION_EVENT,
        doc
      );

      oledRegisterFailed();


      // If nothing was captured, remove
      // the newly-created user.
      if (
        user.embeddingCount == 0
      ) {

        for (
          uint32_t j =
            existingIndex;
          j + 1 <
            database->userCount;
          j++
        ) {

          database->users[j] =
            database->users[j + 1];
        }


        database->userCount--;


        memset(
          &database->users[
            database->userCount
          ],
          0,
          sizeof(FaceUser)
        );
      }


      deviceBusy = false;

      return;
    }
  }


  // ----------------------------------------------------------
  // Save complete registration.
  // ----------------------------------------------------------

  if (
    !saveDatabase()
  ) {

    StaticJsonDocument<256> doc;

    doc["id"] =
      id;

    doc["name"] =
      name;

    doc["status"] =
      "failed";

    doc["reason"] =
      "database_save_failed";


    publishJSON(
      TOPIC_REGISTRATION_EVENT,
      doc
    );

    oledRegisterFailed();

    deviceBusy = false;

    return;
  }


  // ----------------------------------------------------------
  // Success.
  // ----------------------------------------------------------

  StaticJsonDocument<256> success;

  success["id"] =
    id;

  success["name"] =
    name;

  success["status"] =
    "success";

  success["embeddings"] =
    user.embeddingCount;


  publishJSON(
    TOPIC_REGISTRATION_EVENT,
    success
  );

  oledRegisterSuccess(name);

  Serial.println();
  Serial.println(
    "REGISTRATION SUCCESS."
  );


  printDatabase();


  deviceBusy = false;
}


// ============================================================
// DELETE USER
// ============================================================

void performDelete(
  int32_t id
)
{
  deviceBusy = true;
  oledMessage("DELETING", "USER...");


  Serial.println();
  Serial.println(
    "========================================"
  );

  Serial.printf(
    "DELETE USER ID %ld\n",
    (long)id
  );

  Serial.println(
    "========================================"
  );


  int index =
    findUserIndex(id);


  if (
    index < 0
  ) {

    StaticJsonDocument<256> doc;

    doc["id"] =
      id;

    doc["status"] =
      "failed";

    doc["reason"] =
      "user_not_found";


    publishJSON(
      TOPIC_DELETION_EVENT,
      doc
    );

    Serial.printf(
      "DELETE FAILED | ID=%ld | reason=user_not_found\n",
      (long)id
    );

    oledDeleteFailed("USER NOT FOUND");

    deviceBusy = false;

    return;
  }

  FaceUser &userToDelete =
    database->users[index];

  char deletedUserName[MAX_NAME_LENGTH];
  memset(deletedUserName, 0, sizeof(deletedUserName));
  strncpy(
    deletedUserName,
    userToDelete.name,
    MAX_NAME_LENGTH - 1
  );

  uint32_t deletedEmbeddingCount =
    userToDelete.embeddingCount;

  Serial.printf(
    "DELETE FOUND | ID=%ld | Name=%s | embeddings=%lu | index=%d\n",
    (long)userToDelete.id,
    deletedUserName,
    (unsigned long)deletedEmbeddingCount,
    index
  );

  oledMessage("DELETING", deletedUserName);

  // ----------------------------------------------------------
  // Shift following users left.
  // ----------------------------------------------------------

  for (
    uint32_t i = index;
    i + 1 <
      database->userCount;
    i++
  ) {

    database->users[i] =
      database->users[i + 1];
  }


  database->userCount--;


  // Clear now-unused final slot.
  memset(
    &database->users[
      database->userCount
    ],
    0,
    sizeof(FaceUser)
  );


  if (
    !saveDatabase()
  ) {

    StaticJsonDocument<256> doc;

    doc["id"] =
      id;

    doc["status"] =
      "failed";

    doc["reason"] =
      "database_save_failed";


    publishJSON(
      TOPIC_DELETION_EVENT,
      doc
    );

    Serial.printf(
      "DELETE FAILED | ID=%ld | Name=%s | reason=database_save_failed\n",
      (long)id,
      deletedUserName
    );

    oledDeleteFailed("SAVE FAILED");

    deviceBusy = false;

    return;
  }


  StaticJsonDocument<256> success;

  success["id"] =
    id;

  success["status"] =
    "success";


  publishJSON(
    TOPIC_DELETION_EVENT,
    success
  );

  Serial.println(
    "DELETE SUCCESS"
  );

  Serial.printf(
    "Deleted user | ID=%ld | Name=%s | previous embeddings=%lu\n",
    (long)id,
    deletedUserName,
    (unsigned long)deletedEmbeddingCount
  );

  Serial.printf(
    "Deletion event published on: %s\n",
    TOPIC_DELETION_EVENT
  );

  oledDeleteSuccess(deletedUserName);

  printDatabase();


  deviceBusy = false;
}


// ============================================================
// SEND INTRUDER IMAGE
// ============================================================

void sendIntruderImage(
  camera_fb_t *fb
)
{
  if (
    !mqttConnected ||
    mqttClient == nullptr
  ) {

    Serial.println(
      "Cannot send intruder image: MQTT disconnected."
    );

    return;
  }


  if (!fb) {

    Serial.println(
      "Cannot send intruder image: NULL frame."
    );

    return;
  }


  uint8_t *jpgBuf =
    nullptr;

  size_t jpgLen =
    0;


  Serial.println(
    "Converting image to JPEG..."
  );


  bool converted =
    frame2jpg(
      fb,
      70,
      &jpgBuf,
      &jpgLen
    );


  if (
    !converted ||
    jpgBuf == nullptr ||
    jpgLen == 0
  ) {

    Serial.println(
      "ERROR: JPEG conversion failed."
    );

    return;
  }


  Serial.printf(
    "Intruder JPEG size: %u bytes\n",
    (unsigned)jpgLen
  );


  int result =
    esp_mqtt_client_publish(
      mqttClient,
      TOPIC_AUTH_IMAGE,
      (const char *)jpgBuf,
      (int)jpgLen,
      0,
      0
    );


  Serial.printf(
    "Intruder image publish result: %d\n",
    result
  );


  free(jpgBuf);
}


// ============================================================
// AUTHENTICATION
// ============================================================

void performAuthentication()
{
  if (deviceBusy) {
    Serial.println("Authentication ignored: device busy.");
    return;
  }

  deviceBusy = true;

  Serial.println();
  Serial.println("========================================");
  Serial.println("AUTHENTICATION START");
  Serial.println("Searching for a face for 10 seconds...");
  Serial.println("========================================");

  logSystemEvent("auth_started", "button");

  float *embedding = authEmbeddingPSRAM;
  camera_fb_t *capturedFrame = nullptr;
  result_t capturedFace;

  // ----------------------------------------------------------
  // First stage: search for a face for up to 10 seconds.
  // ----------------------------------------------------------
  bool faceCaptured = captureAuthenticationEmbedding(
    embedding,
    &capturedFrame,
    &capturedFace
  );

  // IMPORTANT:
  // No face within 10 seconds is NOT an "unknown person".
  // Do not publish any MQTT authentication event or image.
  if (!faceCaptured) {
    if (capturedFrame) {
      esp_camera_fb_return(capturedFrame);
      capturedFrame = nullptr;
    }

    logSystemEvent("auth_no_face", "no_face_in_10_seconds");

    deviceBusy = false;
    delay(1000);
    oledReady();
    return;
  }

  // ----------------------------------------------------------
  // Face exists. A detected face is NOT an authentication.
  // We must have a valid registered embedding and a similarity
  // score that independently passes the authentication threshold.
  // ----------------------------------------------------------
  float bestSimilarity = -1.0F;
  int bestUserIndex = -1;
  int bestEmbeddingIndex = -1;

  if (!database || database->userCount == 0) {
    Serial.println("AUTH: No registered users. Access denied.");
  }

  for (uint32_t u = 0; database && u < database->userCount; u++) {
    FaceUser &user = database->users[u];

    if (user.embeddingCount == 0 || user.embeddingCount > MAX_EMBEDDINGS_PER_USER) {
      Serial.printf(
        "AUTH: Skipping invalid user record ID=%ld, embeddings=%lu\n",
        (long)user.id,
        (unsigned long)user.embeddingCount
      );
      continue;
    }

    double userSum = 0.0;
    uint32_t validCount = 0;
    float userBest = -1.0F;
    int userBestEmbedding = -1;

    for (uint32_t e = 0; e < user.embeddingCount; e++) {
      float similarity = cosineSimilarity(
        embedding,
        user.embeddings[e]
      );

      Serial.printf(
        "Compare user %ld embedding %lu: %.4f\n",
        (long)user.id,
        (unsigned long)e,
        similarity
      );

      if (isfinite(similarity)) {
        userSum += similarity;
        validCount++;
        if (similarity > userBest) {
          userBest = similarity;
          userBestEmbedding = (int)e;
        }
      }
    }

    if (validCount == 0) continue;

    float userScore = (float)(userSum / validCount);
    Serial.printf(
      "USER PROFILE: ID=%ld avg=%.4f best=%.4f valid=%lu\n",
      (long)user.id, userScore, userBest, (unsigned long)validCount
    );

    if (userScore > bestSimilarity) {
      bestSimilarity = userScore;
      bestUserIndex = (int)u;
      bestEmbeddingIndex = userBestEmbedding;
    }
  }

  Serial.println();
  Serial.printf(
    "BEST MATCH: userIndex=%d embedding=%d similarity=%.4f\n",
    bestUserIndex,
    bestEmbeddingIndex,
    bestSimilarity
  );

  // ----------------------------------------------------------
  // SECOND-FRAME VERIFICATION
  // ----------------------------------------------------------
  // Face detection alone is never enough to unlock the door.
  // If the first embedding passes the threshold, capture a
  // second fresh frame and require the same registered user to
  // pass the threshold again. This reduces false accepts caused
  // by a single poor-quality embedding.
  // NAN means: no second-frame similarity was produced.
  // It is preferable to -1 because it is not a fake similarity score.
  float verificationSimilarity = NAN;
  int verificationUserIndex = -1;
  double maxEmbeddingDelta = 0.0;
  camera_fb_t *verificationFrame = nullptr;

  if (
    bestUserIndex >= 0 &&
    bestEmbeddingIndex >= 0 &&
    isfinite(bestSimilarity) &&
    bestSimilarity >= FACE_THRESHOLD
  ) {
    float *verificationEmbedding = verifyEmbeddingPSRAM;
    result_t verificationFace;

    Serial.println("AUTH: First match passed threshold.");
    Serial.println("AUTH: Capturing second frame for verification...");

    // The second frame is a verification step, not a one-shot capture.
    // Camera exposure, motion, and MNP01 landmark refinement can fail on
    // an individual frame even when the person is still in front of the camera.
    constexpr int SECOND_FRAME_MAX_ATTEMPTS = 3;
    bool secondCaptured = false;

    for (int attempt = 1; attempt <= SECOND_FRAME_MAX_ATTEMPTS; ++attempt) {
      if (attempt > 1) {
        Serial.printf(
          "AUTH: Retrying second frame (%d/%d)...\n",
          attempt,
          SECOND_FRAME_MAX_ATTEMPTS
        );
        delay(120);
      }

      secondCaptured = captureEmbedding(
        verificationEmbedding,
        &verificationFrame,
        &verificationFace
      );

      if (secondCaptured) {
        break;
      }

      // captureEmbedding() may return a frame even when detection fails
      // because the caller requested the captured frame. Release it before
      // trying again.
      if (verificationFrame) {
        esp_camera_fb_return(verificationFrame);
        verificationFrame = nullptr;
      }
    }

    if (secondCaptured) {
      double verifySumFingerprint = 0.0;
      double verifySqFingerprint = 0.0;
      for (int i = 0; i < EMBEDDING_SIZE; i++) {
        verifySumFingerprint += verificationEmbedding[i];
        verifySqFingerprint += (double)verificationEmbedding[i] * verificationEmbedding[i];
      }
      Serial.printf(
        "VERIFY EMBEDDING FINGERPRINT | first=%.7f second=%.7f norm=%.7f sum=%.7f\n",
        verificationEmbedding[0],
        verificationEmbedding[1],
        sqrt(verifySqFingerprint),
        verifySumFingerprint
      );

      maxEmbeddingDelta = 0.0;
      for (int i = 0; i < EMBEDDING_SIZE; i++) {
        double d = fabs((double)verificationEmbedding[i] - (double)embedding[i]);
        if (d > maxEmbeddingDelta) maxEmbeddingDelta = d;
      }
      Serial.printf(
        "AUTH EMBEDDING DELTA | max=%.9f\n",
        maxEmbeddingDelta
      );

      if (maxEmbeddingDelta < 0.000001) {
        Serial.println("AUTH: WARNING - second embedding is identical to first; rejecting stale embedding.");
      }

      FaceUser &candidateUser = database->users[bestUserIndex];

      // IMPORTANT: use the same profile-average metric as the first stage.
      // Previously this loop kept the MAX single-embedding similarity, which
      // naturally made second scores look much higher than first scores.
      double verifySum = 0.0;
      uint32_t verifyValidCount = 0;

      for (uint32_t e = 0; e < candidateUser.embeddingCount; e++) {
        float similarity = cosineSimilarity(
          verificationEmbedding,
          candidateUser.embeddings[e]
        );

        Serial.printf(
          "VERIFY user %ld embedding %lu: %.4f\n",
          (long)candidateUser.id,
          (unsigned long)e,
          similarity
        );

        if (isfinite(similarity)) {
          verifySum += similarity;
          verifyValidCount++;
        }
      }

      if (verifyValidCount > 0) {
        verificationSimilarity = (float)(verifySum / verifyValidCount);
        verificationUserIndex = bestUserIndex;
        Serial.printf(
          "VERIFY PROFILE: ID=%ld avg=%.4f valid=%lu\n",
          (long)candidateUser.id,
          verificationSimilarity,
          (unsigned long)verifyValidCount
        );
      }
    }
    else {
      Serial.println("AUTH: Second frame verification failed after all retries.");
    }

    if (verificationFrame) {
      esp_camera_fb_return(verificationFrame);
      verificationFrame = nullptr;
    }
  }

  bool verifiedMatch =
    bestUserIndex >= 0 &&
    verificationUserIndex == bestUserIndex &&
    isfinite(bestSimilarity) &&
    isfinite(verificationSimilarity) &&
    bestSimilarity >= FACE_THRESHOLD &&
    verificationSimilarity >= FACE_THRESHOLD &&
    maxEmbeddingDelta > 0.000001;

  if (isfinite(verificationSimilarity)) {
    Serial.printf(
      "AUTH VERIFICATION: first=%.4f second=%.4f delta=%.9f result=%s\n",
      bestSimilarity,
      verificationSimilarity,
      maxEmbeddingDelta,
      verifiedMatch ? "PASS" : "FAIL"
    );
  } else {
    Serial.printf(
      "AUTH VERIFICATION: first=%.4f second=N/A delta=%.9f result=%s\n",
      bestSimilarity,
      maxEmbeddingDelta,
      verifiedMatch ? "PASS" : "FAIL"
    );
  }

  if (!verifiedMatch) {
    Serial.println("AUTH: ACCESS DENIED - identity verification failed.");
  }

  // ----------------------------------------------------------
  // AUTHORIZED
  // ----------------------------------------------------------
  if (verifiedMatch) {
    FaceUser &user = database->users[bestUserIndex];

    StaticJsonDocument<512> authorized;

    authorized["status"] = "authorized";
    authorized["id"] = user.id;
    authorized["name"] = user.name;
    authorized["similarity"] = min(bestSimilarity, verificationSimilarity);
    authorized["first_similarity"] = bestSimilarity;
    authorized["second_similarity"] = verificationSimilarity;
    authorized["threshold"] = FACE_THRESHOLD;
    authorized["embedding_delta"] = maxEmbeddingDelta;
    authorized["source"] = "face";

    publishJSON(
      TOPIC_AUTH_EVENT,
      authorized
    );

    Serial.println();
    Serial.println("********************************");
    Serial.println("AUTHENTICATION SUCCESS");
    Serial.printf("User ID : %ld\n", (long)user.id);
    Serial.printf("Name    : %s\n", user.name);
    Serial.printf("Score 1 : %.4f\n", bestSimilarity);
    Serial.printf("Score 2 : %.4f\n", verificationSimilarity);
    Serial.println("********************************");

    // Release the authenticated camera frame BEFORE stopping the camera.
    if (capturedFrame) {
      esp_camera_fb_return(capturedFrame);
      capturedFrame = nullptr;
    }

    // Keep the authenticated user's name on the OLED, then unlock.
    oledAuthorized(user.name);
    delay(1000);

    // Successful face authentication unlocks the door.
    // The door automatically locks again after 10 seconds.
    unlockDoor("face", user.name);
  }

  // ----------------------------------------------------------
  // FACE FOUND BUT IDENTITY UNKNOWN
  // ----------------------------------------------------------
  else {
    StaticJsonDocument<512> denied;

    denied["status"] = "denied";
    denied["id"] = -1;
    denied["name"] = "";
    denied["similarity"] = bestSimilarity;
    if (isfinite(verificationSimilarity)) {
      denied["verification_similarity"] = verificationSimilarity;
      denied["second_similarity"] = verificationSimilarity;
    } else {
      denied["verification_similarity"] = nullptr;
      denied["second_similarity"] = nullptr;
    }
    denied["first_similarity"] = bestSimilarity;
    denied["threshold"] = FACE_THRESHOLD;
    denied["embedding_delta"] = maxEmbeddingDelta;
    denied["image_topic"] = TOPIC_AUTH_IMAGE;
    denied["image_format"] = "image/jpeg";
    denied["intruder_image_stored_on_sd"] = false;
    denied["reason"] = "identity_not_recognized";

    if (bestUserIndex >= 0 && database) {
      denied["candidate_id"] = database->users[bestUserIndex].id;
      denied["candidate_name"] = database->users[bestUserIndex].name;
    }

    // This MQTT event is sent ONLY because a face was found
    // but no stored identity passed the threshold.
    publishJSON(
      TOPIC_AUTH_EVENT,
      denied
    );

    Serial.println();
    Serial.println("********************************");
    Serial.println("AUTHENTICATION DENIED");
    Serial.printf("Best similarity: %.4f\n", bestSimilarity);
    Serial.println("Sending intruder image...");
    Serial.println("********************************");

    oledUnknown();

    // Send the exact same camera frame that was authenticated.
    sendIntruderImage(capturedFrame);

    delay(2500);
  }

  // ----------------------------------------------------------
  // Release the exact frame used for authentication.
  // ----------------------------------------------------------
  if (capturedFrame) {
    esp_camera_fb_return(capturedFrame);
    capturedFrame = nullptr;
  }

  deviceBusy = false;

  // If the door was unlocked, keep the countdown on the OLED.
  // Otherwise return to the normal READY screen.
  if (!lockIsUnlocked) {
    oledReady();
  }
}


// ============================================================
// BUTTON HANDLING
// ============================================================

void checkButton()
{
  bool reading =
    digitalRead(
      BUTTON_PIN
    );


  // Detect electrical change.
  if (
    reading !=
    lastButtonReading
  ) {

    lastButtonChange =
      millis();

    lastButtonReading =
      reading;
  }


  // Wait for stable state.
  if (
    millis() -
    lastButtonChange >=
    BUTTON_DEBOUNCE_MS
  ) {

    if (
      reading !=
      stableButtonState
    ) {

      stableButtonState =
        reading;


      // INPUT_PULLUP:
      // LOW = pressed
      if (
        stableButtonState ==
        LOW
      ) {

        Serial.println();
        Serial.println(
          "********************************"
        );

        Serial.println(
          "BUTTON PRESSED"
        );

        Serial.println(
          "********************************"
        );


        performAuthentication();
      }
    }
  }
}


// ============================================================
// MQTT EVENT HANDLER
// ============================================================

static void mqttEventHandler(
  void *handler_args,
  esp_event_base_t base,
  int32_t event_id,
  void *event_data
)
{
  esp_mqtt_event_handle_t event =
    (esp_mqtt_event_handle_t)
      event_data;


  switch (
    event_id
  ) {


    // --------------------------------------------------------
    // CONNECTED
    // --------------------------------------------------------

    case MQTT_EVENT_CONNECTED:
    {
      mqttConnected =
        true;


      Serial.println();
      Serial.println(
        "MQTT CONNECTED!"
      );

      logSystemEvent("mqtt_connected");


      int registerMsg =
        esp_mqtt_client_subscribe(
          mqttClient,
          TOPIC_REGISTER_CMD,
          0
        );


      int deleteMsg =
        esp_mqtt_client_subscribe(
          mqttClient,
          TOPIC_DELETE_CMD,
          0
        );

      int unlockMsg =
        esp_mqtt_client_subscribe(
          mqttClient,
          TOPIC_UNLOCK_CMD,
          0
        );


      Serial.printf(
        "REGISTER subscription msg_id: %d\n",
        registerMsg
      );


      Serial.printf(
        "DELETE subscription msg_id: %d\n",
        deleteMsg
      );

      Serial.printf(
        "UNLOCK subscription msg_id: %d\n",
        unlockMsg
      );


      break;
    }


    // --------------------------------------------------------
    // DISCONNECTED
    // --------------------------------------------------------

    case MQTT_EVENT_DISCONNECTED:
    {
      mqttConnected =
        false;


      Serial.println(
        "MQTT DISCONNECTED!"
      );

      logSystemEvent("mqtt_disconnected");


      break;
    }


    // --------------------------------------------------------
    // ERROR
    // --------------------------------------------------------

    case MQTT_EVENT_ERROR:
    {
      Serial.println(
        "MQTT ERROR"
      );

      logSystemEvent("mqtt_error");


      break;
    }


    // --------------------------------------------------------
    // DATA
    // --------------------------------------------------------

    case MQTT_EVENT_DATA:
    {
      Serial.println();
      Serial.println(
        "========================================"
      );

      Serial.println(
        "MQTT MESSAGE RECEIVED"
      );

      Serial.println(
        "========================================"
      );


      // ------------------------------------------------------
      // Topic
      // ------------------------------------------------------

      char topic[160];


      int topicLen =
        event->topic_len;


      if (
        topicLen >=
        (int)sizeof(topic)
      ) {

        topicLen =
          sizeof(topic) - 1;
      }


      memcpy(
        topic,
        event->topic,
        topicLen
      );


      topic[topicLen] =
        '\0';


      // ------------------------------------------------------
      // Payload
      // ------------------------------------------------------

      char payload[512];


      int dataLen =
        event->data_len;


      if (
        dataLen >=
        (int)sizeof(payload)
      ) {

        Serial.println(
          "MQTT payload too large."
        );

        break;
      }


      memcpy(
        payload,
        event->data,
        dataLen
      );


      payload[dataLen] =
        '\0';


      Serial.print(
        "Topic: "
      );

      Serial.println(
        topic
      );


      Serial.print(
        "Payload: "
      );

      Serial.println(
        payload
      );


      Serial.println(
        "========================================"
      );

      logMqttCommandReceived(topic, payload);


      // ------------------------------------------------------
      // REGISTER COMMAND
      // ------------------------------------------------------

      if (
        strcmp(
          topic,
          TOPIC_REGISTER_CMD
        ) == 0
      ) {

        if (
          deviceBusy ||
          pendingCommand != CMD_NONE
        ) {

          Serial.println(
            "REGISTER rejected: device busy."
          );

          break;
        }


        StaticJsonDocument<256> doc;


        DeserializationError err =
          deserializeJson(
            doc,
            payload
          );


        if (err) {

          Serial.println(
            "ERROR: Invalid REGISTER JSON."
          );

          break;
        }


        if (
          !doc["id"].is<int>() ||
          !doc["name"].is<const char *>()
        ) {

          Serial.println(
            "ERROR: REGISTER requires id and name."
          );

          break;
        }


        int32_t id =
          doc["id"].as<int32_t>();


        const char *name =
          doc["name"].as<const char *>();


        if (
          id < 0
        ) {

          Serial.println(
            "ERROR: Invalid registration ID."
          );

          break;
        }


        if (
          name == nullptr ||
          strlen(name) == 0
        ) {

          Serial.println(
            "ERROR: Empty registration name."
          );

          break;
        }


        // Copy command into queue.
        pendingID =
          id;


        memset(
          pendingName,
          0,
          sizeof(pendingName)
        );


        strncpy(
          pendingName,
          name,
          MAX_NAME_LENGTH - 1
        );


        pendingCommand =
          CMD_REGISTER;


        Serial.printf(
          "REGISTER queued: "
          "ID=%ld Name=%s\n",
          (long)pendingID,
          pendingName
        );
      }


      // ------------------------------------------------------
      // DELETE COMMAND
      // ------------------------------------------------------

      else if (
        strcmp(
          topic,
          TOPIC_DELETE_CMD
        ) == 0
      ) {

        if (
          deviceBusy ||
          pendingCommand != CMD_NONE
        ) {

          Serial.println(
            "DELETE rejected: device busy."
          );

          break;
        }


        StaticJsonDocument<128> doc;


        DeserializationError err =
          deserializeJson(
            doc,
            payload
          );


        if (err) {

          Serial.println(
            "ERROR: Invalid DELETE JSON."
          );

          break;
        }


        if (
          !doc["id"].is<int>()
        ) {

          Serial.println(
            "ERROR: DELETE requires id."
          );

          break;
        }


        pendingID =
          doc["id"].as<int32_t>();


        pendingCommand =
          CMD_DELETE;


        Serial.printf(
          "DELETE command received and queued: ID=%ld\n",
          (long)pendingID
        );
      }

      // ------------------------------------------------------
      // UNLOCK COMMAND
      // ------------------------------------------------------

      else if (
        strcmp(
          topic,
          TOPIC_UNLOCK_CMD
        ) == 0
      ) {

        if (
          deviceBusy ||
          pendingCommand != CMD_NONE
        ) {

          Serial.println(
            "UNLOCK rejected: device busy."
          );

          break;
        }

        // Manual unlock command. The payload can be "{}"
        // or empty; no special JSON fields are required.
        Serial.println(
          "MANUAL UNLOCK command received and queued."
        );

        pendingCommand =
          CMD_UNLOCK;
      }


      break;
    }


    default:
      break;
  }
}


// ============================================================
// MQTT INITIALIZATION
// ============================================================

bool initMQTT()
{
  esp_mqtt_client_config_t mqttConfig =
    {};


  mqttConfig.broker.address.uri =
    MQTT_URI;


  // Use ESP32's built-in CA bundle.
  mqttConfig.broker.verification
    .crt_bundle_attach =
      esp_crt_bundle_attach;


  // ----------------------------------------------------------
  // Generate unique client ID.
  // ----------------------------------------------------------

  const char *clientID = getDeviceId();


  mqttConfig.credentials.client_id =
    clientID;


  Serial.println();
  Serial.println(
    "Starting MQTT over WSS..."
  );


  Serial.print(
    "URI: "
  );

  Serial.println(
    MQTT_URI
  );


  Serial.print(
    "Client ID: "
  );

  Serial.println(
    clientID
  );


  mqttClient =
    esp_mqtt_client_init(
      &mqttConfig
    );


  if (
    mqttClient == nullptr
  ) {

    Serial.println(
      "ERROR: MQTT client initialization failed."
    );

    return false;
  }


  esp_mqtt_client_register_event(
    mqttClient,
    MQTT_EVENT_ANY,
    mqttEventHandler,
    nullptr
  );


  esp_err_t err =
    esp_mqtt_client_start(
      mqttClient
    );


  Serial.printf(
    "esp_mqtt_client_start(): %d\n",
    err
  );


  return err == ESP_OK;
}


// ============================================================
// WIFI
// ============================================================

bool connectWiFi()
{
  Serial.println();
  Serial.print(
    "Connecting to WiFi"
  );


  WiFi.begin(
    WIFI_SSID,
    WIFI_PASSWORD
  );


  int attempts =
    0;


  while (
    WiFi.status() !=
      WL_CONNECTED &&
    attempts < 60
  ) {

    delay(500);

    Serial.print(".");

    attempts++;
  }


  Serial.println();


  if (
    WiFi.status() !=
    WL_CONNECTED
  ) {

    Serial.println(
      "WiFi connection failed."
    );

    return false;
  }


  Serial.println(
    "WiFi connected!"
  );


  Serial.print(
    "IP address: "
  );

  Serial.println(
    WiFi.localIP()
  );


  return true;
}


// ============================================================
// SD INITIALIZATION
// ============================================================

bool initSD()
{
  Serial.println();
  Serial.println(
    "Initializing SD card..."
  );


  if (
    !SD_MMC.setPins(
      39,  // CLK
      38,  // CMD
      40   // D0
    )
  ) {

    Serial.println(
      "ERROR: SD pin configuration failed."
    );

    return false;
  }


  if (
    !SD_MMC.begin(
      "/sdcard",
      true
    )
  ) {

    Serial.println(
      "ERROR: SD mount failed."
    );

    return false;
  }


  Serial.println(
    "SD mounted successfully."
  );

  sdReady = true;


  if (
    !SD_MMC.exists(
      "/esp32"
    )
  ) {

    Serial.println(
      "Creating /esp32 directory..."
    );


    if (
      !SD_MMC.mkdir(
        "/esp32"
      )
    ) {

      Serial.println(
        "ERROR: Could not create /esp32."
      );

      return false;
    }
  }


  if (!SD_MMC.exists(EVENT_LOG_PATH)) {
    File logFile = SD_MMC.open(
      EVENT_LOG_PATH,
      FILE_WRITE
    );

    if (!logFile) {
      Serial.println("ERROR: Could not create event log file.");
      sdReady = false;
      return false;
    }

    logFile.close();
  }

  Serial.printf(
    "SD event log: %s\n",
    EVENT_LOG_PATH
  );

  return true;
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
  Serial.begin(115200);

  // ----------------------------------------------------------
  // LOCK - fail-safe boot state
  // ----------------------------------------------------------
  // LOW = LOCKED. Set this as early as possible.
  pinMode(LOCK_PIN, OUTPUT);
  digitalWrite(LOCK_PIN, LOW);

  lockIsUnlocked = false;
  unlockStartedAt = 0;

  delay(1000);

  Serial.println();
  Serial.println("==========================================");
  Serial.println("ESP32-S3 FACE AUTHENTICATION SYSTEM");
  Serial.println("==========================================");

  // ----------------------------------------------------------
  // OLED
  // ----------------------------------------------------------
  Wire.begin(OLED_SDA, OLED_SCL);

  if (!display.begin(OLED_ADDR, true)) {
    Serial.println("OLED initialization failed.");
    while (true) {
      delay(1000);
    }
  }

  display.setRotation(3);  // portrait, rotated 180 degrees: logical 64 x 128
  oledMessage("FACE AUTH", "STARTING...");

  // ----------------------------------------------------------
  // Button
  // ----------------------------------------------------------
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  stableButtonState = digitalRead(BUTTON_PIN);
  lastButtonReading = stableButtonState;

  Serial.printf(
    "BUTTON GPIO %d initial state: %s\n",
    BUTTON_PIN,
    stableButtonState ? "HIGH" : "LOW"
  );

  // ----------------------------------------------------------
  // PSRAM database
  // ----------------------------------------------------------
  Serial.println();
  Serial.println("Initializing PSRAM database...");

  Serial.printf(
    "PSRAM total: %u bytes\n",
    (unsigned)ESP.getPsramSize()
  );

  Serial.printf(
    "PSRAM free: %u bytes\n",
    (unsigned)ESP.getFreePsram()
  );

  database =
    (FaceDatabase *)ps_malloc(sizeof(FaceDatabase));

  if (!database) {
    Serial.println("ERROR: Could not allocate database in PSRAM.");
    while (true) {
      delay(1000);
    }
  }

  Serial.printf(
    "Database allocated: %u bytes\n",
    (unsigned)sizeof(FaceDatabase)
  );

  // ----------------------------------------------------------
  // Authentication embedding buffers
  // ----------------------------------------------------------
  // Keep the two 512-float authentication buffers in PSRAM.
  // A 512-float array is 2048 bytes; two of them plus other local
  // variables can overflow the ESP32 loopTask stack.
  authEmbeddingPSRAM =
    (float *)ps_malloc(EMBEDDING_SIZE * sizeof(float));

  verifyEmbeddingPSRAM =
    (float *)ps_malloc(EMBEDDING_SIZE * sizeof(float));

  if (!authEmbeddingPSRAM || !verifyEmbeddingPSRAM) {
    Serial.println(
      "ERROR: Could not allocate authentication buffers in PSRAM."
    );
    while (true) {
      delay(1000);
    }
  }

  Serial.printf(
    "Auth embedding buffers allocated in PSRAM: %u bytes each\n",
    (unsigned)(EMBEDDING_SIZE * sizeof(float))
  );

  // ----------------------------------------------------------
  // SD
  // ----------------------------------------------------------
  if (!initSD()) {
    while (true) {
      delay(1000);
    }
  }

  if (!loadDatabase()) {
    Serial.println("ERROR: Database initialization failed.");
    while (true) {
      delay(1000);
    }
  }

  // ----------------------------------------------------------
  // Camera
  // ----------------------------------------------------------
  if (!initCamera()) {
    while (true) {
      delay(1000);
    }
  }

  // ----------------------------------------------------------
  // Face recognition
  // ----------------------------------------------------------
  recognizer.set_thresh(FACE_THRESHOLD);

  Serial.printf(
    "Face recognition threshold: %.2f\n",
    FACE_THRESHOLD
  );

  // ----------------------------------------------------------
  // WiFi
  // ----------------------------------------------------------
  if (!connectWiFi()) {
    oledMessage("WIFI FAILED");
    while (true) {
      delay(1000);
    }
  }

  // ----------------------------------------------------------
  // CLOCK / NTP
  // ----------------------------------------------------------
  syncSystemClock();

  // ----------------------------------------------------------
  // MQTT
  // ----------------------------------------------------------
  if (!initMQTT()) {
    Serial.println("MQTT initialization failed.");
    oledMessage("MQTT FAILED");
  }

  // ----------------------------------------------------------
  // Ready
  // ----------------------------------------------------------
  Serial.println();
  Serial.println("==========================================");
  Serial.println("SYSTEM READY");
  Serial.println("==========================================");
  Serial.println("REGISTER: MQTT command");
  Serial.println("DELETE  : MQTT command");
  Serial.println("UNLOCK  : MQTT command");
  Serial.println("AUTH    : Physical button GPIO 1");
  Serial.println("LOCK    : GPIO 2 | LOW=LOCKED | HIGH=UNLOCKED");
  Serial.println();

  logSystemEvent("system_ready");
  oledReady();
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
  // ----------------------------------------------------------
  // Automatic lock timer
  // ----------------------------------------------------------
  serviceAutoLock();

  // ----------------------------------------------------------
  // Physical authentication button
  // ----------------------------------------------------------
  checkButton();

  // ----------------------------------------------------------
  // Process queued MQTT command outside the MQTT callback.
  // ----------------------------------------------------------
  if (pendingCommand != CMD_NONE && !deviceBusy) {
    PendingCommandType command = pendingCommand;
    int32_t id = pendingID;

    char name[MAX_NAME_LENGTH];
    strncpy(name, pendingName, MAX_NAME_LENGTH);
    name[MAX_NAME_LENGTH - 1] = '\0';

    pendingCommand = CMD_NONE;
    pendingID = -1;
    pendingName[0] = '\0';

    if (command == CMD_REGISTER) {
      performRegistration(id, name);
      oledReady();
    }
    else if (command == CMD_DELETE) {
      performDelete(id);
      oledReady();
    }
    else if (command == CMD_UNLOCK) {
      // Manual MQTT unlock. It uses the same 10-second
      // automatic relock safety timeout.
      unlockDoor("manual");
    }
  }

  delay(5);
}
