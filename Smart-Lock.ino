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

#define TOPIC_REGISTRATION_EVENT \
  "847291/583104/event/registration"

#define TOPIC_DELETION_EVENT \
  "847291/583104/event/deletion"

#define TOPIC_AUTH_EVENT \
  "847291/583104/event/auth"

#define TOPIC_AUTH_IMAGE \
  "847291/583104/event/auth/image"


esp_mqtt_client_handle_t mqttClient = nullptr;

volatile bool mqttConnected = false;


// ============================================================
// BUTTON
// ============================================================

#define BUTTON_PIN 1

#define BUTTON_DEBOUNCE_MS 50

bool lastButtonReading = HIGH;
bool stableButtonState = HIGH;
unsigned long lastButtonChange = 0;


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

#define MAX_USERS                 10
#define MAX_NAME_LENGTH           32
#define EMBEDDING_SIZE            512
#define MAX_EMBEDDINGS_PER_USER   10


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
#define FACE_THRESHOLD 0.70F


// ============================================================
// COMMAND QUEUE
// ============================================================

enum PendingCommandType {

  CMD_NONE,

  CMD_REGISTER,

  CMD_DELETE
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
  if (
    !mqttConnected ||
    mqttClient == nullptr
  ) {

    Serial.println(
      "MQTT not connected; JSON not published."
    );

    return false;
  }


  char buffer[512];


  size_t len =
    serializeJson(
      doc,
      buffer,
      sizeof(buffer)
    );


  int result =
    esp_mqtt_client_publish(
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

bool detectFace(
  camera_fb_t *fb,
  std::vector<int> &landmarks,
  result_t *bestResult = nullptr
)
{
  if (!fb) {
    return false;
  }

  std::vector<int> shape = {
    (int)fb->height,
    (int)fb->width,
    3
  };

  // Camera is RGB565, so explicitly select the RGB565 path.
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

  detector2.infer(
    (uint16_t *)fb->buf,
    shape,
    candidates
  );

  if (candidates.empty()) {
    Serial.println("MNP01: no face.");
    return false;
  }

  // Pick the highest scoring face.
  auto best = candidates.begin();

  for (auto it = candidates.begin();
       it != candidates.end();
       ++it) {
    if (it->score > best->score) {
      best = it;
    }
  }

  Serial.printf(
    "FACE DETECTED | score=%.3f | box=(%d,%d)-(%d,%d)\n",
    best->score,
    best->box[0],
    best->box[1],
    best->box[2],
    best->box[3]
  );

  landmarks.clear();

  for (int i = 0; i < 10; i++) {
    landmarks.push_back(best->keypoint[i]);
  }

  if (bestResult) {
    *bestResult = *best;
  }

  return true;
}


// ============================================================
// CAPTURE FRESH EMBEDDING
// ============================================================

bool captureEmbedding(
  float *outputEmbedding,
  camera_fb_t **capturedFrame = nullptr,
  result_t *capturedFace = nullptr
)
{
  if (!outputEmbedding) {
    return false;
  }


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


  std::vector<int> shape = {
    (int)fb->height,
    (int)fb->width,
    3
  };


  face_info_t info =
    recognizer.recognize(
      (uint16_t *)fb->buf,
      shape,
      landmarks
    );


  Tensor<float> &embedding =
    recognizer.get_face_emb();


  if (
    embedding.get_size() !=
    EMBEDDING_SIZE
  ) {

    Serial.printf(
      "ERROR: Wrong embedding size: %d\n",
      (int)embedding.get_size()
    );


    if (capturedFrame) {

      *capturedFrame =
        fb;

    } else {

      esp_camera_fb_return(
        fb
      );
    }


    return false;
  }


  memcpy(
    outputEmbedding,
    embedding.element,
    EMBEDDING_SIZE *
    sizeof(float)
  );


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

void oledMessage(
  const char *line1,
  const char *line2 = nullptr,
  const char *line3 = nullptr
)
{
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);
  display.setTextSize(1);

  int y = 30;

  if (line1) {
    int x = max(0, (display.width() - (int)strlen(line1) * 6) / 2);
    display.setCursor(x, y);
    display.println(line1);
    y += 14;
  }

  if (line2) {
    int x = max(0, (display.width() - (int)strlen(line2) * 6) / 2);
    display.setCursor(x, y);
    display.println(line2);
    y += 14;
  }

  if (line3) {
    int x = max(0, (display.width() - (int)strlen(line3) * 6) / 2);
    display.setCursor(x, y);
    display.println(line3);
  }

  display.display();
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
  left   = constrain(left,   0, imageWidth - 1);
  top    = constrain(top,    0, imageHeight - 1);
  right  = constrain(right,  0, imageWidth - 1);
  bottom = constrain(bottom, 0, imageHeight - 1);

  int faceWidth  = right - left;
  int faceHeight = bottom - top;

  if (faceWidth <= 0 || faceHeight <= 0) {
    return;
  }

  display.clearDisplay();

  int oledWidth  = display.width();
  int oledHeight = display.height();

  float scaleX = (float)oledWidth / faceWidth;
  float scaleY = (float)oledHeight / faceHeight;
  float scale  = min(scaleX, scaleY);

  int drawWidth  = max(1, (int)(faceWidth * scale));
  int drawHeight = max(1, (int)(faceHeight * scale));

  int offsetX = (oledWidth - drawWidth) / 2;
  int offsetY = (oledHeight - drawHeight) / 2;

  for (int y = 0; y < drawHeight; y++) {
    for (int x = 0; x < drawWidth; x++) {
      int srcX = left + (x * faceWidth) / drawWidth;
      int srcY = top  + (y * faceHeight) / drawHeight;

      uint16_t pixel = image[srcY * imageWidth + srcX];

      uint8_t r = ((pixel >> 11) & 0x1F) << 3;
      uint8_t g = ((pixel >> 5)  & 0x3F) << 2;
      uint8_t b = (pixel & 0x1F) << 3;

      uint8_t gray =
        (uint16_t)(r * 30 + g * 59 + b * 11) / 100;

      if (gray > 110) {
        display.drawPixel(
          offsetX + x,
          offsetY + y,
          SH110X_WHITE
        );
      }
    }
  }

  display.display();
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

      delay(500);

      *capturedFrame = fb;
      *capturedFace = face;

      std::vector<int> shape = {
        (int)fb->height,
        (int)fb->width,
        3
      };

      oledAuthenticating();

      face_info_t info =
        recognizer.recognize(
          (uint16_t *)fb->buf,
          shape,
          landmarks
        );

      (void)info;

      Tensor<float> &embedding =
        recognizer.get_face_emb();

      if (embedding.get_size() != EMBEDDING_SIZE) {
        Serial.printf(
          "ERROR: Wrong embedding size: %d\n",
          (int)embedding.get_size()
        );
        return false;
      }

      memcpy(
        outputEmbedding,
        embedding.element,
        EMBEDDING_SIZE * sizeof(float)
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

      float embedding[
        EMBEDDING_SIZE
      ];

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

  float embedding[EMBEDDING_SIZE];
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

    deviceBusy = false;
    delay(1000);
    oledReady();
    return;
  }

  // ----------------------------------------------------------
  // Face exists. Compare it with every registered embedding.
  // ----------------------------------------------------------
  float bestSimilarity = -1.0F;
  int bestUserIndex = -1;
  int bestEmbeddingIndex = -1;

  for (uint32_t u = 0; u < database->userCount; u++) {
    FaceUser &user = database->users[u];

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

      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestUserIndex = (int)u;
        bestEmbeddingIndex = (int)e;
      }
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
  // AUTHORIZED
  // ----------------------------------------------------------
  if (bestUserIndex >= 0 && bestSimilarity >= FACE_THRESHOLD) {
    FaceUser &user = database->users[bestUserIndex];

    StaticJsonDocument<256> authorized;

    authorized["status"] = "authorized";
    authorized["id"] = user.id;
    authorized["name"] = user.name;
    authorized["similarity"] = bestSimilarity;

    publishJSON(
      TOPIC_AUTH_EVENT,
      authorized
    );

    Serial.println();
    Serial.println("********************************");
    Serial.println("AUTHENTICATION SUCCESS");
    Serial.printf("User ID : %ld\n", (long)user.id);
    Serial.printf("Name    : %s\n", user.name);
    Serial.printf("Score   : %.4f\n", bestSimilarity);
    Serial.println("********************************");

    oledAuthorized(user.name);
    delay(2500);
  }

  // ----------------------------------------------------------
  // FACE FOUND BUT IDENTITY UNKNOWN
  // ----------------------------------------------------------
  else {
    StaticJsonDocument<256> denied;

    denied["status"] = "denied";
    denied["id"] = -1;
    denied["name"] = "";
    denied["similarity"] = bestSimilarity;
    denied["image_topic"] = TOPIC_AUTH_IMAGE;
    denied["image_format"] = "image/jpeg";
    denied["reason"] = "identity_not_recognized";

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
  oledReady();
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


      Serial.printf(
        "REGISTER subscription msg_id: %d\n",
        registerMsg
      );


      Serial.printf(
        "DELETE subscription msg_id: %d\n",
        deleteMsg
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

  uint64_t chipid =
    ESP.getEfuseMac();


  char clientID[40];


  snprintf(
    clientID,
    sizeof(clientID),
    "ESP32CAM_%04X%08X",
    (uint16_t)(chipid >> 32),
    (uint32_t)chipid
  );


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


  return true;
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
  Serial.begin(115200);
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

  display.setRotation(1);  // portrait: logical 64 x 128
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
  Serial.println("AUTH    : Physical button GPIO 1");
  Serial.println();

  oledReady();
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
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
    }
    else if (command == CMD_DELETE) {
      performDelete(id);
    }

    oledReady();
  }

  delay(5);
}
