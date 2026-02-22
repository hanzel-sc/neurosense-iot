#include <WiFi.h>
#include <WiFiClient.h>
#include <Wire.h>

#include "MAX30105.h"
#include "heartRate.h"

#include <DHT.h>

// ---------------- WIFI + THINGSPEAK ----------------
const char* WIFI_SSID = "xxxx";
const char* WIFI_PASS = "xxxx";

const char* THINGSPEAK_HOST = "api.thingspeak.com";
const char* THINGSPEAK_WRITE_KEY = "xxxxx";

// add your own credentials people :)
// it's not good to steal yk...

const unsigned long UPLOAD_INTERVAL_MS = 20000; // >= 15s for free ThingSpeak

// ---------------- PINS ----------------
#define I2C_SDA 21
#define I2C_SCL 22

#define DHT_PIN 27
#define DHT_TYPE DHT11

#define LED_GREEN 16
#define LED_BLUE  17
#define LED_RED   18

// ---------------- BASELINE / LED LOGIC ----------------
const int BASELINE_SECONDS = 30;

// NEW: baseline countdown starts only after finger is continuously present for this long:
const unsigned long FINGER_STABLE_MS = 2000;

const int MIN_BASELINE_HR_SAMPLES = 15;
const int MIN_BASELINE_RMSSD_SAMPLES = 8;

const int LED_STREAK_REQUIRED = 3;

// ---------------- STRESS SCORING TUNING ----------------
const float HR_RISE_BPM    = 15.0;
const float RMSSD_DROP_MS  = 15.0;

const float HUMID_HIGH   = 75.0; // %
const float AMBIENT_HOT  = 32.0; // Â°C

// ---------------- HR smoothing ----------------
const int HR_SMOOTH_N = 5;

// ---------------- VALIDITY FILTERS ----------------
const float HR_MIN_VALID = 40.0;
const float HR_MAX_VALID = 180.0;

const float RMSSD_MIN_VALID = 5.0;
const float RMSSD_MAX_VALID = 250.0;

const float TEMP_MIN_VALID = 0.0;
const float TEMP_MAX_VALID = 60.0;

const float HUM_MIN_VALID = 1.0;
const float HUM_MAX_VALID = 100.0;

// ---------------- MAX30102 ----------------
MAX30105 particleSensor;

// HRV (IBI buffer)
static const int IBI_BUF = 20;
float ibi_ms[IBI_BUF];
int ibi_idx = 0;
int ibi_count = 0;

float currentBPM = NAN;     // raw BPM
float rmssd = NAN;
unsigned long lastBeat = 0;

// Smoothed HR
float hrBuf[HR_SMOOTH_N];
int hrBufIdx = 0;
int hrBufCount = 0;
float smoothedBPM = NAN;

// ---------------- DHT ----------------
DHT dht(DHT_PIN, DHT_TYPE);
float ambTempC = NAN;
float humidity = NAN;

// ---------------- BASELINES ----------------
bool baselineDone = false;
bool baselineArmed = false;            // NEW: baseline countdown running only after finger stable
unsigned long baselineStartMs = 0;     // NEW: set when baselineArmed becomes true
unsigned long fingerStableStartMs = 0; // NEW: tracks continuous finger presence

float baseHR = NAN;
float baseRMSSD = NAN;

float baselineHRSum = 0; int baselineHRSamples = 0;
float baselineRMSSDSum = 0; int baselineRMSSDSamples = 0;

// Upload timer
unsigned long lastUploadMs = 0;

// ---------------- LED STREAK LOGIC ----------------
enum StressState { ST_CALM=0, ST_NEUTRAL=1, ST_HIGH=2, ST_NONE=3 };
StressState lastState = ST_NONE;
int stateStreak = 0;

// ---------------- HELPERS ----------------
float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

bool isValidRange(float v, float lo, float hi) {
  return !isnan(v) && v >= lo && v <= hi;
}

void setLEDs(bool g, bool b, bool r) {
  digitalWrite(LED_GREEN, g ? HIGH : LOW);
  digitalWrite(LED_BLUE,  b ? HIGH : LOW);
  digitalWrite(LED_RED,   r ? HIGH : LOW);
}

void showStateLED(StressState st) {
  if (st == ST_CALM)         setLEDs(true,  false, false);
  else if (st == ST_NEUTRAL) setLEDs(false, true,  false);
  else if (st == ST_HIGH)    setLEDs(false, false, true );
  else                       setLEDs(false, false, false);
}

float computeRMSSD() {
  if (ibi_count < 6) return NAN;

  float sumSq = 0;
  int diffs = 0;

  for (int i = 1; i < ibi_count; i++) {
    int idx1 = (ibi_idx - i + IBI_BUF) % IBI_BUF;
    int idx2 = (ibi_idx - i - 1 + IBI_BUF) % IBI_BUF;
    float d = ibi_ms[idx1] - ibi_ms[idx2];
    sumSq += d * d;
    diffs++;
  }
  if (diffs <= 0) return NAN;
  return sqrt(sumSq / diffs);
}

void pushHR(float bpm) {
  if (isnan(bpm)) return;
  hrBuf[hrBufIdx] = bpm;
  hrBufIdx = (hrBufIdx + 1) % HR_SMOOTH_N;
  if (hrBufCount < HR_SMOOTH_N) hrBufCount++;
}

float getSmoothedHR() {
  if (hrBufCount == 0) return NAN;
  float sum = 0;
  for (int i = 0; i < hrBufCount; i++) sum += hrBuf[i];
  return sum / hrBufCount;
}

void resetBaseline() {
  baselineHRSum = 0; baselineHRSamples = 0;
  baselineRMSSDSum = 0; baselineRMSSDSamples = 0;

  baselineDone = false;

  baselineArmed = false;
  fingerStableStartMs = 0;
  baselineStartMs = 0;

  lastState = ST_NONE;
  stateStreak = 0;

  showStateLED(ST_NEUTRAL);
  Serial.println("Waiting for finger (stable) to start baseline...");
}

int computeStressScore(float hrSmooth, float rmssdNow, float tC, float hum) {
  if (!baselineDone || isnan(hrSmooth) || isnan(rmssdNow)) return -1;

  float hrDelta = hrSmooth - baseHR;
  float rmssdDelta = baseRMSSD - rmssdNow;

  float hrScore  = clampf((hrDelta / (HR_RISE_BPM * 2.0f)) * 45.0f, 0, 45);
  float hrvScore = clampf((rmssdDelta / (RMSSD_DROP_MS * 2.0f)) * 55.0f, 0, 55);

  float score = hrScore + hrvScore;

  bool hotOrHumid = (isValidRange(tC, TEMP_MIN_VALID, TEMP_MAX_VALID) && tC > AMBIENT_HOT) ||
                    (isValidRange(hum, HUM_MIN_VALID, HUM_MAX_VALID) && hum > HUMID_HIGH);
  if (hotOrHumid) score *= 0.85f;

  return (int)clampf(score, 0, 100);
}

StressState classifyState(int score) {
  if (score < 0) return ST_NONE;
  if (score < 30) return ST_CALM;
  if (score < 60) return ST_NEUTRAL;
  return ST_HIGH;
}

const char* stressLabelFromState(StressState st) {
  if (st == ST_CALM) return "CALM";
  if (st == ST_NEUTRAL) return "NEUTRAL";
  if (st == ST_HIGH) return "HIGH";
  return "NO_BASELINE/NO_DATA";
}

void updateLEDWithStreak(StressState current) {
  if (!baselineDone) { showStateLED(ST_NEUTRAL); return; }
  if (current == ST_NONE) return;

  if (current == lastState) stateStreak++;
  else { lastState = current; stateStreak = 1; }

  if (stateStreak >= LED_STREAK_REQUIRED) showStateLED(current);
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("Connecting to WiFi");
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
    if (millis() - start > 20000) {
      Serial.println("\nWiFi connect timeout. Rebooting in 3s...");
      delay(3000);
      ESP.restart();
    }
  }
  Serial.println("\nWiFi connected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
}

bool uploadToThingSpeak(float bpm, float rmssdMs, float tempC, float hum, int stressScore) {
  WiFiClient client;
  if (!client.connect(THINGSPEAK_HOST, 80)) {
    Serial.println("ThingSpeak connect failed.");
    return false;
  }

  char url[256];
  int n = snprintf(url, sizeof(url), "/update?api_key=%s", THINGSPEAK_WRITE_KEY);
  if (n < 0 || n >= (int)sizeof(url)) { client.stop(); return false; }

  auto appendField = [&](const char* field, const char* val) {
    size_t len = strlen(url);
    snprintf(url + len, sizeof(url) - len, "&%s=%s", field, val);
  };

  char buf[32];
  int fieldsAdded = 0;

  // Add fields only if valid AND count them
  if (isValidRange(bpm, HR_MIN_VALID, HR_MAX_VALID)) {
    snprintf(buf, sizeof(buf), "%.1f", bpm); appendField("field1", buf); fieldsAdded++;
  }
  if (isValidRange(rmssdMs, RMSSD_MIN_VALID, RMSSD_MAX_VALID)) {
    snprintf(buf, sizeof(buf), "%.1f", rmssdMs); appendField("field2", buf); fieldsAdded++;
  }
  if (isValidRange(tempC, TEMP_MIN_VALID, TEMP_MAX_VALID)) {
    snprintf(buf, sizeof(buf), "%.1f", tempC); appendField("field3", buf); fieldsAdded++;
  }
  if (isValidRange(hum, HUM_MIN_VALID, HUM_MAX_VALID)) {
    snprintf(buf, sizeof(buf), "%.1f", hum); appendField("field4", buf); fieldsAdded++;
  }
  if (stressScore >= 0) {
    snprintf(buf, sizeof(buf), "%d", stressScore); appendField("field5", buf); fieldsAdded++;
  }

  if (fieldsAdded == 0) {
    Serial.println("ThingSpeak: SKIPPED (no valid fields to send).");
    client.stop();
    return false;
  }

  // Print the exact URL once (debug)
  Serial.print("TS URL: ");
  Serial.println(url);

  client.print(String("GET ") + url + " HTTP/1.1\r\n" +
               "Host: " + String(THINGSPEAK_HOST) + "\r\n" +
               "Connection: close\r\n\r\n");

  // Read response fully
  String resp = "";
  unsigned long timeout = millis();
  while (client.connected() && millis() - timeout < 6000) {
    while (client.available()) {
      char c = client.read();
      resp += c;
      timeout = millis();
    }
  }
  client.stop();

  // Extract body (ThingSpeak returns entry id or 0)
  int bodyPos = resp.lastIndexOf("\r\n\r\n");
  String body = (bodyPos >= 0) ? resp.substring(bodyPos + 4) : resp;
  body.trim();

  Serial.print("ThingSpeak body: '");
  Serial.print(body);
  Serial.println("'");

  if (body == "0" || body.length() == 0) {
    Serial.println("ThingSpeak REJECTED the update (0).");
    return false;
  }

  Serial.println("ThingSpeak OK (entry id above).");
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(200);

  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_BLUE, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  setLEDs(false, false, false);

  Wire.begin(I2C_SDA, I2C_SCL);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX3010x not found.");
    showStateLED(ST_HIGH);
    while (1) delay(100);
  }

  // PPG config
  byte ledBrightness = 60;
  byte sampleAverage = 4;
  byte ledMode = 2;
  int sampleRate = 200;
  int pulseWidth = 411;
  int adcRange = 16384;

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeIR(0x1F);

  dht.begin();

  connectWiFi();

  Serial.println("\nESP32 Stress Monitor (MAX30102 + DHT11) + ThingSpeak + LEDs");
  Serial.print("Baselining for "); Serial.print(BASELINE_SECONDS);
  Serial.println(" seconds... Keep relaxed and still.");
  resetBaseline();
}

void loop() {
  // ----- PPG -----
  long irValue = particleSensor.getIR();
  bool fingerPresent = irValue > 2000;

  if (fingerPresent) {
    if (checkForBeat(irValue)) {
      unsigned long now = millis();
      unsigned long delta = now - lastBeat;
      lastBeat = now;

      // reject crazy beats (reduces spikes)
      if (delta > 350 && delta < 1500) {
        float ibi = (float)delta;

        ibi_ms[ibi_idx] = ibi;
        ibi_idx = (ibi_idx + 1) % IBI_BUF;
        if (ibi_count < IBI_BUF) ibi_count++;

        currentBPM = 60000.0 / ibi;
        rmssd = computeRMSSD();

        pushHR(currentBPM);
        smoothedBPM = getSmoothedHR();
      }
    }
  } else {
    currentBPM = NAN;
    rmssd = NAN;
    smoothedBPM = NAN;
    hrBufCount = 0;
  }

  // ----- DHT11 (every ~2s) -----
  static unsigned long lastDHTRead = 0;
  if (millis() - lastDHTRead > 2000) {
    lastDHTRead = millis();
    float h = dht.readHumidity();
    float t = dht.readTemperature();
    if (isValidRange(h, HUM_MIN_VALID, HUM_MAX_VALID)) humidity = h;
    if (isValidRange(t, TEMP_MIN_VALID, TEMP_MAX_VALID)) ambTempC = t;
  }

  // ----- BASELINE GATING (NEW) -----
  if (!baselineDone) {
    if (!baselineArmed) {
      // waiting for stable finger
      if (fingerPresent) {
        if (fingerStableStartMs == 0) fingerStableStartMs = millis();
        if (millis() - fingerStableStartMs >= FINGER_STABLE_MS) {
          baselineArmed = true;
          baselineStartMs = millis();
          Serial.println("\nFinger stable âœ… Starting baseline countdown now...");
        }
      } else {
        fingerStableStartMs = 0;
      }
    } else {
      // baseline armed, but finger removed: pause baseline
      if (!fingerPresent) {
        baselineArmed = false;
        fingerStableStartMs = 0;
        Serial.println("\nFinger removed âŒ Baseline paused. Waiting for stable finger again...");
      }
    }
  }

  // ----- BASELINE COLLECTION (FILTERED, only when ARMED) -----
  if (!baselineDone && baselineArmed) {
    if (fingerPresent && isValidRange(smoothedBPM, HR_MIN_VALID, HR_MAX_VALID)) {
      baselineHRSum += smoothedBPM;
      baselineHRSamples++;
    }
    if (fingerPresent && isValidRange(rmssd, RMSSD_MIN_VALID, RMSSD_MAX_VALID)) {
      baselineRMSSDSum += rmssd;
      baselineRMSSDSamples++;
    }

    if ((millis() - baselineStartMs) >= (unsigned long)BASELINE_SECONDS * 1000UL) {
      if (baselineHRSamples < MIN_BASELINE_HR_SAMPLES) {
        Serial.println("\nBaseline failed: not enough VALID HR samples. Keep finger still and retry.");
        resetBaseline();
      } else if (baselineRMSSDSamples < MIN_BASELINE_RMSSD_SAMPLES) {
        Serial.println("\nBaseline failed: not enough VALID RMSSD samples. Keep finger still and retry.");
        resetBaseline();
      } else {
        baseHR = baselineHRSum / baselineHRSamples;
        baseRMSSD = baselineRMSSDSum / baselineRMSSDSamples;
        baselineDone = true;

        Serial.println("\n--- BASELINE SET (FILTERED + STABLE FINGER) ---");
        Serial.print("Base HR (smooth): "); Serial.println(baseHR, 2);
        Serial.print("Base RMSSD: "); Serial.println(baseRMSSD, 2);
        Serial.print("HR samples: "); Serial.println(baselineHRSamples);
        Serial.print("RMSSD samples: "); Serial.println(baselineRMSSDSamples);
        Serial.println("----------------------------------------------\n");
      }
    }
  }

  // ----- Score + classification -----
  int score = computeStressScore(smoothedBPM, rmssd, ambTempC, humidity);
  StressState st = classifyState(score);

  // ----- LED with streak requirement -----
  updateLEDWithStreak(st);

  // ----- Serial output -----
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 1000) {
    lastPrint = millis();

    Serial.print("IR: "); Serial.print(irValue);
    Serial.print(" | HR(raw): "); (isnan(currentBPM) ? Serial.print("NA") : Serial.print(currentBPM, 1));
    Serial.print(" | HR(smooth): "); (isnan(smoothedBPM) ? Serial.print("NA") : Serial.print(smoothedBPM, 1));
    Serial.print(" | RMSSD: "); (isnan(rmssd) ? Serial.print("NA") : Serial.print(rmssd, 1));
    Serial.print(" | TempC: "); (isnan(ambTempC) ? Serial.print("NA") : Serial.print(ambTempC, 1));
    Serial.print(" | Hum%: "); (isnan(humidity) ? Serial.print("NA") : Serial.print(humidity, 1));
    Serial.print(" | Score: "); Serial.print(score);
    Serial.print(" ("); Serial.print(stressLabelFromState(st)); Serial.print(")");
    Serial.print(" | Streak: "); Serial.print(stateStreak);

    if (!baselineDone && !baselineArmed) Serial.print(" | WAIT_FINGER_STABLE...");
    if (!baselineDone && baselineArmed)  Serial.print(" | BASELINING...");
    if (!fingerPresent) Serial.print(" | NO_FINGER");

    Serial.println();
  }

  // ----- Upload -----
  if (WiFi.status() != WL_CONNECTED) connectWiFi();

  if (millis() - lastUploadMs >= UPLOAD_INTERVAL_MS) {
    lastUploadMs = millis();
    uploadToThingSpeak(smoothedBPM, rmssd, ambTempC, humidity, score);
  }
}
