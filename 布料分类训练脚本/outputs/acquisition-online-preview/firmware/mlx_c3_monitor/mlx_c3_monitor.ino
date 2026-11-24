// 触觉毛笔 · MLX90393 monitor 兼容固件（XIAO ESP32-C3）
// ------------------------------------------------------------------
// XIAO ESP32-C3 固件；优先使用本机实测 GPIO3/GPIO4，保留原引脚候选。
// FQBN: esp32:esp32:XIAO_ESP32C3。
//
// 目的: 让 FYP 的 monitor（docs/monitor.md, src/brushbot/monitor/）能直接用。
//       monitor 串口源要求固件输出 't_us,Bx,By,Bz' CSV @115200（sources.py parse_serial_line）。
//
// 接线（按 ESP32-S3 方案使用 GPIO 编号）:
//          VCC(3.3V) --------------- 3V3
//          GND      --------------- GND
//          SDA      --------------- GPIO1
//          SCL      --------------- GPIO2
//   本机接线 SDA=GPIO3、SCL=GPIO4；优先探测该组合，其余组合作为备用。
//
// 输出 (115200 8N1): 每样本一行 CSV + 每 ~1s 一行 '#' 状态(监控端忽略)
//   t_us,Bx_uT,By_uT,Bz_uT
//   # rate=333.3Hz max|B|=... uT SDA=1 SCL=2 addr=0x0C
// 依赖: Adafruit MLX90393 库 (2.0.5)
// 新增：REBOOT\n -> 确认 -> ESP.restart()；启动后保持原 READY 与四列数据。
// 仅编译（不烧录；路径替换为本 sketch 目录）:
//   arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 <mlx_c3_monitor目录>
// ------------------------------------------------------------------

#include <Wire.h>
#include "Adafruit_MLX90393.h"
#include "serial_commands.h"

SerialCommandParser serialCommands;

// ---- 串口/采样 ----
static const uint32_t SERIAL_BAUD = 115200;   // monitor 串口源默认 115200
static const uint32_t SAMPLE_INTERVAL_US = 3000;  // 定时采样, 333Hz 目标
static const uint8_t MEASURE_DELAY_MS = 2;        // OSR_0/FILTER_0 转换 ~1.5ms
static const uint32_t STATUS_INTERVAL_MS = 1000;  // '#' 状态行周期(不污染 CSV)

// ---- 基线标定（只用于暖机; monitor 端自己取基线, 这里 B0 仅打印参考）----
static const int WARMUP_SAMPLES = 20;
static const int TARE_SAMPLES = 100;

// ---- I2C 引脚：优先本机已确认接线，再尝试原候选 ----
// 首选 GPIO1/GPIO2（及反接）验证 S3 风格接线；XIAO C3 排针没有 GPIO1，
// 因此继续检查 D0/D1、D2/D1、D4/D5 对应的 GPIO 组合。
struct PinPair { uint8_t sda, scl; };
static const PinPair PAIRS[] = {
  {3, 4}, {4, 3},  // Prefer the confirmed wiring; avoid unrelated pin probes on reboot.
  {1, 2}, {2, 1},  // Original S3-style GPIO1/GPIO2 fallback
  {2, 3}, {3, 2},  // XIAO C3 D0/D1
  {6, 7}, {7, 6}   // XIAO C3 D4/D5
};
static const size_t N_PAIRS = sizeof(PAIRS) / sizeof(PAIRS[0]);
static const uint32_t I2C_CLOCK = 400000;
static const uint16_t I2C_TIMEOUT_MS = 10;   // 总线被钳死时快速失败, 不整固件挂死
// 扫描当前实板已见地址，并包含 S3 之前探测到的 0x10。
static const uint8_t CAND_ADDRS[] = {0x0C, 0x0D, 0x10, 0x1E, 0x1F};
static const size_t N_CAND = sizeof(CAND_ADDRS) / sizeof(CAND_ADDRS[0]);

// ---- MLX90393 配置 ----
static const mlx90393_gain_t CFG_GAIN = MLX90393_GAIN_1X;
static const mlx90393_resolution_t CFG_RES = MLX90393_RES_16;
static const mlx90393_oversampling_t CFG_OSR = MLX90393_OSR_0;
static const mlx90393_filter_t CFG_FILTER = MLX90393_FILTER_0;

Adafruit_MLX90393 sensor;
uint8_t mlxAddr = 0;
uint8_t triedAddr = 0xFF;    // 防重复 begin_I2C 触发库崩溃
uint8_t activeSda = 0, activeScl = 0;

unsigned long lastSampleUs = 0;
unsigned long lastStatusMs = 0, windowCount = 0;
float maxMag = 0;

// REBOOT is handled on the acquisition board; magnetic CSV stays unchanged.
void handleSerialCommands() {
  for (unsigned count = 0; count < 64 && Serial.available(); ++count) {
    if (serialCommands.feed(static_cast<char>(Serial.read()))) {
      Serial.println("# fw=mlx_c3 reboot requested");
      Serial.flush();
      delay(200);
      ESP.restart();
      return;
    }
  }
}

// ---- 引脚探测 / 地址识别 ----
uint8_t scanBus() {
  for (int pass = 0; pass < 3; pass++) {
    for (size_t i = 0; i < N_CAND; i++) {
      Wire.beginTransmission(CAND_ADDRS[i]);
      if (Wire.endTransmission() == 0) {
        Serial.print("# +0x");
        if (CAND_ADDRS[i] < 16) Serial.print('0');
        Serial.println(CAND_ADDRS[i], HEX);
        return CAND_ADDRS[i];
      }
    }
  }
  return 0;
}

bool probePair(const PinPair& p, uint8_t& addrOut) {
  Serial.print("# [probe] SDA=");
  Serial.print(p.sda);
  Serial.print(" SCL=");
  Serial.print(p.scl);
  Serial.println(":");
  Wire.end();
  delay(1);
  Wire.begin(p.sda, p.scl);
  Wire.setClock(I2C_CLOCK);
  Wire.setTimeOut(I2C_TIMEOUT_MS);
  delay(5);
  uint8_t a = scanBus();
  if (a) { addrOut = a; activeSda = p.sda; activeScl = p.scl; return true; }
  return false;
}

uint8_t initMlx(uint8_t addr) {
  if (!sensor.begin_I2C(addr, &Wire)) {
    Serial.print("# ERROR: init failed at 0x");
    if (addr < 16) Serial.print('0');
    Serial.println(addr, HEX);
    return 0;
  }
  Wire.setTimeOut(I2C_TIMEOUT_MS);   // begin_I2C 内部会重开 Wire, 超时需重设
  sensor.setGain(CFG_GAIN);
  sensor.setResolution(MLX90393_X, CFG_RES);
  sensor.setResolution(MLX90393_Y, CFG_RES);
  sensor.setResolution(MLX90393_Z, CFG_RES);
  sensor.setOversampling(CFG_OSR);
  sensor.setFilter(CFG_FILTER);
  return addr;
}

bool readFrame(float& x, float& y, float& z) {
  sensor.startSingleMeasurement();
  delay(MEASURE_DELAY_MS);
  return sensor.readMeasurement(&x, &y, &z);
}

void warmupAndTare() {
  Serial.println("CALIBRATING...");
  float x, y, z;
  for (int i = 0; i < WARMUP_SAMPLES; i++) {
    handleSerialCommands();
    if (readFrame(x, y, z)) delay(5);
  }
  // 取 B0 打印参考（monitor 端会自己取, 这里仅给命令行观察用）
  long collected = 0;
  double sx = 0, sy = 0, sz = 0;
  while (collected < TARE_SAMPLES) {
    handleSerialCommands();
    if (readFrame(x, y, z)) { sx += x; sy += y; sz += z; collected++; }
  }
  Serial.print("# tare B0 = (");
  Serial.print(sx / TARE_SAMPLES, 1); Serial.print(", ");
  Serial.print(sy / TARE_SAMPLES, 1); Serial.print(", ");
  Serial.print(sz / TARE_SAMPLES, 1); Serial.println(") uT (monitor 端自带基线)");
  Serial.println("READY");
  Serial.println("# now streaming: t_us,Bx_uT,By_uT,Bz_uT  @115200  (for monitor)");
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0 < 4000)) { handleSerialCommands(); delay(10); }
  delay(300);

  Serial.println();
  Serial.println("# ================================================");
  Serial.println("# MLX90393 monitor-compat @ XIAO ESP32-C3 (S3-style GPIO1/GPIO2)");
  Serial.println("# fw=mlx_c3 build=reboot-v2 target=XIAO_ESP32C3");
  Serial.println("# ================================================");

  for (size_t i = 0; i < N_PAIRS && mlxAddr == 0; i++) {
    handleSerialCommands();
    uint8_t addr = 0;
    if (probePair(PAIRS[i], addr)) {
      Serial.print("# FOUND @ SDA=");
      Serial.print(activeSda);
      Serial.print(" SCL=");
      Serial.print(activeScl);
      Serial.print(" addr=0x");
      if (addr < 16) Serial.print('0');
      Serial.println(addr, HEX);
      if (triedAddr != addr) {
        triedAddr = addr;
        mlxAddr = initMlx(addr);
      }
    }
  }
  if (mlxAddr == 0) {
    Serial.println("# ERROR: 未找到传感器。检查 VCC=3.3V / GND / SDA-SCL 接线。");
    while (true) { handleSerialCommands(); delay(1000); }
  }

  warmupAndTare();
  lastStatusMs = millis();
}

void loop() {
  handleSerialCommands();
  // 定时采样（333Hz 等间隔）
  unsigned long now = micros();
  if (now - lastSampleUs < SAMPLE_INTERVAL_US) return;
  lastSampleUs += SAMPLE_INTERVAL_US;
  if (now - lastSampleUs > SAMPLE_INTERVAL_US) lastSampleUs = now;

  float x, y, z;
  if (!readFrame(x, y, z)) return;   // 失败跳过（monitor 按缺失帧容忍）

  // monitor 要求的 CSV 格式: t_us,Bx,By,Bz
  uint32_t t = micros();
  Serial.print(t); Serial.print(',');
  Serial.print(x, 3); Serial.print(',');
  Serial.print(y, 3); Serial.print(',');
  Serial.println(z, 3);

  // 周期状态行: 人看的, monitor 忽略 '#' 行 (约 333 样本数)
  windowCount++;
  float mag = sqrt(x * x + y * y + z * z);
  if (mag > maxMag) maxMag = mag;
  if (millis() - lastStatusMs >= STATUS_INTERVAL_MS) {
    float rate = windowCount * 1000.0f / (float)(millis() - lastStatusMs);
    Serial.print("# rate=");
    Serial.print(rate, 1);
    Serial.print(" Hz  max|B|=");
    Serial.print(maxMag, 1);
    Serial.print(" uT  SDA=");
    Serial.print(activeSda);
    Serial.print(" SCL=");
    Serial.print(activeScl);
    Serial.print(" addr=0x");
    if (mlxAddr < 16) Serial.print('0');
    Serial.println(mlxAddr, HEX);
    lastStatusMs = millis();
    windowCount = 0;
    maxMag = 0;
  }
}