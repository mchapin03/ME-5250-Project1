#include <Arduino.h>
#include <FastAccelStepper.h>
#include <Wire.h>

//enable is handled by physical switch
#define stepPin 4
#define dirPin  5
#define changeDirPin 6
#define RPM 15
#define SPR 200
#define uStep 8
#define MultiplexerAddress 0x70
#define EncoderAddress 0x36
#define RawAngleMSB 0x0C
#define RawAngleLSB 0x0D
#define SCL_0 15
#define SDA_0 16

bool sendUpdate = true;
long joint_A_prev;
long joint_A_angle;
long joint_A_offset;
long joint_D_prev;
long joint_D_angle;
long joint_D_offset;
long joint_F_prev;
long joint_F_angle;
long joint_F_offset;
static int speed = RPM * ((SPR * uStep)/60);
static int accel = speed;
hw_timer_t *timer = nullptr;
FastAccelStepperEngine  engine = FastAccelStepperEngine();
FastAccelStepper *stepper = NULL;
volatile bool runForward = true;
volatile bool changeDir = false;
volatile bool sendData = false;
bool initalizePosition = true;

void ARDUINO_ISR_ATTR switchFlip() {
    changeDir = true;
}
void ARDUINO_ISR_ATTR onTimer() {
    sendData = true;
}
void  CCW() {
    stepper->runBackward();
    neopixelWrite(48,RGB_BRIGHTNESS,0,0);
    Serial.println("CCW");
}
void  CW() {
    stepper->runForward();
    neopixelWrite(48,0,0,RGB_BRIGHTNESS);
    Serial.println("CW");
}
bool selectMuxChannel(uint8_t bus) {
    if (bus > 7) return false;
    Wire.beginTransmission(MultiplexerAddress);
    Wire.write(1 << bus);
    return Wire.endTransmission() == 0;
}
int readRawAngle(int selector) {
    if (!selectMuxChannel(selector)) return -1;
    Wire.beginTransmission(EncoderAddress);
    Wire.write(RawAngleMSB);
    if (Wire.endTransmission(false) != 0) return -1;
    if (Wire.requestFrom(EncoderAddress, 2) != 2) return -1;
    uint8_t hi = Wire.read();
    uint8_t lo = Wire.read();
    return ((hi & 0x0F) << 8) | lo;
}
void phaseUnwrap(const long current, long* previous, long* position) {
    int delta = current - *previous;

    if (delta > 2048) {
        delta -= 4096;
    } else if (delta < -2048) {
        delta += 4096;
    }

    *position -= delta;
    *previous = current;
}

void setup() {
    Serial.begin(500000);

    //200hz timer
    timer = timerBegin(0, 80, true);
    timerAttachInterrupt(timer, &onTimer, false);
    timerAlarmWrite(timer, 5000, true);
    timerAlarmEnable(timer);
    //stepper motor setup
    engine.init();
    stepper = engine.stepperConnectToPin(stepPin);
    if (stepper) {
        stepper->setAutoEnable(true);
        stepper->setDirectionPin(dirPin);
        stepper->setSpeedInHz(speed);
        stepper->setAcceleration(accel);
    }
    pinMode(changeDirPin, INPUT);
    digitalPinToInterrupt(changeDirPin);
    attachInterrupt(changeDirPin, switchFlip, CHANGE);

    //I2C setup
    Wire.begin(SDA_0, SCL_0, 400000);

    for (uint8_t ch = 0; ch < 8; ch++) {
        if (!selectMuxChannel(ch)) { Serial.println("mux not responding"); break; }
        Serial.printf("ch %u:", ch);
        for (int a = 1; a < 127; a++) {
            Wire.beginTransmission(a);
            if (Wire.endTransmission() == 0) Serial.printf(" 0x%02X", a);
        }
        Serial.println();
    }

}

void loop() {
    //stupid software debounce
    if (changeDir && millis()%3==0) {
        changeDir = false;
        if (digitalRead(changeDirPin) == HIGH) {
            CW();
        } else {
            CCW();
        }
    }

    char str[128];
    //i know i can optimize this code but i dont care to for this purpose
    if (initalizePosition) {
        initalizePosition = false;
        joint_A_prev = readRawAngle(2);
        joint_A_angle = 0;
        joint_D_prev = readRawAngle(1);
        joint_D_angle = 0;
        joint_F_prev = readRawAngle(0);
        joint_F_angle = 0;
    }

    if (sendData) {
        sendData = false;
        int a = readRawAngle(2), d = readRawAngle(1), f = readRawAngle(0);
        if (a >= 0) phaseUnwrap(a, &joint_A_prev, &joint_A_angle);
        if (d >= 0) phaseUnwrap(d, &joint_D_prev, &joint_D_angle);
        if (f >= 0) phaseUnwrap(f, &joint_F_prev, &joint_F_angle);
        snprintf(str,sizeof(str),"Joint A: %li | Joint D: %li | Joint F: %li",joint_A_angle, joint_D_angle, joint_F_angle);
        Serial.println(str);
    }
}