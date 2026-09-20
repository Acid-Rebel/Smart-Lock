#ifndef CUSTOM_I2C_DISPLAY_H
#define CUSTOM_I2C_DISPLAY_H

/*
  CustomI2CDisplay.h
  ------------------
  Self-contained software-I2C SH1106 128x64 OLED driver.

  No Wire.h
  No Adafruit_GFX
  No Adafruit_SH110X

  Target:
    ESP32-S3
    SH1106 128x64 I2C OLED
    SDA = GPIO14
    SCL = GPIO21
    I2C address = 0x3C

  This driver implements:
    - software I2C master
    - SH1106 initialization
    - 128x64 1-bit framebuffer
    - pixel drawing
    - lines / rectangles
    - basic 5x7 text
    - rotation 0/1/2/3
    - framebuffer transfer
*/

#include <Arduino.h>

class CustomI2CDisplay {
public:
    static const uint8_t WIDTH  = 128;
    static const uint8_t HEIGHT = 64;

    CustomI2CDisplay(uint8_t sdaPin = 14,
                     uint8_t sclPin = 21,
                     uint8_t address = 0x3C)
        : _sda(sdaPin),
          _scl(sclPin),
          _address(address),
          _rotation(0),
          _cursorX(0),
          _cursorY(0) {
        clearBuffer();
    }

    bool begin() {
        pinMode(_sda, INPUT_PULLUP);
        pinMode(_scl, INPUT_PULLUP);

        // Bus idle
        i2cReleaseSDA();
        i2cReleaseSCL();
        delayMicroseconds(10);

        // Check whether the display acknowledges its address.
        if (!probe()) {
            return false;
        }

        // SH1106 initialization.
        sendCommand(0xAE); // Display OFF

        sendCommand(0xD5); // Display clock
        sendCommand(0x80);

        sendCommand(0xA8); // Multiplex ratio
        sendCommand(0x3F); // 1/64

        sendCommand(0xD3); // Display offset
        sendCommand(0x00);

        sendCommand(0x40); // Display start line = 0

        sendCommand(0xAD); // DC-DC control
        sendCommand(0x8B); // Internal DC-DC ON

        sendCommand(0xA1); // Segment remap
        sendCommand(0xC8); // COM scan direction

        sendCommand(0xDA); // COM pins
        sendCommand(0x12);

        sendCommand(0x81); // Contrast
        sendCommand(0x7F);

        sendCommand(0xD9); // Pre-charge
        sendCommand(0x1F);

        sendCommand(0xDB); // VCOM detect
        sendCommand(0x40);

        sendCommand(0xA4); // Entire display ON follows RAM
        sendCommand(0xA6); // Normal display

        sendCommand(0xAF); // Display ON

        clear();
        display();

        return true;
    }

    void setRotation(uint8_t rotation) {
        _rotation = rotation & 0x03;

        // SH1106 controller orientation is configured separately from
        // the framebuffer coordinate transform.
        switch (_rotation) {
            case 0:
                sendCommand(0xA1);
                sendCommand(0xC0);
                break;

            case 1:
                sendCommand(0xA0);
                sendCommand(0xC0);
                break;

            case 2:
                sendCommand(0xA0);
                sendCommand(0xC8);
                break;

            case 3:
                sendCommand(0xA1);
                sendCommand(0xC8);
                break;
        }
    }

    void clear() {
        clearBuffer();
    }

    void fill(uint8_t color) {
        memset(_buffer, color ? 0xFF : 0x00, sizeof(_buffer));
    }

    void setCursor(int16_t x, int16_t y) {
        _cursorX = x;
        _cursorY = y;
    }

    void drawPixel(int16_t x, int16_t y, bool color = true) {
        transformCoordinates(x, y);

        if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) {
            return;
        }

        const uint16_t index = x + (y / 8) * WIDTH;
        const uint8_t mask = 1 << (y & 7);

        if (color) {
            _buffer[index] |= mask;
        } else {
            _buffer[index] &= ~mask;
        }
    }

    void drawFastHLine(int16_t x, int16_t y, int16_t w, bool color = true) {
        for (int16_t i = 0; i < w; i++) {
            drawPixel(x + i, y, color);
        }
    }

    void drawFastVLine(int16_t x, int16_t y, int16_t h, bool color = true) {
        for (int16_t i = 0; i < h; i++) {
            drawPixel(x, y + i, color);
        }
    }

    void drawRect(int16_t x, int16_t y, int16_t w, int16_t h,
                  bool color = true) {
        if (w <= 0 || h <= 0) return;

        drawFastHLine(x, y, w, color);
        drawFastHLine(x, y + h - 1, w, color);
        drawFastVLine(x, y, h, color);
        drawFastVLine(x + w - 1, y, h, color);
    }

    void fillRect(int16_t x, int16_t y, int16_t w, int16_t h,
                  bool color = true) {
        if (w <= 0 || h <= 0) return;

        for (int16_t yy = 0; yy < h; yy++) {
            drawFastHLine(x, y + yy, w, color);
        }
    }

    void drawBitmap(int16_t x, int16_t y,
                    const uint8_t *bitmap,
                    int16_t w, int16_t h,
                    bool color = true) {
        if (!bitmap) return;

        for (int16_t yy = 0; yy < h; yy++) {
            for (int16_t xx = 0; xx < w; xx++) {
                const uint16_t byteIndex = (yy * w + xx) / 8;
                const uint8_t bit = 0x80 >> ((yy * w + xx) & 7);

                if (bitmap[byteIndex] & bit) {
                    drawPixel(x + xx, y + yy, color);
                }
            }
        }
    }

    // Basic 5x7 ASCII font.
    void print(const char *text, bool color = true) {
        if (!text) return;

        while (*text) {
            char c = *text++;

            if (c == '\n') {
                _cursorX = 0;
                _cursorY += 8;
                continue;
            }

            if (c == '\r') {
                continue;
            }

            drawChar(_cursorX, _cursorY, c, color);

            _cursorX += 6;

            if (_cursorX + 5 >= WIDTH) {
                _cursorX = 0;
                _cursorY += 8;
            }

            if (_cursorY >= HEIGHT) {
                break;
            }
        }
    }

    void drawChar(int16_t x, int16_t y, char c, bool color = true) {
        if (c < 32 || c > 126) c = '?';

        const uint8_t *glyph = font5x7[c - 32];

        for (uint8_t col = 0; col < 5; col++) {
            uint8_t bits = glyph[col];

            for (uint8_t row = 0; row < 7; row++) {
                if (bits & (1 << row)) {
                    drawPixel(x + col, y + row, color);
                }
            }
        }
    }

    // Send the complete framebuffer to the SH1106.
    void display() {
        for (uint8_t page = 0; page < 8; page++) {
            // SH1106 has a 2-column RAM offset compared with SSD1306.
            sendCommand(0xB0 | page);
            sendCommand(0x02); // lower column address
            sendCommand(0x10); // upper column address

            sendData(&_buffer[page * WIDTH], WIDTH);
        }
    }

    bool probe() {
        i2cStart();

        bool ack = i2cWriteByte(static_cast<uint8_t>(_address << 1));

        i2cStop();
        return ack;
    }

private:
    uint8_t _sda;
    uint8_t _scl;
    uint8_t _address;
    uint8_t _rotation;

    int16_t _cursorX;
    int16_t _cursorY;

    // 128 * 64 / 8 = 1024 bytes.
    // The buffer is kept in the object; do not put multiple copies on stack.
    uint8_t _buffer[WIDTH * HEIGHT / 8];

    void clearBuffer() {
        memset(_buffer, 0, sizeof(_buffer));
    }

    // ------------------------------------------------------------
    // Software I2C
    // ------------------------------------------------------------

    void i2cReleaseSDA() {
        pinMode(_sda, INPUT_PULLUP);
    }

    void i2cPullSDA() {
        pinMode(_sda, OUTPUT);
        digitalWrite(_sda, LOW);
    }

    void i2cReleaseSCL() {
        pinMode(_scl, INPUT_PULLUP);
    }

    void i2cPullSCL() {
        pinMode(_scl, OUTPUT);
        digitalWrite(_scl, LOW);
    }

    void i2cDelay() {
        delayMicroseconds(4);
    }

    void i2cStart() {
        i2cReleaseSDA();
        i2cReleaseSCL();
        i2cDelay();

        // START: SDA HIGH -> LOW while SCL HIGH
        i2cPullSDA();
        i2cDelay();

        i2cPullSCL();
        i2cDelay();
    }

    void i2cStop() {
        // Ensure SDA is LOW first.
        i2cPullSDA();
        i2cDelay();

        // SCL HIGH
        i2cReleaseSCL();
        i2cDelay();

        // STOP: SDA LOW -> HIGH while SCL HIGH
        i2cReleaseSDA();
        i2cDelay();
    }

    bool i2cWriteBit(bool bit) {
        if (bit) {
            i2cReleaseSDA();
        } else {
            i2cPullSDA();
        }

        i2cDelay();

        i2cReleaseSCL();
        i2cDelay();

        // Give the clock some time to settle.
        bool clockHigh = digitalRead(_scl);

        i2cPullSCL();
        i2cDelay();

        return clockHigh;
    }

    bool i2cReadBit() {
        i2cReleaseSDA();
        i2cDelay();

        i2cReleaseSCL();
        i2cDelay();

        bool bit = digitalRead(_sda);

        i2cPullSCL();
        i2cDelay();

        return bit;
    }

    bool i2cWriteByte(uint8_t data) {
        for (uint8_t bit = 0; bit < 8; bit++) {
            i2cWriteBit((data & 0x80) != 0);
            data <<= 1;
        }

        // Receiver drives ACK low.
        bool ack = !i2cReadBit();

        return ack;
    }

    // ------------------------------------------------------------
    // SH1106 protocol
    // ------------------------------------------------------------

    void sendCommand(uint8_t command) {
        i2cStart();

        // Address + write bit
        if (!i2cWriteByte(static_cast<uint8_t>(_address << 1))) {
            i2cStop();
            return;
        }

        // SH1106 command control byte
        if (!i2cWriteByte(0x00)) {
            i2cStop();
            return;
        }

        i2cWriteByte(command);

        i2cStop();
    }

    void sendData(const uint8_t *data, uint16_t length) {
        if (!data || length == 0) return;

        i2cStart();

        if (!i2cWriteByte(static_cast<uint8_t>(_address << 1))) {
            i2cStop();
            return;
        }

        // SH1106 data control byte
        if (!i2cWriteByte(0x40)) {
            i2cStop();
            return;
        }

        for (uint16_t i = 0; i < length; i++) {
            i2cWriteByte(data[i]);
        }

        i2cStop();
    }

    void transformCoordinates(int16_t &x, int16_t &y) {
        int16_t tx = x;
        int16_t ty = y;

        switch (_rotation) {
            case 0:
                break;

            case 1:
                x = WIDTH - 1 - ty;
                y = tx;
                break;

            case 2:
                x = WIDTH - 1 - tx;
                y = HEIGHT - 1 - ty;
                break;

            case 3:
                x = ty;
                y = HEIGHT - 1 - tx;
                break;
        }
    }

    // 5x7 font, ASCII 32..126.
    // Each entry contains five columns, least-significant bit at top.
    static const uint8_t font5x7[95][5];
};

// Standard compact 5x7 ASCII font.
const uint8_t CustomI2CDisplay::font5x7[95][5] = {
    {0,0,0,0,0},{0,0,95,0,0},{0,7,0,7,0},{20,127,20,127,20},
    {36,42,127,42,18},{35,19,8,100,98},{54,73,85,34,80},{0,5,3,0,0},
    {0,28,34,65,0},{0,65,34,28,0},{20,8,62,8,20},{8,8,62,8,8},
    {0,80,48,0,0},{8,8,8,8,8},{0,96,96,0,0},{32,16,8,4,2},
    {62,81,73,69,62},{0,66,127,64,0},{66,97,81,73,70},{33,65,69,75,49},
    {24,20,18,127,16},{39,69,69,69,57},{60,74,73,73,48},{1,113,9,5,3},
    {54,73,73,73,54},{6,73,73,41,30},{0,54,54,0,0},{0,86,54,0,0},
    {8,20,34,65,0},{20,20,20,20,20},{0,65,34,20,8},{2,1,81,9,6},
    {50,73,121,65,62},{126,17,17,17,126},{127,73,73,73,54},
    {62,65,65,65,34},{127,65,65,34,28},{127,73,73,73,65},
    {127,9,9,9,1},{62,65,73,73,122},{127,8,8,8,127},
    {0,65,127,65,0},{32,64,65,63,1},{127,8,20,34,65},
    {127,64,64,64,64},{127,2,12,2,127},{127,4,8,16,127},
    {62,65,65,65,62},{127,9,9,9,6},{62,65,81,33,94},{127,9,25,41,70},
    {38,73,73,73,50},{1,1,127,1,1},{63,64,64,64,63},{31,32,64,32,31},
    {63,64,56,64,63},{99,20,8,20,99},{3,4,120,4,3},{97,81,73,69,67},
    {0,127,65,65,0},{2,4,8,16,32},{0,65,65,127,0},{4,2,1,2,4},
    {64,64,64,64,64},{0,3,5,0,0},{32,84,84,84,120},{127,72,68,68,56},
    {56,68,68,68,32},{56,68,68,72,127},{56,84,84,84,24},{8,126,9,1,2},
    {12,82,82,82,62},{127,8,4,4,120},{0,68,125,64,0},{32,64,68,61,0},
    {127,16,40,68,0},{0,65,127,64,0},{124,4,120,4,120},{124,8,4,4,120},
    {56,68,68,68,56},{124,20,20,20,8},{8,20,20,24,124},{124,8,4,4,8},
    {72,84,84,84,36},{4,63,68,64,32},{60,64,64,32,124},{28,32,64,32,28},
    {60,64,48,64,60},{68,40,16,40,68},{12,80,80,80,60},{68,100,84,76,68},
    {0,8,54,65,0},{0,0,127,0,0},{0,65,54,8,0},{2,1,2,4,2}
};

#endif
