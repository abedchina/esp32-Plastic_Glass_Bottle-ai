# MicroPython for ESP32: reads the bottle detection from the cloud server
# and sorts each new item with a servo.
#
# The phone/computer camera page (/camera) sends pictures to the server.
# The ESP32 asks the server every half second: GET /result
# (a home ESP32 can't receive calls from the cloud, so it asks instead).

import network
import utime
import urequests
import gc
from machine import Pin, PWM

SSID = 'your-wifi-name'
PASSWORD = 'your-wifi-password'
SERVER = 'https://esp32-plastic-glass-bottle-ai.onrender.com'
DEVICE_TOKEN = 'your-device-token'   # same password as on the camera page

POLL_MS = 500
STABLE_POLLS = 3   # same material 3 times in a row (~1.5 s) = a real item
EMPTY_POLLS = 2    # zone must be empty twice before the next item counts

led = Pin(2, Pin.OUT)            # built-in LED
servo = PWM(Pin(18), freq=50)    # servo signal wire on GPIO 18


def servo_angle(angle):
    # 0.5 ms .. 2.5 ms pulse in a 20 ms period
    servo.duty_u16(int(1638 + angle / 180 * (8192 - 1638)))


def handle_arrival(label, confidence, count):
    print('========================')
    print('Item #', count)
    print('AI class:', label)
    print('Confidence:', confidence)
    led.on()
    if label == 'plastic':
        print('PLASTIC -> left bin')
        servo_angle(30)
    elif label == 'glass':
        print('GLASS -> right bin')
        servo_angle(150)
    utime.sleep_ms(1000)
    servo_angle(90)              # back to the middle
    led.off()
    print('========================')


def wifi_connect():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print('Connecting to WiFi...')
        wlan.connect(SSID, PASSWORD)
        started = utime.ticks_ms()
        while not wlan.isconnected():
            if utime.ticks_diff(utime.ticks_ms(), started) > 20000:
                raise OSError('WiFi connection timeout')
            utime.sleep_ms(250)
    print('ESP32 IP:', wlan.ifconfig()[0])
    return wlan


def get_result():
    response = None
    try:
        response = urequests.get(
            SERVER + '/result',
            headers={'X-Device-Token': DEVICE_TOKEN}
        )
        if response.status_code == 401:
            print('Wrong DEVICE_TOKEN')
            return None
        if response.status_code != 200:
            # 502/503 while the free Render server wakes up
            print('Server not ready:', response.status_code)
            return None
        return response.json()
    except Exception as error:
        print('Request error:', error)
        return None
    finally:
        if response is not None:
            response.close()
        gc.collect()


def main():
    wlan = wifi_connect()
    servo_angle(90)
    candidate = None
    same = 0
    empty = 0
    armed = True
    total = 0
    print('Waiting for detections...')

    while True:
        if not wlan.isconnected():
            wlan = wifi_connect()

        result = get_result()
        if result is not None:
            label = result.get('class', 'none') if result.get('detected') else 'none'

            if label not in ('plastic', 'glass'):
                candidate = None
                same = 0
                empty += 1
                if empty >= EMPTY_POLLS:
                    armed = True
            else:
                empty = 0
                if label == candidate:
                    same += 1
                else:
                    candidate = label
                    same = 1
                if armed and same >= STABLE_POLLS:
                    armed = False
                    total += 1
                    handle_arrival(label, result.get('confidence', 0), total)

        utime.sleep_ms(POLL_MS)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        servo.deinit()
        print('Stopped')
