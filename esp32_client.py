# MicroPython for ESP32: gets the bottle detection from the cloud server.
#
# The phone/computer camera page (/camera) sends pictures to the server.
# The server counts ONE event per bottle: when a bottle enters the
# SORTING ZONE it is counted once, and the next bottle counts only after
# this one has left the zone (good for a moving conveyor).
# The ESP32 asks the server every 0.3 s: GET /result
#
# LED on GPIO 2:  glass   = 3 fast blinks
#                 plastic = 2 fast blinks

import network
import utime
import urequests
import gc
from machine import Pin

SSID = 'your-wifi-name'
PASSWORD = 'your-wifi-password'
SERVER = 'https://esp32-plastic-glass-bottle-ai.onrender.com'
DEVICE_TOKEN = 'your-device-token'   # same password as on the camera page

POLL_MS = 300
BLINK_MS = 120

led = Pin(2, Pin.OUT)
led.off()


def blink(times):
    for _ in range(times):
        led.on()
        utime.sleep_ms(BLINK_MS)
        led.off()
        utime.sleep_ms(BLINK_MS)


def handle_new_bottle(event):
    label = event['class']
    print('========================')
    print('NEW BOTTLE #', event['id'])
    print('AI class:', label.upper())
    print('Confidence: {}%'.format(round(event['confidence'] * 100)))
    if label == 'glass':
        print('GLASS -> LED 3 blinks')
        blink(3)
        # Add your glass motor/servo action here.
    elif label == 'plastic':
        print('PLASTIC -> LED 2 blinks')
        blink(2)
        # Add your plastic motor/servo action here.
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
    last_id = None          # last bottle the ESP32 already handled
    last_state = None       # what was printed last, to avoid repeating lines
    print('Waiting for bottles...')

    while True:
        if not wlan.isconnected():
            wlan = wifi_connect()

        result = get_result()
        if result is not None:
            event = result.get('event') or {}
            event_id = event.get('id', 0)

            if last_id is None:
                # first answer: handle only a bottle from the last 3 seconds
                age = event.get('age')
                fresh = event_id and age is not None and age < 3
                last_id = event_id - 1 if fresh else event_id

            if event_id and event_id != last_id:
                last_id = event_id
                handle_new_bottle(event)

            # show what the camera sees now (printed only when it changes)
            if result.get('detected'):
                state = 'In zone: ' + result['class']
                if not result.get('armed', True):
                    state += ' (counted, waiting to leave)'
            else:
                state = 'Zone empty'
            if state != last_state:
                print(state)
                last_state = state

        utime.sleep_ms(POLL_MS)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        led.off()
        print('Stopped')
