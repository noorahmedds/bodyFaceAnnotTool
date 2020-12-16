
NEXT_IMAGE = "n"
PREVIOUS_IMAGE = "p"

ANNOTATE_MODE = "m"

CANCEL_ANNOTATION = "c"

SAVE_KEY = "s"

import time

class KeyTracker():
    key = ''
    last_press_time = 0
    last_release_time = 0

    def __init__(self):
        self.pressed = False

    def track(self, key):
        self.key = key

    def is_pressed(self):
        return time.time() - self.last_press_time < .1

    def report_key_press(self, event):
        # print("ALT PRESSED")
        self.pressed = True
        if event.keysym == self.key:
            if not self.is_pressed():
                on_key_press(event)
            self.last_press_time = time.time()

    def report_key_release(self, event):
        self.pressed = False
        # print("ALT RELEASED")
        if event.keysym == self.key:
            timer = threading.Timer(.1, self.report_key_release_callback, args=[event])
            timer.start()

    def report_key_release_callback(self, event):
        if not self.is_pressed():
            on_key_release(event)
        self.last_release_time = time.time()