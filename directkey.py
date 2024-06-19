import keyboard
import time

def attack():
    keyboard.press('j')
    time.sleep(0.3)
    keyboard.release('j')

def go_forward():
    keyboard.press('w')
    time.sleep(0.3)
    keyboard.release('w')

def go_back():
    keyboard.press('s')
    time.sleep(0.3)
    keyboard.release('s')

def go_left():
    keyboard.press('a')
    time.sleep(0.3)
    keyboard.release('a')

def go_right():
    keyboard.press('d')
    time.sleep(0.3)
    keyboard.release('d')

def press_esc():
    keyboard.press('esc')
    time.sleep(0.3)
    keyboard.release('esc')
