from PIL import ImageGrab
import numpy as np
import cv2
import keyboard
from time import sleep

quit = False
def on_quit(event):
    global quit
    quit = True  

def main():
    keyboard.on_press_key("q", on_quit, suppress=True)
    while not quit:
        img = ImageGrab.grab(bbox=(0,0,400,400))
        img = np.array(img)
        cv2.imshow("test", img)
        cv2.waitKey(10)
        keyboard.write("w")
    cv2.destroyAllWindows()
    keyboard.unhook_all()

if __name__ == "__main__":
    main()