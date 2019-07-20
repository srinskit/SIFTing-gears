from PIL import ImageGrab
import numpy as np
import cv2
import keyboard
from time import sleep
from simple_pid import PID
import imutils

speed_patch = np.zeros((100, 1200, 3))
map_patch = np.zeros((250, 250, 3))
initial_blur_kernel = (17,17)
template_paths = ['./CV_DS/temp_yellow_1.png', './CV_DS/temp_yellow_1.png']
templates = [cv2.imread(t, cv2.IMREAD_COLOR) for t in template_paths]
templates = [cv2.GaussianBlur(t, initial_blur_kernel, 0) for t in templates]
out = None
key, err, control = "w", 0, 0
frame_size = (480, 158)
fps = 1


quit = True
def toggle_quit(event):
    global quit
    quit = not quit

def dist_point(line, point):
    a, b, c = line
    x, y = tuple(point)
    return abs(a * x + b * y + c) / (a ** 2 + b ** 2) ** 0.5

def consider_line(line, points, dist_threshold, count_threshold):
    count_inliers = 0
    for point in points:
        if dist_point(line, point) <= dist_threshold:
            count_inliers += 1
    if count_inliers >= count_threshold * len(points):
        return True
    else:
        return False


def er_cycle(frame, n):
    kernel = np.ones((5,5),np.uint8)
    frame = cv2.dilate(frame,kernel,iterations = 1)
    # frame = cv2.erode(frame,kernel,iterations = 15)
    # frame = cv2.dilate(frame,kernel,iterations = n-1)
    return frame

def get_err(frame):
    # frame[-speed_patch.shape[0]:, -speed_patch.shape[1]:] = speed_patch
    frame[-250:, :250] = map_patch
    frame = frame[int(frame.shape[0] / 1.6):-130, 400:-400]
    h, w, _ = frame.shape
    frame = cv2.GaussianBlur(frame, initial_blur_kernel, 0)
    frame = imutils.auto_canny(frame)
    frame = er_cycle(frame, 5)
    # cv2.imshow("w",frame)
    def foo(r):
        tmp = np.where(r>0)
        if tmp[0].shape[0] == 0:
            return 0
        else:
            return int(np.mean(tmp))
    centers =  np.apply_along_axis(foo, 1, frame)
    new_frame = np.copy(frame)
    new_frame.fill(0)
    for y in range(centers.shape[0]):
        if centers[y] != 0:
            cv2.circle(new_frame, (centers[y],y), 1, 255, -1)
    kernel = np.ones((5,5),np.uint8)
    new_frame = cv2.dilate(new_frame,kernel,iterations = 1)
    new_frame = cv2.erode(new_frame,kernel,iterations = 1)
    frame = new_frame
    points = list(zip(*np.where(frame > 0)[::-1]))
    # print(len(points))
    if len(points) < 2:
        err = 0
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB) 
    else:
        [vx, vy, x, y] = cv2.fitLine(np.array(points), cv2.DIST_WELSCH, 0, 0.01, 0.01)
        m = 100
        x0, y0 = int(x[0] - m * vx[0]), int(y[0] - m * vy[0])
        x1, y1 = int(x[0] + m * vx[0]), int(y[0] + m * vy[0])
        err_y = h-1
        if y1 == y0:
            err_x = x1
        else:
            err_x = x1+(err_y-y1)*(x0-x1)/(y0-y1)
        err = err_x - w//2
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB) 
        frame = cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 5)
    cv2.putText(frame,'key: %s'%key,(10,h-50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,'err: %d'%err,(10,h-25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    out.write(frame)
    return err


def update(control, err):
    global key
    if -100 <= err <= 100:
        key = "w"
        keyboard.press(key)
        sleep(.03)
        keyboard.release(key)
        sleep(.25)
    elif -500 < err < -100:
        key = "w+a"
        keyboard.press("w+a")
        sleep(.02)
        keyboard.release("w")
        sleep(.2)
        keyboard.release("a")
        sleep(.1)
    elif 100 < err < 500:
        key = "w+d"
        keyboard.press("w+d")
        sleep(.02)
        keyboard.release("w")
        sleep(.2)
        keyboard.release("d")
        sleep(.1)
    else:
        key = "space"
        keyboard.press(key)
        sleep(.3)
        keyboard.release(key)
        sleep(.2)

def main1():  
    global quit, out
    vid = cv2.VideoCapture("./CV_DS/11.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter('.\CV_DS\debug_2.avi', fourcc, fps, frame_size)
    success = 1
    while success:
        success, frame = vid.read()
        if not success:
            break
        get_err(frame)
        ch = cv2.waitKey(10)
        if ch == ord('q'):
            break
    out.release()

def main():  
    global quit, out, control, err
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter('.\CV_DS\debug_1.avi', fourcc, fps, frame_size)
    keyboard.on_press_key("1", toggle_quit, suppress=True)
    pid = PID(1, 0.1, 0.05, setpoint=1)
    pid.sample_time = 1
    print("Press 1 to start")
    while quit:
        sleep(1)
    print("Press 1 to quit")        
    while not quit:
        frame = np.array(ImageGrab.grab())
        err = get_err(frame)
        control = pid(err)
        update(control, err)
    cv2.destroyAllWindows()
    keyboard.unhook_all()
    out.release()


if __name__ == '__main__':
    main()