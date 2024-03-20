from pynput import keyboard, mouse
from listen import listen_k_press, listen_k_release, listen_m_click, listen_init, get_D_L, mouse_redirection, move_mouse
import argparse
from args_ import *
from threading import Thread
from multiprocessing import Process, Pipe, Value
from capture import *
import time
import numpy as np
from draw import show_target
import capture
import listen
##import cv2
##from datetime import datetime

global detecting, listening


def listeners():
    key_listener = keyboard.Listener(on_press=listen_k_press, on_release=listen_k_release)
    key_listener.start()

    mouse_listener = mouse.Listener(on_click=listen_m_click)
    mouse_listener.start()

    key_listener.join()


if __name__ == "__main__":
    os.system("")
    print("\033[01;04;31m" + "A" + "\033[32m" + "N" + "\033[33m" + "S" + "\033[34m" + "I" + "\033[00m" + " enabled")
    # create an arg set
    listening = True
    print("listener start")

    args = argparse.ArgumentParser()
    args = arg_init(args)
    listen_init(args)

    thread_1 = Thread(
        target=listeners,
        args=(),
    )
    thread_1.start()
    print(thread_1)

    capture_init(args)
    if args.model[-3:] == ".pt":
        from predict import *

        predict_init(args)
    else:
        from trt import trt_init, trt

        trt_init(args)
    print("main start")
    time_start = time.time()
    count = 0
    time_capture_total = 0
    while listening:

        detecting, listening = get_D_L()
        # take a screenshot
        time_shot = time.time()
        img = take_shot(args)
        time_capture = time.time()
        time_capture_total += time_capture - time_shot
        # print("shot time: ", time.time() - time_shot)
        # predict the image
        time.sleep(args.wait)
        if args.model[-3:] == ".pt":
            predict_output = predict(args, img)
            # print(predict_output.boxes.conf, predict_output.boxes.cls)
            boxes = predict_output.boxes
            boxes = boxes[boxes[:].cls == args.target_index].cpu().xyxy.numpy()
        else:
            boxes, scores, cls_inds = trt(args, img)
            # print(scores, cls_inds)
        time_predict = time.time()
        # print("predict time: ", time.time() - time_capture)
        if detecting:
            if boxes.size != 0:
                if args.draw_boxes:
                    for i in range(0, int(boxes.size/4)):
                        show_target([int(boxes[i,0])+capture.x, int(boxes[i,1])+capture.y, int(boxes[i,2])+capture.x, int(boxes[i,3])+capture.y])
##            elif args.save_img and listen.shift_pressed:
##                cv2.imwrite('screenshots/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f') + '.png', img)
            # print(boxes)
            mouse_redirection(args, boxes)
            move_mouse(args)
        # print("post-process time: ", time.time() - time_predict)
        # print("total time: ", time.time() - time_shot)
        count += 1

        if (count % 100 == 0):
            time_per_100frame = time.time() - time_start
            time_start = time.time()
            print("Screenshot fps: ", count / time_capture_total)
            print("fps: ", count / time_per_100frame)
            interval = time_per_100frame / count
            print("interval: ", interval)
            print("[LEFT_LOCK]" if listen.left_lock else "[         ]", "[\033[30;41mRIGHT_LOCK\033[00m]" if listen.right_lock else "[          ]")
            count = 0
            time_capture_total = 0

    print("main stop")
