import argparse
import os
import shutil
import cv2
from shared import apply_window_to_img
from shutil import copy
from multiprocessing import Process, Queue, set_start_method

endThread = (-1, -1, -1)
set_start_method('fork')


def run(ID, q):
    print("worker {} starting".format(ID))
    while not endThread:
        if not q.empty():
            item = q.get()
            if item == endThread:
                return
            process(item[0], item[1], item[2])


def process(src, target, filename):
    print(src, target, filename)
    if os.path.splitext(filename)[1] in {".jpg", ".gif", ".png", ".tiff", ".bmp"}:
        try:
            img = cv2.imread(os.path.join(src, filename))
            img = apply_window_to_img(img, win_mode="alg")
            saveimg(target, filename, img)
        except TypeError:
            print("An error occurred when processing", filename)
    else:
        if src != target:
            copyfile(src, target, filename)


def saveimg(path, filename, img):
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(os.path.join(path, filename), img)


def copyfile(src, target, filename):
    if not os.path.exists(target):
        os.makedirs(target)
    filename = os.path.join(src, filename)
    copy(filename, target)


nWorker = 8


def reinforce_Xray_dataset(rootdir, targetdir, report_interval=0, use_multithread=False):
    # make grayscale images distribute in a right way that is much clearer
    
    if use_multithread:
        q = Queue()
        workers = [Process(target=run, args=(i, q)) for i in range(nWorker)]
        for each in workers:
            each.start()

    for root, dirs, files in os.walk(rootdir, topdown=False):
        for name in files:
            relpath = os.path.relpath(root, rootdir)
            target = os.path.join(targetdir, relpath)
            if use_multithread:
                q.put((root, target, name))
            else:
                process(root, target, name)
    if use_multithread:
        for i in range(nWorker):
            q.put(endThread)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input directory")
    parser.add_argument("--output", type=str, help="output directory")
    parser.add_argument("-i", "--interval", type=int, default=0, help="report interval")

    args = parser.parse_args()
    reinforce_Xray_dataset(args.input, args.output, args.interval)
