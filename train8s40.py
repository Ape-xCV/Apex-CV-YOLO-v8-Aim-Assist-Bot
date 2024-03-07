from ultralytics import YOLO
from args_ import *
import argparse


# Train YOLOv8s on dataset for 40 epochs
def train(args):
    model = YOLO("yolov8s.pt")
    model.train(data=args.dir + "/datasets/apex/data.yaml", epochs=40, batch=32, workers=2, imgsz=640, val=True)
    # model = YOLO(args.dir + "/best.pt")
    # model.val(data=args.dir + "/datasets/apex/data.yaml", imgsz=640)

# batch=32 GPU_mem=7.3G
# workers=CPU_cores

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args = arg_init(args)
    train(args)
