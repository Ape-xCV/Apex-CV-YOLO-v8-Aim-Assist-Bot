from ultralytics import YOLO
import os


def predict_init(args):
    global model
    print("Predict on", os.path.normpath(args.model_dir + args.model), "(FP16)" if args.half else "(FP32)")
    model = YOLO(args.model_dir + args.model)


def predict(args,img):
    if img is not None:
        output = model(img, verbose=args.verbose, half=args.half, iou=args.iou, conf=args.conf)
    return output[0]
