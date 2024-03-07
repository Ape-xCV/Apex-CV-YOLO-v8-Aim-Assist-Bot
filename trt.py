from utils.utils import BaseEngine
import os


class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)


def trt_init(args):
    global engine
    print("Predict on", os.path.normpath(args.model_dir + args.model))
    engine = Predictor(engine_path=args.model_dir + args.model)


def trt(args, img):
    if img is not None:
        boxes, scores, cls_inds = engine.inference(img, iou=args.iou, conf=args.conf, classes=args.classes, end2end=args.end2end)
    return boxes, scores, cls_inds
