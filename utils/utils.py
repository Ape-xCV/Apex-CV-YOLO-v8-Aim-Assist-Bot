import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2


class BaseEngine(object):
    def __init__(self, engine_path):
        self.n_classes = 80
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'')  # initialize TensorRT plugins
        with open(engine_path, 'rb') as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get imgsz
        self.n_classes = engine.get_binding_shape(1)[1] - 4  # get n_classes from "(x1 y1 x2 y2) c1 c2 c3 cn"
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})


    def inference(self, img, iou=0.45, conf=0.25, classes=[], end2end=False):
        cuda_img, ratio = preprocess(img, self.imgsz)
        self.inputs[0]['host'] = np.ravel(cuda_img)
        # copy inputs to GPU
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # copy outputs from GPU
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        data = [out['host'] for out in self.outputs]

        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([
                final_boxes[:num[0]],
                np.array(final_scores)[:num[0]].reshape(-1, 1),
                np.array(final_cls_inds)[:num[0]].reshape(-1, 1)
                ], axis=-1)
        else:
            # predictions = np.reshape(data, (1, 8400, -1), order="F")[0]
            predictions = np.reshape(data, (1, -1, int(4+self.n_classes)), order="F")[0]
            dets = self.postprocess(predictions, ratio, iou, conf)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]  # final_boxes=[0,1,2,3] final_scores=[4] final_cls_inds=[5]
            score_mask = final_scores > conf
            if len(classes) > 0:
                class_mask = np.isin(final_cls_inds, classes)
                mask = np.logical_and(score_mask, class_mask)
            else:
                mask = score_mask
            boxes = final_boxes[mask]
            scores = final_scores[mask]
            cls_inds = final_cls_inds[mask]
        else:
            boxes = np.empty((0,4))
            scores = np.empty((0,self.n_classes))
            cls_inds = np.empty((0,1))
        return boxes, scores, cls_inds


    @staticmethod
    def postprocess(predictions, ratio, iou_thr, conf_thr):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        # dets = multiclass_nms(boxes_xyxy, scores, iou_thr=0.45, conf_thr=0.25)
        dets = multiclass_nms(boxes_xyxy, scores, iou_thr, conf_thr)
        return dets


def nms(boxes, scores, iou_thr):
    """Single class NMS implemented in NumPy"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # get boxes with more conf first

    keep = []
    while order.size > 0:
        i = order[0]  # pick maximum conf box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0, yy2 - yy1 + 1)  # maximum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, iou_thr, conf_thr):
    """Multiclass NMS implemented in NumPy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > conf_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_boxes = boxes[valid_score_mask]
            valid_scores = cls_scores[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, iou_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def preprocess(image, input_size):  # imgsz, size of input images as integer or w,h
    # padded_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)  # HWC (640x640x3)
    # ratio = min(input_size[0]/image.shape[1], input_size[1]/image.shape[0])  # ratio=min, (padded_img) strech the least or shrink the most
    ratio = max(input_size[0]/image.shape[1], input_size[1]/image.shape[0])  # ratio=max, (cropped_img) strech the most or shrink the least
    resized_img = cv2.resize(image, (int(image.shape[1]*ratio), int(image.shape[0]*ratio)), interpolation=cv2.INTER_LINEAR)  # cv2.resize(image, (width, height), interpolation)
    h, w = resized_img.shape[:2]
    min_size = min(h, w)
    cropped_img = resized_img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    cropped_img = cropped_img[:, :, ::-1].transpose(2, 0, 1)  # convert BGR to RGB, permute HWC to CHW (3x640x640)
    cropped_img = np.ascontiguousarray(cropped_img, dtype=np.float32)  # uint8 to float32
    cropped_img /= 255  # convert uint8 0-255 to float32 0.0-1.0
    return cropped_img, ratio
