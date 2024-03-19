import os
import win32api


def arg_init(args):
    dirpath = os.path.dirname(os.path.realpath(__file__))
    args.add_argument("--dir", type=str, default=dirpath, help="root dir path")
    args.add_argument("--save_dir", type=str,
                    default=dirpath + "/predict", help="save dir")
    args.add_argument("--model_dir", type=str,
                    default=dirpath + "/model", help="model dir")
    args.add_argument("--model", type=str,
                    default="/best_8s.pt", help="model path")
    args.add_argument("--end2end", type=bool,
                    default=False, help="use TensorRT efficientNMSPlugin")
    args.add_argument("--classes", type=int,
                    default=[1,2], help="classes to be detected TensorRT(.trt); can be expanded but needs to be an array. "
                    "0 represents 'Ally', "
                    "1 represents 'Enemy', "
                    "2 represents 'Tag'... "
                    "Change default accordingly if your dataset changes")
    args.add_argument("--target_index", type=int,
                    default=1, help="class to be targeted PyTorch(.pt)")
    args.add_argument("--half", type=bool,
                    default=True, help="use FP16 to predict PyTorch(.pt)")
    args.add_argument("--iou", type=float,
                    default=0.8, help="predict intersection over union")  # 0.8 is recommended
    args.add_argument("--conf", type=float,
                    default=0.6, help="predict confidence")  # 0.6+ is recommended
    screen_height = win32api.GetSystemMetrics(1)
    args.add_argument("--crop_size", type=float,
                    default=640/screen_height, help="the portion to detect from the screen. 1/3 for 1440P or 1/2 for 1080P, imgsz/screen_height=direct")
    args.add_argument("--wait", type=float, default=0, help="wait time")
    args.add_argument("--verbose", type=bool, default=False, help="predict verbose")
    args.add_argument("--draw_boxes", type=bool,
                    default=False, help="outline detected target, borderless window")
    args.add_argument("--caps_lock", type=bool,
                    default=False, help="use CAPS_LOCK as LEFT_LOCK")
    # args.add_argument("--mouse_speed", type=float,
    #                 default=3.50, help="mouse speed (mouse sensitivity in the game)")

    # PID args
    args.add_argument("--pid", type=bool, default=True, help="use proportional–integral–derivative control")
    args.add_argument("--Kp", type=float, default=0.3, help="Kp")  # proporcional to distance 0.4 nimble 0.1 slack
    args.add_argument("--Ki", type=float, default=0.04, help="Ki")  # integral accumulator 0.04 explosive 0.01 composed
    args.add_argument("--Kd", type=float, default=0.3, help="Kd")  # derivative absorber 0.4 stiff 0.1 soft

    args = args.parse_args(args=[])
    return args
