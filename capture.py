import win32api
# import dxcam
import utils.dxshot as dxcam

region = None
camera = None


def capture_init(args):
    global x, y, region, camera
    screen_width = win32api.GetSystemMetrics(0)
    screen_height = win32api.GetSystemMetrics(1)
    crop_height = int(screen_height * args.crop_size)
    crop_width = int(crop_height * (screen_width / screen_height))
    x = (screen_width - crop_height) // 2
    y = (screen_height - crop_height) // 2
    region = (x, y, x + crop_height, y + crop_height)
    camera = dxcam.create(region=region)


def take_shot(args):
    global region, camera
    # img = camera.get_latest_frame()
    img = None
    while img is None:
        img = camera.grab(region=camera.region)
    return img
