# Apex CV YOLO v8 Aim Assist Bot

- Due to the widespread adoption of controller aim assist on PC gaming...

## Introduction

- It is the aim of this project to provide a quality object detection model.
- Object detection is a technique used in **computer vision** for the identification and localization of objects within an image or a video.
- This project is based on: [Franklin-Zhang0/Yolo-v8-Apex-Aim-assist](https://github.com/Franklin-Zhang0/Yolo-v8-Apex-Aim-assist).
- It is safe to use if you follow the best practices.
- This project is regularly maintained. Better models are made available as more training is created. It is a slowly but surely endeavour.

## Features and Requirements

Features:

* [x] Faster screen capture with _dxshot_
* [x] Faster CPU NMS (Non Maximum Supression) with _NumPy_
* [x] Optional GPU NMS (Non Maximum Supression) with _**TensorRT** efficientNMSPlugin_
* [x] Class targeting `Ally`, `Enemy`, `Tag`
* [x] Humanized mouse control with PID (Proportional-Integral-Derivative)

Requirements:

* [x] NVIDIA RTX Series GPU

## Benchmarks

<details>
<summary>Test system:</summary>

    - OS: Windows 10 Enterprise 1803 (OS build 17134)
    - CPU: Intel Core i7 3770K @ 4.0 GHz
    - GPU: NVIDIA GeForce RTX 2070_8G / 3080_12G
    - RAM: 16G DDR3 @ 2133 MHz
    - Monitor resolution: 1920 x 1080
    - In-game resolution: 1920 x 1080
    - In-game FPS: RTSS async locked @ 72 FPS
</details>

| GPU          | imgsz         | apex_8s.pt | apex_8s.trt | Precision |
| ------------ | ------------- | ---------- | ----------- | --------- |
| RTX 2070_8G  | 640/<br>1080p | 51/35 FPS  | 72/50 FPS   | FP16/32   |
| RTX 3080_12G | 640/<br>1080p | 53 FPS     | 72 FPS      | FP32      |

<details>
<summary>Video settings:</summary>

    - Aspect Ratio               16:9
    - Resolution                 1920 x 1080
    - Brightness                 50%
    - Field of View (FOV)        90
    - FOV Ability Scaling        Enabled
    - Sprint View Shake          Normal
    - V-Sync                     Disabled
    - NVidia Reflex              Enabled+Boost
    - Adaptive Resolution FPS    60
    - Adaptive Supersampling     Disabled
    - Anti-aliasing              TSAA
    - Texture Streaming Budget   Ultra (8GB VRAM)
    - Texture Filtering          Anisotropic 16X
    - Ambient Occlusion Quality  High
    - Sun Shadow Coverage        Low
    - Sun Shadow Detail          Low
    - Spot Shadow Detail         Low
    - Volumetric Lighting        Enabled
    - Dynamic Spot Shadows       Enabled
    - Model Detail               High
    - Effects Detail             High
    - Impact Marks               Disabled
    - Ragdolls                   Low
</details>

### 0. Disclaimer

- This guide has been tested twice, and each time on a fresh install of Windows.
    - Every detail matters. If you are having issues, you are not following the guide.

### 1. Environment set up in Windows

- Version checklist:

    | CUDA   | cuDNN | TensorRT | PyTorch |
    | :----: | :---: | :------: | :-----: |
    | 11.8.0 | 8.6.0 | 8.5.3.1  | 2.0.1   |


- Extract `Apex-CV-YOLO-v8-Aim-Assist-Bot-main.zip` to **C:\TEMP\Ape-xCV**


- Install `Visual Studio 2019 Build Tools`.
    - Download from: [`Microsoft website`](https://aka.ms/vs/16/release/vs_buildtools.exe).
    - On `Individual components` tab:
        - ✅ MSVC v142 - VS 2019 C++ x64/x86 build tools (Latest)
        - ✅ C++ CMake tools for Windows
        - ✅ Windows 10 SDK (10.0.19041.0)
    - ➡️ Install


- Install `CUDA 11.8.0` from: [`NVIDIA website`](https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe).
    - The minimum Visual Studio components were installed in previous step.
        - ✅ I understand, and wish to continue the installation regardless.


- Install `cuDNN 8.6.0`.
    - Download from: [`OneDrive`](https://1drv.ms/u/s!Ap42eSVvSYEggSTk1AM8objdDIbd).
    - OR
    - Register for the [`NVIDIA developer program`](https://developer.nvidia.com/login).
        - Go to the cuDNN download site:[`cuDNN download archive`](https://developer.nvidia.com/rdp/cudnn-archive).
        - Click `Download cuDNN v8.6.0 (October 3rd, 2022), for CUDA 11.x`.
        - Download `Local Installer for Windows (Zip)`.
    - Unzip `cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip`.
    - Copy all three folders (`bin`,`include`,`lib`) and paste <_overwrite_> them into `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`


- Install `Python 3.10.0 (64-bit)` from: [`Python website`](https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe).
    - ❌ Install launcher for all users
    - ✅ Add Python 3.10 to PATH
    - ➡️ Install Now
        - `🔰 Disable path length limit`


- Install `TensorRT`.
    - Go to the TensorRT download site: [NVIDIA TensorRT 8.x Download](https://developer.nvidia.com/nvidia-tensorrt-8x-download).
    - Download `TensorRT 8.5 GA Update 2 for Windows 10 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7 and 11.8 ZIP Package` from: [NVIDIA website](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.5.3/zip/TensorRT-8.5.3.1.Windows10.x86_64.cuda-11.8.cudnn8.6.zip).
    - Extract `TensorRT-8.5.3.1.Windows10.x86_64.cuda-11.8.cudnn8.6.zip` to **C:\TEMP**
    - Press **_[Win+R]_** and enter **cmd** to open a _Command Prompt_. Then input:
    ```shell
    cd /D C:\TEMP\Ape-xCV
    addenv C:\TEMP\TensorRT-8.5.3.1\lib
    ```
    - TensorRT was added to PATH. Close that _Command Prompt_ and open a new one. Then input:
    ```shell
    cd /D C:\TEMP\TensorRT-8.5.3.1\python
    pip install tensorrt-8.5.3.1-cp310-none-win_amd64.whl
    ```


- Install `python requirements`.
``` shell
cd /D C:\TEMP\Ape-xCV
pip install numpy==1.23.1
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2.1 Usage

- Lock your in-game FPS. Give your GPU some slack. It is needed for object detection.
    - Do not use V-Sync to lock your FPS. V-Sync introduces input lag.
        - Use NVIDIA Control Panel.
        - OR
        - Use RTSS.
- Set in-game mouse sensitivity to **3.50**.
    - The PID control (`Kp`, `Ki`, `Kd`) values in `args_.py` come already fine-tuned.
    - If the mouse moves too fast, **EAC will flag your account** and you will be banned on the next ban wave.
        - So, don't mess with the PID. Set in-game mouse sensitivity to **3.50**. Change your mouse DPI instead.


- SHIFT
    - Hold to _aim and fire_ **automatically** non-automatic weapons such as: `Hemlok, Nemesis, Prowler, G7, 3xTake, 30-30, Mastiff, P2020, Wingman`. **Do not use with automatic weapons**.
- LEFT_LOCK
    - Enabled when pressing `'1'` or `'2'`. Disabled when pressing `'G'`.
    - OR
    - Use CURSOR_LEFT to toggle _aim assist_ **while firing** automatic weapons.
- RIGHT_LOCK
    - **This is not recommended**, and you also need to change your ADS from **toggle** to **hold**.
    - Use CURSOR_RIGHT to toggle _aim and fire_ **while scoping**. You will have to quickly follow up with MOUSE1 when using automatic weapons, or else **your firing pattern will be a dead giveaway**.
- HOME
    - 💀 Terminate script.


- Load the Firing Range and give this script a go!
    - 🐵 Run `Ape-xCV.bat`.

### 2.2 Best pratices

- To summarize:
    - ✅ Set in-game mouse sensitivity to **3.50**. Use default PID.
    - ❌ No SHIFT with automatic weapons.
    - ❌ No RIGHT_LOCK.

### 3.1 TensorRT

- If `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\zlibwapi.dll` is missing.
    - Copy `C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.4.2\host-windows-x64\zlib.dll` to **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin** and then rename it to **zlibwapi.dll**


- To export `best_8s.pt` to `best_8s.trt`:
    - Press **_[Win+R]_** and enter **cmd** to open a _Command Prompt_. Then input:
    ```shell
    set CUDA_MODULE_LOADING=LAZY
    cd /D C:\TEMP\Ape-xCV\MODEL
    yolo export model=best_8s.pt format=onnx opset=12
    ```
    - If RTX 20 Series (FP16):
    ```shell
    C:\TEMP\TensorRT-8.5.3.1\bin\trtexec.exe --onnx=best_8s.onnx --saveEngine=best_8s.trt --buildOnly --workspace=7168 --fp16
    ```
    - If RTX 30 Series (FP32):
    ```shell
    C:\TEMP\TensorRT-8.5.3.1\bin\trtexec.exe --onnx=best_8s.onnx --saveEngine=best_8s.trt --buildOnly --workspace=7168
    ```
- Install Notepad++ from: [`Notepad++ website`](https://notepad-plus-plus.org/downloads/).
- Open `C:\TEMP\Ape-xCV\args_.py` with Notepad++.
```shell
def arg_init(args):
    ...
    args.add_argument("--model", type=str,
                    default="/best_8s.pt", help="model path")
```
- Do not change the identation! In --model change `best_8s.pt` to `best_8s.trt`
- Save `args_.py`.
    - 🐵 Run `Ape-xCV.bat`.

### 3.2 TensorRT with GPU NMS

- Cons:
    - ❌ No speed increase.
    - ❌ IoU (Intersection over Union) threshold is hardcoded into engine, ignoring `args_.py`.


- Download: [Linaom1214/TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series).
- Extract `TensorRT-For-YOLO-Series-main.zip` to **C:\TEMP**
- Rename `C:\TEMP\TensorRT-For-YOLO-Series-main` to `C:\TEMP\Linaom1214`
- To export `best_8s.onnx` to `best_8s_e2e.trt`:
    - Press **_[Win+R]_** and enter **cmd** to open a _Command Prompt_. Then input:
    ```shell
    set CUDA_MODULE_LOADING=LAZY
    cd /D C:\TEMP\Linaom1214
    ```
    - If RTX 20 Series (FP16):
    ```shell
    python export.py -o C:/TEMP/Ape-xCV/MODEL/best_8s.onnx -e C:/TEMP/Ape-xCV/MODEL/best_8s_e2e.trt -p fp16 -w 7 --end2end --conf_thres 0.6 --iou_thres 0.8 --v8
    ```
    - If RTX 30 Series (FP32):
    ```shell
    python export.py -o C:/TEMP/Ape-xCV/MODEL/best_8s.onnx -e C:/TEMP/Ape-xCV/MODEL/best_8s_e2e.trt -p fp32 -w 7 --end2end --conf_thres 0.6 --iou_thres 0.8 --v8
    ```
- Open `C:\TEMP\Ape-xCV\args_.py` with Notepad++.
```shell
def arg_init(args):
    ...
    args.add_argument("--model", type=str,
                    default="/best_8s.trt", help="model path")
    args.add_argument("--end2end", type=bool,
                    default=False, help="use TensorRT efficientNMSPlugin")
```
- In --model change `best_8s.trt` to `best_8s_e2e.trt`
- In --end2end change `False` to `True`
- Save `args_.py`.
    - 🐵 Run `Ape-xCV.bat`.

### 4. args_.py

- Open `C:\TEMP\Ape-xCV\args_.py` with Notepad++.
```shell
def arg_init(args):
    ...
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
```
- Until you understand how NMS works, do not change `--iou`.
- `--crop_size` "the portion to detect from the screen". Will be scaled down to 640x640 for input. The default **640/screen_height** is the best value.
- `--draw_boxes` "outline detected target, borderless window". Set to `True` in the Firing Range only.

# ❤️ Sponsor

| PayPal | Litecoin (LTC) |
| :----: | :------------: |
| [![](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate/?hosted_button_id=QCSCYPD8HTATY) | [![](http://api.qrserver.com/v1/create-qr-code/?data=LKLYRUadyHitp5B7aB56yfBYSuG2UnuLEz&size=100x100)](https://quickref.me/emoji)<br>LKLYRUadyHitp5B7aB56yfBYSuG2UnuLEz |

### 5. Train your own model

- Download starter dataset from: [`OneDrive`](https://1drv.ms/u/s!Ap42eSVvSYEggShTC0z5f0AsYCEe).
- Extract **apex.zip** to `C:\TEMP\Ape-xCV\datasets\apex`
- Press **_[Win+R]_** and enter **cmd** to open a _Command Prompt_. Then input:
```shell
cd /D C:\TEMP\Ape-xCV
python train8s40.py
```
- This will **train** your YOLO v**8 s**mall model for **40** epochs with images and labels from `C:\TEMP\Ape-xCV\datasets\apex` and save it to `C:\TEMP\Ape-xCV\runs\detect\train\weights\best.pt`.


- You can add your own images (640x640); and create the labels with: [`developer0hye/Yolo_Label`](https://github.com/developer0hye/Yolo_Label).
    - Copy those into `C:\TEMP\Ape-xCV\SPLIT\input`
    - Press **_[Win+R]_** and enter **cmd** to open a _Command Prompt_. Then input:
    ```shell
    cd /D C:\TEMP\Ape-xCV\SPLIT
    python split.py
    ```
    - Your images and labels are now split into `train` and `valid`.
    - Browse `C:\TEMP\Ape-xCV\datasets\apex` and distribute them accordingly.
