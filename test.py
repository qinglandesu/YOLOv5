import torch
print("CUDA 是否可用：", torch.cuda.is_available())
print("PyTorch 版本：", torch.__version__)
print("GPU 型号：", torch.cuda.get_device_name(0))
print("GPU 数量：", torch.cuda.device_count())

# from ultralytics import YOLO
# yolo = YOLO("./yolov5s.pt.pt", task="detect")
# result = yolo(source="./ultralytics/assets/bus.jpg", save=True)