import torch
print(torch.cuda.is_available())  # ✅ ต้องได้ True
print(torch.cuda.get_device_name(0))  # ✅ ควรขึ้น "NVIDIA GeForce RTX 3070"
