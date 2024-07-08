import torch

print(torch.cuda.is_available())  # 检查cuda是否可用
print(torch.version.cuda)  # 查看cuda版本

print(torch.backends.cudnn.is_available())  # 检查cudnn是否可用
print(torch.backends.cudnn.version())  # 查看cudnn版本
