import torch
from torchvision import models

# 检查是否有MPS支持
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ or no MPS-enabled device is detected.")
else:
    # 加载预训练的ResNet-50模型
    model = models.resnet50(weights='DEFAULT')
    device = torch.device("mps")
    model.to(device)
    print("Model loaded and moved to MPS device.")

# 创建一个随机输入张量
x = torch.randn(1, 3, 224, 224)  # 对于ImageNet，输入大小通常是(3, 224, 224)
x = x.to(device)

# 推理
with torch.no_grad():
    outputs = model(x)
print(outputs)