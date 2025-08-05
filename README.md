> 整个学习过程由大模型问答给予指导！
## 背景

我之前做过客户端、后端，最近对于算法也好奇起来，也着手买了几本书，学习了下。

学习肯定要实践嘛，于是也基于自己已有的资源，尝试训练框架（TensorFlow、PyTorch）的学习和使用，体验整个训练过程。

## 环境搭建

学了conda之后，发现使用conda 配置多个独立环境，非常方便。

安装conda和创建独立环境：
- maoOS M芯片 安装conda
```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

## add source ~/miniconda3/bin/activate to terminal open profile or xxxrc

```

- 独立训练环境
```bash
# 创建独立环境，其中 mps_env 是环境的名称
conda create -n mps_env python=3.10
# 查看环境列表 
conda env list
# 进入独立环境
conda activate mps_env
# 退出独立环境
conda deactivate
```
## TensorFlow 训练模型demo

[环境验证脚本](./verify_gpu_by_tenserflow.py)

[训练模型demo脚本](./ResNet50_verify_tenser_flow.py)

脚本说明：

```python 

import tensorflow as tf  # 导入 TensorFlow 库

# 加载 CIFAR-100 数据集
cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()

# 构建 ResNet50 模型，适配 CIFAR-100 数据集
model = tf.keras.applications.ResNet50(
    include_top=True,        # 保留顶层全连接层
    weights=None,            # 不加载预训练权重，随机初始化
    input_shape=(32, 32, 3), # 输入图片尺寸为 32x32x3
    classes=100,             # CIFAR-100 有 100 个类别
)

# 定义损失函数，这里使用稀疏分类交叉熵
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# 编译模型，指定优化器、损失函数和评估指标
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# 训练模型，设置训练轮数和批次大小
model.fit(x_train, y_train, epochs=5, batch_size=64)
```


## PyTorch 训练模型demo

[distilgpt2 因果语言模型训练demo](./distilgpt2_train_model.py)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# 设置设备，优先使用 Apple Silicon 的 mps（如果可用），否则使用 CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 加载预训练的 distilgpt2 因果语言模型，并移动到指定设备
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
# 加载与模型对应的分词器
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# 加载 wikitext-2 数据集的 1% 作为训练数据，避免资源消耗过大
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")  # 小批量

# 定义分词函数，将文本转换为模型可接受的 token，并做截断和填充
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

# 对整个数据集进行分词处理
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",                # 训练结果保存目录
    per_device_train_batch_size=4,         # 每个设备上的 batch size
    num_train_epochs=1,                    # 训练轮数
    logging_dir="./logs",                  # 日志保存目录
)

# 创建 Trainer 实例，封装模型、训练参数和数据集
trainer = Trainer(
    model=model,                           # 训练的模型
    args=training_args,                    # 训练参数
    train_dataset=tokenized_dataset,       # 训练数据集
)

# 开始训练
trainer.train()
```

[ResNet-50模型训练demo](./ResNet-50_train_demo.py)

```python
import torch
from torchvision import models

# 检查当前PyTorch是否支持MPS（Apple Silicon加速），并给出详细提示
if not torch.backends.mps.is_available():
    # 如果PyTorch没有编译MPS支持
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    # 如果MacOS版本过低或没有检测到MPS设备
    else:
        print("MPS not available because the current MacOS version is not 12.3+ or no MPS-enabled device is detected.")
else:
    # 加载预训练的ResNet-50模型（ImageNet权重）
    model = models.resnet50(weights='DEFAULT')
    # 指定设备为MPS
    device = torch.device("mps")
    # 将模型移动到MPS设备上
    model.to(device)
    print("Model loaded and moved to MPS device.")

# 创建一个随机输入张量，模拟一张3通道224x224的图片（ImageNet标准输入）
x = torch.randn(1, 3, 224, 224)
# 将输入张量移动到与模型相同的设备（MPS）
x = x.to(device)

# 关闭梯度计算，进行推理（前向传播）
with torch.no_grad():
    outputs = model(x)  # 得到模型输出
# 打印输出张量（通常为1000维，代表ImageNet的1000个类别的得分）
print(outputs)
```