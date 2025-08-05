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
    labels=tokenized_dataset["input_ids"],  # 标签为输入的 token ids
)

# 开始训练
trainer.train()