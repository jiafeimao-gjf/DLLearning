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