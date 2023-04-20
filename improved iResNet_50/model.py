import tensorflow as tf
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.set_printoptions(threshold=np.inf)

fashion = tf.keras.datasets.fashion_mnist



class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False, BR_path=False, B0_path=False, R_path=False, B1_path=False, double_res=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path
        self.BR_path = BR_path
        self.B0_path = B0_path
        self.R_path = R_path
        self.B1_path = B1_path
        self.double_res = double_res

        if B0_path:
            self.b0 = BatchNormalization()

        if R_path:
            self.a0 = Activation('relu')

        self.c1 = Conv2D(filters, (1, 1), strides=1, padding='valid', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')  # 激活函数

        self.c2 = Conv2D(filters, (3, 3), strides=strides, padding='same',
                         use_bias=False)  # fielters个卷积核，64个开始每个resnet块*2，line75；卷积核大小3*3；padding使用全零填充；use_bias不使用偏置
        self.b2 = BatchNormalization()  # 批标准化
        self.a2 = Activation('relu')  # 激活函数

        self.c3 = Conv2D(4*filters, (1, 1), strides=1, padding='valid', use_bias=False)

        if B1_path:
            self.b4 = BatchNormalization()

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv2D(4*filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        if BR_path:
            self.b5 = BatchNormalization()
            self.a5 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        a = inputs
        # 将输入通过卷积、BN层、激活层，计算F(x)
        if self.B0_path:
            a = self.b0(a)

        if self.R_path:
            a = self.a0(a)

        a = self.c1(a)
        x = self.b1(a)
        x = self.a1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)

        y = self.c3(x)

        if self.B1_path:
            y = self.b4(y)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        if self.double_res:
            if self.BR_path:
                out = self.b5(y + residual + residual)
                out = self.a5(out)
            else:
                out = y + residual + residual  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        else:
            if self.BR_path:
                out = self.b5(y + residual)
                out = self.a5(out)
            else:
                out = y + residual
        return out


class iResNet50(Model):

    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(iResNet50, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        # 第一个卷积层
        self.c1 = Conv2D(self.out_filters, (7, 7), 2, activation='relu', use_bias=False)
        self.b1 = BatchNormalization()
        self.p1 = MaxPool2D((3, 3), 2, padding='same')

        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样，第一个ResNet块为两条实线跳连
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True, B1_path=True)  # 虚线连接residual_path=True，步长为2
                elif block_id == 0 and layer_id == 0:
                    block = ResnetBlock(self.out_filters, residual_path=True, B1_path=True, double_res=True)
                elif layer_id == block_list[block_id] and block_id == 0:
                    block = ResnetBlock(self.out_filters, residual_path=False, BR_path=True, double_res=True)
                elif layer_id == block_list[block_id]:
                    block = ResnetBlock(self.out_filters, residual_path=False, BR_path=True)
                elif block_id != 0 and layer_id == 1:
                    block = ResnetBlock(self.out_filters, residual_path=False, R_path=True)  # 实线连接residual_path=False
                elif block_id == 0 and layer_id == 1:
                    block = ResnetBlock(self.out_filters, residual_path=False, R_path=True, double_res=True)
                elif block_id != 0 and layer_id != 1 and layer_id != 0:
                    block = ResnetBlock(self.out_filters, residual_path=False, R_path=True, B0_path=True)  # 实线连接residual_path=False
                elif block_id == 0 and layer_id != 1 and layer_id != 0:
                    block = ResnetBlock(self.out_filters, residual_path=False, R_path=True, B0_path=True, double_res=True)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p2 = tf.keras.layers.GlobalAveragePooling2D()  # 平均全局池化
        self.f1 = tf.keras.layers.Dense(5, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())  # 全连接，10个神经元，softmax激活使输出符合概率分布

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.p1(x)
        x = self.blocks(x)
        x = self.p2(x)
        y = self.f1(x)
        return y


model = iResNet50([3, 4, 6, 3])

model_save_path = './checkpoint/iResNet50.ckpt'
model.load_weights(model_save_path)

txt = './image_label/test_jpg.txt'
path = './image_label/test_jpg/'
f = open(txt, 'r')
contents = f.readlines()
f.close()
x, y_ = [], []
i = 0
s = 0
tp = [0, 0, 0, 0, 0]
fp = [0, 0, 0, 0, 0]
fn = [0, 0, 0, 0, 0]
for content in contents:
    value = content.split()
    img_path = path + value[0]
    img = Image.open(img_path)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img_arr = np.array(img.convert('RGB'))
    img_arr = (img_arr / 255.0).astype(np.float16)
    x_predict = img_arr[tf.newaxis, ...]
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)
    if int(pred) == int(value[1]):
        tp[int(pred)] = tp[int(pred)] + 1
        i = i + 1
    else:
        fp[int(pred)] = fp[int(pred)] + 1
        fn[int(value[1])] = fn[int(value[1])] + 1
    s = s + 1
    print(i, s)
r = float(i) / float(s)
print("test accuracy:" + str(r))
print(tp)
print(fp)
print(fn)