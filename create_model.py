#coding:utf-8
import numpy as np
import os
import glob
import cv2
import shutil
import time
import torch
import torch.nn as nn

from mobilenetv2 import MobileNetV2


# 获取模型实例
model = MobileNetV2()
model.classifier = nn.Sequential(nn.Linear(1280, 8), nn.Sigmoid())
#model.load_state_dict(torch.load("latest.pt"))

img_size = 224
# 生成一个样本供网络前向传播 forward()
example = torch.rand(1, 3, img_size, img_size)

# 使用 torch.jit.trace 生成 torch.jit.ScriptModule 来跟踪
traced_script_module = torch.jit.trace(model, example)

#test_path = "/home/lishundong/Desktop/torch_project/pytorch-regress/data/"
#img_list = glob.glob(test_path + "*.jpg")[:1000]
img_list = ["test.jpg"]

s = time.time()
for i in img_list:
    img_org = cv2.imread(i)
    org_shape = img_org.shape[:-1]
    org_shape = org_shape[::-1]
    # process data
    img = cv2.resize(img_org, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # 1. BGR to RGB; 2. change hxwx3 to 3xhxw
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    inputs = torch.from_numpy(img)
    inputs = inputs.unsqueeze(0)
    output = traced_script_module(inputs)
    print("output:", output)

traced_script_module.save("model_cpp.pt")
print("create c++ model done...")
