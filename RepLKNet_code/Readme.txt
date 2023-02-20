各个函数的作用：
imgae_processing.py  包含对图像进行白边裁剪、分辨率调整、从中心切割、转为tensor等操作
RepLKNet.py               包含自编类实现RepLKNet
test.py                         加载训练保存模型，在测试集上进行测试评估
train.py                        训练RepLKNet模型
utils.py                         包含用热力图现实测试结果的函数


代码的使用：
训练：在该目录下输入python train.py
测试：在该目录下输入python test.py