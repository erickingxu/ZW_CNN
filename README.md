# 介绍
大家好，这是我和XiwoWu用C++开发的一个用OpenCV中的Mat作为基础数据结构的的CNN，目标是利用多线程技术，指令集优化技术打造一个在PC端的CPU上运行的快速卷积神经网络，并完全开放源码，欢迎大家STAR或者加入QQ群与我们一起讨论开发CNN或计算机视觉相关的算法问题。
# 依赖
- OpenCV 2.4.13 可以使用tools下面的csv2xml_opencv2.4.13.cpp转换。
- OpenCV 3.4.0  因为OpenCV3存在一个读取csv转Mat的BUG，会导致读取CSV少一行，可以使用OpenCV2.4.13或者使用Python转换。
# 目录
- include 实现神经网络的头文件。
- src 实现神经网络的源码。
- data 保存了Mnist数据集的xml文件，通过tools下的工具从csv文件转换过来。
- tools 实现了csv文件和xml文件转换，训练，测试工具。
- examples 训练Mnist数据集的例子。
- benchmark 一些测试结果记录。
# 训练

- 运行examples下面的train.cpp进行训练和模型保存

# 测试

- 运行examples下面的test.cpp加载模型并测试

# QQ群
- 663852348