本套程序对应的文章名称是《3D Face Reconstruction with the Geometric Guidance of Facial Part Segmentation》
，它是CVPR2024(Highlight)高亮文章，这就说明它很牛逼了。
训练源码在https://github.com/wang-zidu/3DDFA-V3

模型包含人脸检测，106个人脸关键点检测，人脸特征向量提取，一共3个模型
，opencv-dnn都能成功推理运行。在编写这套程序时，最复杂的地方在人脸重建的后处理，这里面涉及到了计算机图形学渲染的知识，
这是重点也是难点。

onnx文件和人脸重建的参数文件在百度云盘，链接: https://pan.baidu.com/s/1W3GkQ02VTOGxfgg7P_jrzQ 提取码: tpfu
