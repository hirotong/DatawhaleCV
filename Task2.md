# 数据读取
因为返回的图像tensor和label，bbox的tensor的尺寸不一致，应该采用自定义collate-fn的方式，返回图像tensor和其他标签的dict。

# 数据扩增
参考SSD的数据扩增，进行随机反转、放大、缩小、裁剪等。
