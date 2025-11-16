face_recognition 是基于 dlib 的人脸识别库

HAAR
HOG
SSD
MTCNN
YuNet
DSFD
RetinaNet-mobienetv1
RetinaNet-resnet50


人脸采集
人脸检测：定位人脸位置，分离出人脸区域（排除背景干扰）
人脸图像预处理：灰度校正、噪声过滤、光纤补偿、灰度变换...
人脸特征提取：将预处理后的人脸图像，转化为计算机可识别的特征向量（提取五官布局、纹理等关键信息）
人脸图像匹配：把提取的特征向量，与数据库中已存储的人脸特征进行比对，计算相似度（欧氏距离、余弦距离）
人脸图像识别：根据匹配结果的相似度阈值，判断是否为同一人，输出识别结论




RetinaNet：提出Focal Loss，解决类别不平衡的检测利器
EfficientDet：提出Compound Scaling策略，高效多尺度检测的极致优化
DenseNet：提出Dense Connection（密集连接），每个卷积层都与前面所有层直接连接，实现特征的充分复用，减少梯度消失，提升模型泛化能力。


