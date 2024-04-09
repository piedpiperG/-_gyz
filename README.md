# 文本相似度实验

### 数据集
 --data文件夹

 来源：从https://tianchi.aliyun.com/dataset/106411?t=1710225696845获得了AFQMC: 蚂蚁金融语义相似度数据集

### 代码结构
#### 方法一：稀疏向量的五种算法

 --Main.py

 运行后测试数据集中预测相似度的准确率

 --model.py

 存放所有方法类的代码

 --单个句子测试相似度.ipynb
 
 提供接口，可以手动输入两个句子，使用不同方法进行测试

 #### 方法二：bert大模型方法

 位于bert文件夹下

 --bert_model

 预训练bert模型，由于文件过大，存放于百度云盘链接链接：https://pan.baidu.com/s/11fe0xgrn5KwvSBiUl9GYEA?pwd=3404 
提取码：3404 
--来自百度网盘超级会员V5的分享，下载后放入其文件夹下即可

 --model

 自己训练得到的模型，由于文件过大，存放于百度云盘链接链接：链接：https://pan.baidu.com/s/1-Rlav78mljrs9WMIBSChoA?pwd=3404 
提取码：3404 
--来自百度网盘超级会员V5的分享下载后放入其文件夹下即可

 --train.py

 训练脚本代码

 --test.py

 测试脚本代码

 #### 方法三：稠密向量方法

 word2vec文件夹下，由于文件过大，存放于百度云盘链接：
链接：https://pan.baidu.com/s/1yu7zo1d9s6OWdkC7MMTKmA?pwd=3404 
提取码：3404 
--来自百度网盘超级会员V5的分享

