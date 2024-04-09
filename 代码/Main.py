import csv
import json
from sklearn.metrics import accuracy_score
from model import *
import matplotlib.pyplot as plt
# from word2vec import calculate

texts = []  # 存储句子对和标签
with open('data/dev.json', 'r', encoding='utf-8') as file:
    for line in file:
        json_line = json.loads(line)
        # 直接从json_line中提取sentence1, sentence2和label，并转换为元组格式
        texts.append((json_line['sentence1'], json_line['sentence2'], int(json_line['label'])))


# 函数用于测试模型性能
def test_model(model_class, threshold=0.5):
    predictions = []
    actuals = []
    num = 0
    for s1, s2, label in texts:
        model = model_class(s1, s2)
        try:
            similarity = model.main()
        except Exception as e:  # 捕获所有异常
            # print(f"计算相似度时发生错误: {e}")  # 打印错误信息
            continue  # 继续下一次循环
        prediction = 1 if similarity >= threshold else 0
        predictions.append(prediction)
        actuals.append(label)  # 转换label值为0或1
        num += 1
        # if num % 400 == 0:
        #     print(f'处理了第{num}条数据')

    return accuracy_score(actuals, predictions)


# 测试每个模型
# model_classes = [calculate.word2vec, CosineSimilarity, JaccardSimilarity, LevenshteinSimilarity, MinHashSimilarity,
#                  SimHashSimilarity,
#                  ]
model_classes = [CosineSimilarity, JaccardSimilarity, LevenshteinSimilarity, MinHashSimilarity, SimHashSimilarity]
threshold = 0.5  # 可以根据需要调整
# 存储模型名称和对应的准确率
model_names = []
accuracies = []

for model_class in model_classes:
    accuracy = test_model(model_class, threshold)
    print(f"{model_class.__name__} Accuracy: {accuracy}")
    model_names.append(model_class.__name__)
    accuracies.append(accuracy)

# 绘制条形图
plt.figure(figsize=[10, 6])  # 设置图表大小
plt.bar(model_names, accuracies, color='skyblue')  # 创建条形图
plt.xlabel('Model')  # 设置x轴标签
plt.ylabel('Accuracy')  # 设置y轴标签
plt.title('Model Accuracy Comparison')  # 设置图表标题
plt.ylim(0, 1)  # 设置y轴的范围
plt.show()  # 显示图表
