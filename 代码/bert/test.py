import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_sentences_from_file(file_path):
    texts = []  # 存储句子对和标签
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_line = json.loads(line)
            # 直接从json_line中提取sentence1, sentence2和label，并转换为元组格式
            texts.append((json_line['sentence1'], json_line['sentence2'], int(json_line['label'])))
    return texts


class MyDataset(Data.Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # 获取索引对应的句子对和标签
        sentence1, sentence2, label = self.texts[index]

        # 使用tokenizer处理句子对
        # 这里我们处理两个句子，所以要传入两个句子作为参数
        encoded_pair = self.tokenizer.encode_plus(
            sentence1, sentence2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        token_ids = encoded_pair['input_ids'].squeeze(0)  # Tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # Binary tensor indicating padded values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(
            0)  # Binary tensor indicating sentence1 and sentence2 tokens

        return token_ids, attn_masks, token_type_ids, torch.tensor(label)


class BertForSentencePairClassification(nn.Module):
    def __init__(self, bert_model, hidden_size=768, num_labels=2):
        super(BertForSentencePairClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(bert_model)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # 通过BERT模型获取句子表示
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output

        # 将表示传递给分类器来获取最终的相似度分数
        logits = self.classifier(pooled_output)

        return logits


def train_model(model, train_loader, val_loader, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num = 0
        for batch in train_loader:
            batch = tuple(b.to(device) for b in batch)
            input_ids, attention_mask, token_type_ids, labels = batch

            # 清除之前的梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num += 1
            if num % 5 == 0:
                print(f'has processed {num} batches')

        torch.save(model.state_dict(), 'model.pth')

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.5f}")

        # 在每个epoch后评估模型
        evaluate(model, val_loader)


def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = tuple(b.to(device) for b in batch)
            input_ids, attention_mask, token_type_ids, labels = batch

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            predictions = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')


def predict_similarity(sentence1, sentence2, model, tokenizer):
    """
    预测两个句子的相似度。

    :param sentence1: 第一个句子
    :param sentence2: 第二个句子
    :param model: 已经加载的模型
    :param tokenizer: 使用的tokenizer
    :return: 预测的相似度得分
    """
    # 使用tokenizer处理输入的句子对
    encoded_pair = tokenizer.encode_plus(
        sentence1, sentence2,
        max_length=128,  # 与之前一致
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoded_pair['input_ids'].to(device)
    attention_mask = encoded_pair['attention_mask'].to(device)
    token_type_ids = encoded_pair['token_type_ids'].to(device)

    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs  # 这里已经正确得到了模型的输出










    # 将输出转换为概率
    probs = torch.softmax(logits, dim=1)  # 应该不会引发错误，因为logits形状为(1, 2)

    # 选择概率最高的类别作为预测结果
    prediction = torch.argmax(probs, dim=1)

    # 获取最高概率值
    max_probability = probs[0][prediction.item()].item()

    return prediction.item(), max_probability


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert_model/bert-base-chinese')
    # 加载保存的模型
    model_path = 'model/model.pth'  # 模型保存路径
    model = BertForSentencePairClassification('bert_model/bert-base-chinese').to(device)
    model.load_state_dict(torch.load(model_path), False)
    model.eval()  # 切换到预测模式

    # 读取数据
    # texts = read_sentences_from_file('./data/dev.json')
    # dataset = MyDataset(texts, tokenizer)
    # loader = DataLoader(dataset, batch_size=32, shuffle=False)  # 可以调整batch_size
    #
    # # 预测并收集结果
    # all_predictions = []
    # all_labels = []
    # with torch.no_grad():
    #     for batch in loader:
    #         batch = tuple(b.to(device) for b in batch)
    #         input_ids, attention_mask, token_type_ids, labels = batch
    #         outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #         batch_predictions = torch.argmax(outputs, dim=1)
    #         all_predictions.extend(batch_predictions.cpu().numpy())
    #         all_labels.extend(labels.cpu().numpy())
    #
    # # 计算准确率
    # accuracy = accuracy_score(all_labels, all_predictions)
    # print(f'Accuracy: {accuracy * 100:.2f}%')

    sentence1 = "太阳从西边升起"
    sentence2 = "太阳从东边升起"
    prediction, probability = predict_similarity(sentence1, sentence2, model, tokenizer)
    print(f"Prediction: {prediction}, Probability: {probability}")
