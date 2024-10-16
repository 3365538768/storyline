import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from audio_emotion_analyse import draw_emotion
from get_word_embedding import get_one_bert
from get_emotion import get_emotion_vec
# 检查是否可以使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentimentDataset(Dataset):
    def __init__(self, bert_feature_dir, sentiment_feature_dir, file_list):
        self.bert_features = []
        self.sentiment_features = []
        self.sequence_lengths = []
        self.file_names = []

        for file_name in file_list:
            bert_path = os.path.join(bert_feature_dir, file_name + '.pt')
            sentiment_path = os.path.join(sentiment_feature_dir, file_name + '.pt')

            if os.path.exists(bert_path) and os.path.exists(sentiment_path):
                # 加载 BERT 特征
                bert_data = torch.load(bert_path,weights_only=False)  # [1, 21128]
                bert_feature = bert_data.squeeze(0).float()  # [21128]
                self.bert_features.append(bert_feature)

                # 加载情感特征
                sentiment_feature = torch.load(sentiment_path,weights_only=False).float()  # [序列长度, 9]
                self.sentiment_features.append(sentiment_feature)

                # 记录序列长度
                self.sequence_lengths.append(sentiment_feature.size(0))

                self.file_names.append(file_name)
            else:
                print(f"文件 {file_name} 的特征不存在，已跳过。")

    def __len__(self):
        return len(self.bert_features)

    def __getitem__(self, idx):
        return self.bert_features[idx], self.sentiment_features[idx], self.sequence_lengths[idx]

def collate_fn(batch):
    bert_features = []
    sentiment_features = []
    lengths = []

    for bert_feature, sentiment_feature, length in batch:
        bert_features.append(bert_feature)
        sentiment_features.append(sentiment_feature)
        lengths.append(length)

    # 将 BERT 特征堆叠，[batch_size, 21128]
    bert_features = torch.stack(bert_features)

    # 对情感特征进行填充
    sentiment_features_padded = nn.utils.rnn.pad_sequence(sentiment_features, batch_first=True, padding_value=0)  # [batch_size, max_seq_len, 9]

    lengths = torch.tensor(lengths)

    return bert_features, sentiment_features_padded, lengths

class AutoregressiveModel(nn.Module):
    def __init__(self, bert_feature_size=21128, sentiment_feature_size=9, hidden_size=256, num_layers=2):
        super(AutoregressiveModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = bert_feature_size + sentiment_feature_size

        # 双层双向LSTM
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, sentiment_feature_size)  # 双向LSTM，输出维度翻倍

    def forward(self, bert_features, sentiment_features, lengths):
        batch_size, max_seq_len, _ = sentiment_features.size()

        # 扩展 BERT 特征到序列长度
        bert_features_expanded = bert_features.unsqueeze(1).repeat(1, max_seq_len, 1)  # [batch_size, max_seq_len, 21128]

        # 添加一个全零的起始情感特征
        zero_sentiment = torch.zeros(batch_size, 1, sentiment_features.size(2)).to(device)
        sentiment_input = torch.cat([zero_sentiment, sentiment_features[:, :-1, :]], dim=1)  # [batch_size, max_seq_len, 9]

        # 拼接输入特征
        inputs = torch.cat([bert_features_expanded, sentiment_input], dim=2)  # [batch_size, max_seq_len, 21137]

        # 使用 pack_padded_sequence 处理可变长度序列
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # LSTM 前向传播
        packed_outputs, _ = self.lstm(packed_inputs)

        # 解包序列
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # 全连接层
        outputs = self.fc(outputs)  # [batch_size, max_seq_len, 9]

        return outputs

def train_model(model, train_dataloader, val_dataloader, num_epochs=10, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for bert_features, sentiment_features, lengths in train_dataloader:
            optimizer.zero_grad()

            # 将数据移动到GPU（如果可用）
            bert_features = bert_features.to(device)
            sentiment_features = sentiment_features.to(device)
            lengths = lengths.to(device)

            # 前向传播
            outputs = model(bert_features, sentiment_features, lengths)

            # 计算损失
            packed_outputs = nn.utils.rnn.pack_padded_sequence(outputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_targets = nn.utils.rnn.pack_padded_sequence(sentiment_features, lengths.cpu(), batch_first=True, enforce_sorted=False)

            loss = criterion(packed_outputs.data, packed_targets.data)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # 验证步骤
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for bert_features, sentiment_features, lengths in val_dataloader:
                bert_features = bert_features.to(device)
                sentiment_features = sentiment_features.to(device)
                lengths = lengths.to(device)

                outputs = model(bert_features, sentiment_features, lengths)

                packed_outputs = nn.utils.rnn.pack_padded_sequence(outputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_targets = nn.utils.rnn.pack_padded_sequence(sentiment_features, lengths.cpu(), batch_first=True, enforce_sorted=False)

                loss = criterion(packed_outputs.data, packed_targets.data)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")

    # 绘制训练和验证损失曲线
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='train loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='validate loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def predict(model, bert_feature, initial_sentiment, max_length):
    model.eval()
    predicted_sentiments = []

    with torch.no_grad():
        bert_feature_expanded = bert_feature.unsqueeze(0).float().to(device)  # [1, 21128]

        hidden = None
        sentiment_input = initial_sentiment.unsqueeze(0).unsqueeze(0).float().to(device)  # [1, 1, 9]

        for _ in range(max_length):
            # 拼接输入特征
            input_feature = torch.cat([bert_feature_expanded, sentiment_input.squeeze(1)], dim=1)  # [1, 21137]
            input_feature = input_feature.unsqueeze(1).float()  # [1, 1, 21137]

            # LSTM 前向传播
            output, hidden = model.lstm(input_feature, hidden)

            # 全连接层
            sentiment_output = model.fc(output)  # [1, 1, 9]

            predicted_sentiments.append(sentiment_output.squeeze(0).cpu())  # [1, 9]

            # 更新输入
            sentiment_input = sentiment_output

    predicted_sentiments = torch.cat(predicted_sentiments, dim=0)  # [max_length, 9]

    return predicted_sentiments

def train(bert_feature_dir,sentiment_feature_dir,pretrained,num_epochs):
    # 获取文件列表
    file_list = [f.split('.pt')[0] for f in os.listdir(bert_feature_dir) if f.endswith('.pt')]

    # 划分数据集
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

    # 创建数据集和数据加载器
    train_dataset = SentimentDataset(bert_feature_dir, sentiment_feature_dir, train_files)
    val_dataset = SentimentDataset(bert_feature_dir, sentiment_feature_dir, val_files)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 定义模型
    model = AutoregressiveModel().to(device)
    if pretrained:
        model.load_state_dict(torch.load("models/text2emo.pth",weights_only=False))
    # 训练模型
    train_model(model, train_dataloader, val_dataloader, num_epochs, learning_rate=1e-3)

    save_path = "models/text2emo.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型参数已保存到: {save_path}")

def main():
    pass


if __name__ == "__main__":
    main()
