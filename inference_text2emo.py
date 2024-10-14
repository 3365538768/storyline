from train_text2emo import predict
from train_text2emo import AutoregressiveModel
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=AutoregressiveModel()
model.load_state_dict(torch.load("models/text2emo.pth"))


bert_feature=torch.load("resources/bert_features/shoulinrui.m4a/shoulinrui.m4a_0000513280_0000795840.wav.pt")
initial_sentiment=torch.load("resources/emotion/shoulinrui.m4a/shoulinrui.m4a_0000513280_0000795840.wav.pt")
max_length = 12  # 设置最大预测长度
predicted_sentiments = predict(model, bert_feature, initial_sentiment, max_length)
print(predicted_sentiments)