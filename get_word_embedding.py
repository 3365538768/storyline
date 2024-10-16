from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import os

def get_bert(file_name):
    # 1. 加载预训练的分词器和模型
    local_model_path = "models/bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path,trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(local_model_path,trust_remote_code=True)
    model.eval()  # 设置为评估模式


    list_file_path = f"resources/asr/{file_name}/{file_name}.list"  # 替换为您的 .list 文件路径
    output_dir = f"resources/bert_features/{file_name}"  # 定义保存 .pt 文件的目录
    os.makedirs(output_dir, exist_ok=True)  # 创建保存目录，如果 exist_ok=True 且目录已存在，os.makedirs() 不会报错，并会继续执行后续代码。

    with open(list_file_path, "r", encoding="utf-8") as file:
        for line in file:
            # 分割文件路径、文件名、语言、文本
            parts = line.strip().split("|")
            if len(parts) != 4:
                print(f"Skipping malformed line: {line}")
                continue

            # 提取文件路径和文本
            full_audio_path = parts[0]
            audio_file_name = os.path.basename(full_audio_path)
            text = parts[3]

            # 使用 BERT 分词器对文本进行编码
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            # 获取 BERT 模型输出
            with torch.no_grad():
                outputs = model(**inputs)

            token_embeddings=outputs.logits

            # 提取 [CLS] 向量作为特征
            cls_embedding = token_embeddings[:, 0, :]  # [1, hidden_size]

            # 保存为 .pt 文件
            output_file_path = os.path.join(output_dir, f"{audio_file_name}.pt")
            torch.save(cls_embedding, output_file_path)  #torch.Size([1, 21128])
    print("bert提取完毕")

def get_one_bert(text):
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
    model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-chinese")
    model.eval()  # 设置为评估模式
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.logits

    # 提取 [CLS] 向量作为特征
    cls_embedding = token_embeddings[:, 0, :]  # [1, hidden_size]
    return cls_embedding

if __name__ == "__main__":
    get_bert("shoulinrui.m4a")