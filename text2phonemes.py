# 定义声母、韵母和音调的映射字典
initials_map = {
    'b': 1, 'p': 2, 'm': 3, 'f': 4, 'd': 5, 't': 6, 'n': 7, 'l': 8,
    'g': 9, 'k': 10, 'h': 11, 'j': 12, 'q': 13, 'x': 14,
    'zh': 15, 'ch': 16, 'sh': 17, 'r': 18, 'z': 19, 'c': 20, 's': 21,
    'y': 22, 'w': 23, '': 0  # '' 表示无声母
}

finals_map = {
    'a': 1, 'ai': 2, 'an': 3, 'ang': 4, 'ao': 5, 'e': 6, 'ei': 7, 'en': 8, 'eng': 9, 'er': 10,
    'i': 11, 'ia': 12, 'ian': 13, 'iang': 14, 'iao': 15, 'ie': 16, 'in': 17, 'ing': 18, 'iong': 19, 'iu': 20,
    'o': 21, 'ong': 22, 'ou': 23, 'u': 24, 'ua': 25, 'uai': 26, 'uan': 27, 'uang': 28, 'ue': 29,
    'ui': 30, 'un': 31, 'uo': 32, 'ü': 33, 'üe': 34
}

tones_map = {
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5  # 1-4 为四个音调，5 为轻声
}

punctuation_map = {
    '，': 101, '。': 102, '！': 103, '？': 104, '、': 105, '；': 106, '：': 107,
    '“': 108, '”': 109, '‘': 110, '’': 111, '（': 112, '）': 113, '《': 114, '》': 115
}

import re
import pypinyin
import string
import os
import torch


def text_to_pinyin(text):
    # 使用 pypinyin 库将汉字转换为拼音
    pinyin_list = pypinyin.pinyin(text, style=pypinyin.Style.TONE3, heteronym=False)
    return [item[0] for item in pinyin_list]


def pinyin_to_numeric(pinyin):
    # 使用正则表达式将拼音拆分为声母、韵母和音调
    match = re.match(r'^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw]?)([aeiouvü]+[a-z]*)([1-5]?)$', pinyin)
    if not match:
        raise ValueError(f"无效的拼音输入: {pinyin}")

    initial, final, tone = match.groups()

    initial_num = initials_map.get(initial, 0)
    final_num = finals_map.get(final, 0)
    tone_num = tones_map.get(tone, 5)  # 如果没有音调，则默认为轻声

    return [initial_num, final_num, tone_num]


def process_pinyin_file(filepath, output_dir):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        # 提取文本中的汉字部分
        parts = line.strip().split('|')
        if len(parts) >= 4:
            chinese_text = parts[3]
            pinyin_list = text_to_pinyin(chinese_text)
            numeric_vectors = []
            for char in chinese_text:
                if char in punctuation_map:
                    numeric_vectors.append([punctuation_map[char], 0, 0])  # 添加标点符号的数字表示，补足为长度 3
                elif char in string.ascii_letters or char in string.punctuation:
                    numeric_vectors.append([999, 0, 0])  # 使用特殊标记 999 表示符号和英文，补足为长度 3
                else:
                    pinyin = pypinyin.lazy_pinyin(char, style=pypinyin.Style.TONE3)[0]
                    try:
                        numeric_representation = pinyin_to_numeric(pinyin.strip())
                        numeric_vectors.append(numeric_representation)
                    except ValueError as e:
                        print(e)

            # 保存每行文本的向量表示为 .pt 文件
            filename = os.path.basename(parts[0]).replace('.', '_') + '.pt'
            output_path = os.path.join(output_dir, filename)
            torch_tensor = torch.tensor(numeric_vectors, dtype=torch.int)
            torch.save(torch_tensor, output_path)

def get_phonemes(train_file):
# 示例：处理 .list 文件并保存结果
    input_filepath = f'resources/asr/{train_file}/{train_file}.list'
    output_directory = f'resources/text2phonemes/{train_file}'
    os.makedirs(output_directory, exist_ok=True)
    process_pinyin_file(input_filepath, output_directory)

    # 输出结果示例
    print(f"音素提取完成，结果已保存到 {output_directory} 目录中")

if __name__ == '__main__':
    get_phonemes("shoulinrui.m4a")
#
# tensor([[  0,   1,   5],
#         [101,   0,   0],
#         [ 22,  11,   1],
#         [ 14,  16,   1],
#         [ 15,  22,   4],
#         [ 22,   5,   4],
#         [  5,   6,   5],
#         [  9,  22,   1],
#         [ 21,  11,   1],
#         [ 11,  30,   4],
#         [ 22,  11,   3],
#         [  9,   6,   4],
#         [ 15,  22,   3],
#         [ 15,  24,   3],
#         [  6,  11,   2],
#         [  4,   3,   3],
#         [  4,  24,   4],
#         [ 16,  24,   1],
#         [ 14,  13,   4],
#         [ 19,   2,   4],
#         [ 17,   4,   1],
#         [ 22,   6,   4],
#         [ 12,  20,   4],
#         [ 17,  11,   4],
#         [ 15,   6,   4],
#         [ 22,   4,   4],
#         [  5,   6,   5],
#         [ 12,  16,   2],
#         [  3,  24,   4],
#         [ 15,  22,   1],
#         [101,   0,   0],
#         [  8,  11,   4],
#         [ 18,  24,   2],
#         [  0,   1,   5],
#         [ 14,  18,   1],
#         [  1,   1,   1],
#         [ 10,   6,   4],
#         [102,   0,   0]], dtype=torch.int32)
# torch.Size([38, 3])
