from audio_emotion_analyse import get_all_emotion, draw_emotion, save_list
import librosa
import os
import shutil
import soundfile as sf
import numpy as np
import torch
from tools.new_slice_audio import slice
from subprocess import Popen
from tools.config import python_exec
from get_word_embedding import get_bert,get_one_bert
from get_emotion import  get_emotion_vec
from train_text2emo import train,AutoregressiveModel,predict
from get_hubert import wav2hubert
from text2phonemes import get_phonemes

def slice_wav(train_file_name):
    slice_inp_path = "resources/train/" + train_file_name
    slice_opt_root = "resources/slice/" + train_file_name
    threshold = -34
    min_length = 4000
    min_interval = 300
    hop_size = 10
    max_sil_kept = 500
    _max = 0.9
    alpha = 0.25
    n_process = 4
    slice(slice_inp_path, slice_opt_root, threshold, min_length, min_interval, hop_size,
                                 max_sil_kept, _max, alpha, 0,n_process)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>切割结束\n")


def build_asr_command(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision, python_exec):
    from tools.asr.config import asr_dict

    # 构建 ASR 命令
    cmd = f'"{python_exec}" tools/asr/{asr_dict[asr_model]["path"]}'
    cmd += f' -i "{asr_inp_dir}"'
    cmd += f' -o "{asr_opt_dir}"'
    cmd += f' -s {asr_model_size}'
    cmd += f' -l {asr_lang}'
    cmd += f" -p {asr_precision}"

    return cmd


def open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision, python_exec="python"):
    # 构建 ASR 命令
    cmd = build_asr_command(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision, python_exec)

    # 输出文件路径
    output_file_name = os.path.basename(asr_inp_dir)
    output_folder = asr_opt_dir or "output/asr_opt"
    output_file_path = os.path.abspath(f'{output_folder}/{output_file_name}.list')

    # 启动 ASR 任务
    yield f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>ASR任务开启"

    p_asr = Popen(cmd, shell=True)
    p_asr.wait()

    # 完成 ASR 任务
    yield f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>ASR任务完成"
def asr_slice(train_file_name):
    asr_inp_dir = "resources/slice/" + train_file_name
    asr_opt_dir = "resources/asr/"+train_file_name
    asr_model = "达摩 ASR (中文)"
    asr_size = "large"
    asr_lang = "zh"
    asr_precision = "float32"

    asr_generator = open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_size, asr_lang, asr_precision)
    for message in asr_generator:
        print(message)
    print(">>>>>>asr结束\n")


def slice_for_emotion(audio_name, window, hop):  # 窗口切片音频
    # 加载音频文件
    audio, sr = librosa.load(audio_name)

    # 设置切片时长（以秒为单位）
    window_length = window * sr  # 窗口长度
    hop_length = hop * sr  # 移动步幅

    # 计算补全长度（半个窗口长度）
    pad_length = window_length // 2

    # 在音频的两端进行补全
    audio = np.pad(audio, (pad_length, pad_length), mode='constant')  # 将两边补全，这样不论窗口大小输出向量数都相同

    # 获得文件去除后缀的名字
    filename = audio_name
    new_filename = os.path.basename(filename)

    # 创建保存切片的文件夹
    path = f"resources/temp_emo_slice/{new_filename}/{window}"
    if not os.path.exists(path):
        os.makedirs(path)

    for start in range(0, len(audio) - window_length + 1, hop_length):
        slice_audio = audio[start:start + window_length]
        sf.write(f"resources/temp_emo_slice/{new_filename}/{window}/slice_{start // hop_length}.wav", slice_audio, sr)

    print(f">>>>文件({new_filename})切片成功,保存路径({path})")
    return path


def weighted_sum(vectors, weights):
    # 确保权值是一个一维数组
    weights = np.array(weights)

    # 计算加权求和
    result = np.tensordot(weights, vectors, axes=(0, 0))
    result = np.round(result, decimals=2)
    return result

def get_average_emo(filename):
    emo = []
    for window in range(4, 20, 4):
        emo_slice_path = slice_for_emotion(f"resources/train/{filename}", window, 1)
        labels, audio_emotion = get_all_emotion(emo_slice_path)  # 获取每个切片的情感
        # save_list(labels, f"resources/emotion_data/{filename}/labels_{window}.txt")
        # save_list(audio_emotion, f"resources/emotion_data/{filename}/audio_emotion_{window}.txt")
        # draw_emotion(audio_emotion, labels, filename, window)  # 绘制情感图-
        emo.append(audio_emotion)
        shutil.rmtree(emo_slice_path)
        print(f">>>>>>>>>>>>>>>>>>>>{window}窗口结束")

    weights = [0.35, 0.3, 0.2, 0.15]
    result = weighted_sum(emo, weights)
    # save_list(result, f"resources/emotion_data/{filename}/audio_emotion_average.txt")
    draw_emotion(result, labels, filename, "average")  # 绘制情感图
    return result


def read_pt(path):
    file = torch.load(path)
    print(file)
    print(file.shape)
    return file


def get_audio_durations(folder_path):
    durations = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav') or filename.endswith('.mp3'):  # 可以根据需要添加其他音频格式
            file_path = os.path.join(folder_path, filename)
            duration = librosa.get_duration(filename=file_path)  # 获取音频持续时间
            durations[filename] = duration
    return durations


def read_intervals_from_txt(file_path):  # 读取每个slice的音频范围，返回一个元组(start,end)，以便寻找对应的感情区域
    intervals = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去除空白字符和换行符
            line = line.strip()
            if line:  # 确保行不是空的
                # 分割字符串并转换为浮点数
                start, end = map(float, line.split('/'))
                intervals.append((start, end))  # 将元组添加到列表中

    return intervals



def interval_to_emo(intervals, average_emo, filename):
    # 创建保存 .pt 文件的目录
    save_dir = f"resources/emotion/{filename}"
    os.makedirs(save_dir, exist_ok=True)

    number_of_vectors = len(average_emo)

    # 计算每个向量对应的时间长度（假设向量均匀分布在音频上）
    duration_per_vector = 1

    # 为每个向量分配时间戳（这里使用向量时间区间的起始点）
    vector_times = [i * duration_per_vector for i in range(number_of_vectors)]

    # 为每个区间找到对应的向量索引
    interval_vectors = []

    for interval in intervals:
        start_time, end_time = interval
        # 找到所有时间戳在当前区间内的向量索引
        indices = [i for i, t in enumerate(vector_times) if t >= start_time and t < end_time]
        interval_vectors.append(indices)

    inp_text = f"resources/asr/{filename}/{filename}.list"
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    for idx, indices in enumerate(interval_vectors):
        wav_name, spk_name, language, text = lines[idx].split("|")
        wav_name = os.path.basename(wav_name)
        corresponding_vectors = [average_emo[i] for i in indices]
        if corresponding_vectors:
            # 将对应的向量列表转换为张量
            corresponding_array = np.array(corresponding_vectors)
            corresponding_tensor = torch.tensor(corresponding_array)
            # 定义保存路径
            save_path = os.path.join(save_dir, f"{wav_name}.pt")
            # 保存张量为 .pt 文件
            torch.save(corresponding_tensor, save_path)
            print(f"已保存区间 {idx} 的向量到 {save_path}")
        else:
            print(f"区间 {idx} 中没有找到对应的向量")

    return 0

def clear_slice(train_filename):
    path=f"resources/slice/{train_filename}"
    shutil.rmtree(path)

def prepare(train_filename):
    slice_wav(train_filename)  # 切割音频（原GPT切割）
    get_slice_hubert(train_filename)
    asr_slice(train_filename)  # 切割音频asr识别
    slice_log_path = f"resources/slice/intervals/{train_filename}/slice_log.txt"
    intervals = read_intervals_from_txt(slice_log_path)
    clear_slice(train_filename)
    average_emo = get_average_emo(train_filename)  # 获得平均情感向量 (93,9) 93和步幅、音频长度有关，步幅为1s则就这里就表示93s每一秒一个取样，9为9种类型情感
    interval_to_emo(intervals, average_emo, train_filename)  # 得到每段音频对应的情感特征，保存在logs/emotion里
    get_bert(train_filename)

def model1_train(train_filename,pretrained,num_epochs):
    bert_feature_dir = "resources/bert_features/" + train_filename
    sentiment_feature_dir = "resources/emotion/" + train_filename
    train(bert_feature_dir, sentiment_feature_dir, pretrained, num_epochs)

def model1_infer(text,prompt_audio_path,max_length):
    model = AutoregressiveModel().to(device)
    model.load_state_dict(torch.load("models/text2emo.pth",weights_only=False))

    bert_feature = get_one_bert(text)[0].to(device)
    print(bert_feature)
    labels, initial_sentiment = get_emotion_vec(prompt_audio_path)
    initial_sentiment = torch.tensor(initial_sentiment).to(device)
    print(initial_sentiment)
    predict_emotion = predict(model, bert_feature, initial_sentiment, max_length)
    print(predict_emotion)
    draw_emotion(predict_emotion, labels, "predict", 0)


def get_slice_hubert(train_file_name):
    folder_path = 'resources/slice/' + train_file_name
    # 获取文件夹下所有非 .txt 文件的名字
    file_names = [file for file in os.listdir(folder_path)
                  if os.path.isfile(os.path.join(folder_path, file)) and not file.endswith('.txt')]

    # 逐个处理文件
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        wav2hubert(file_path, train_file_name)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>获取hubert结束\n")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_filename = "shoulinrui.m4a"
    # prepare(train_filename)
    # model1_train(train_filename,pretrained=True,num_epochs=200)
    # text = "我恨你"
    # prompt_audio_path = "resources/train/shoulinrui.m4a_0000513280_0000795840.wav"
    # max_length=10
    # model1_infer(text,prompt_audio_path,max_length)
    get_phonemes(train_filename)

