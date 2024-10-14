import librosa
import soundfile as sf
import os
from get_emotion import get_emotion_vec
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
import ast


def slice_audio(audio_name,window,hop):  #窗口切片音频
    # 加载音频文件
    audio, sr = librosa.load(audio_name)

    # 设置切片时长（以秒为单位）
    window_length = window * sr  # 10秒
    hop_length = hop*sr  # 移动5秒

    # 获得文件去除后缀的名字
    filename = audio_name
    new_filename = os.path.basename(filename)
    new_filename = os.path.splitext(new_filename)[0]

    # 创建保存切片的文件夹
    path = f"slice_audio/{new_filename}"
    if not os.path.exists(path):
        os.makedirs(path)

    for start in range(0, len(audio) - window_length + 1, hop_length):
        slice_audio = audio[start:start + window_length]
        sf.write(f"slice_audio/{new_filename}/slice_{start // hop_length}.wav", slice_audio, sr)

    print(f">>>>文件({audio_name})切片成功,保存路径({path})")
    return path,new_filename

def get_all_emotion(slice_path):  #对所有切片获得情感值，合并成一个list
    audio_emotion=[]
    for root, dirs, files in os.walk(slice_path):  # os.walk 遍历指定目录及其子目录中的所有文件  #root: 当前目录的路径。dirs: 当前目录中的子目录列表。files: 当前目录中的文件列表。
        for file in files:
            file_path = os.path.join(root, file)
            labels,scores=get_emotion_vec(file_path)
            audio_emotion.append(scores)
    return labels,audio_emotion

def draw_emotion(data,labels,name,window): #绘图
    labels=["生气/angry","厌恶/disgusted","恐惧/fearful","开心/happy","中立/neutral","其他/other","难过/sad","吃惊/surprised","<unk>"]
    transposed_data = list(zip(*data))

    # 创建一个图形
    plt.figure(figsize=(100, 20))

    # 绘制每个特征的曲线
    for i in range(len(transposed_data)):
        plt.plot(transposed_data[i], label=f'{labels[i]}')

    # 添加图例和标签
    plt.legend()
    plt.title('情感变化')
    plt.xlabel('时间')
    plt.ylabel('特征值')
    os.makedirs(f"resources/temp/{name}",exist_ok=True)
    plt.savefig(f'resources/temp/{name}/{window}.png')

def save_list(data,filepath):  #保存list为txt
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  #没有就创建文件夹
    with open(filepath, 'w') as f:
        for item in data:
            f.write("%s\n" % item)

def read_txt(filepath):  #读取list
    with open(filepath, 'r') as f:
        data = [line.strip() for line in f.readlines()]  # 去除换行符
    return data

def read_label_emotion(filename):  #读取labels和emotion并转化为list
    labels = read_txt(f"emotion_data/{filename}/labels.txt")
    audio_emotion = read_txt(f"emotion_data/{filename}/audio_emotion.txt")
    for i in range(len(audio_emotion)):
        audio_emotion[i] = string2list(audio_emotion[i])
    return labels, audio_emotion

def string2list(string_list):  #字符串list变为数字list
    actual_list = ast.literal_eval(string_list)
    return actual_list

def main():
    filename="shoulinrui.m4a"
    slice_path=f"resources/slice/{filename}"
    labels,audio_emotion=get_all_emotion(slice_path)  #获取每个切片的情感
    save_list(labels,f"resources/emotion_data/{filename}/labels.txt")
    save_list(audio_emotion,f"resources/emotion_data/{filename}/audio_emotion.txt")
    draw_emotion(audio_emotion,labels,filename)  #绘制情感图

if __name__ == '__main__':
    new_data=True  #如果是新的数据
    if new_data:
        main()
    else: #如果仅仅是读取已保存的
        filename="shoulinrui"
        labels,audio_emotion=read_label_emotion(filename)
        draw_emotion(audio_emotion,labels,filename)  #绘制情感图
