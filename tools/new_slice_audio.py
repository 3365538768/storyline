import os
import sys
import numpy as np
import traceback
from scipy.io import wavfile
from tools.my_utils import load_audio
from tools.slicer2 import Slicer


def slice(inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, i_part, all_part):
    os.makedirs(opt_root, exist_ok=True)

    filename=os.path.basename(inp)
    output_log_path = os.path.join(f"resources/slice/intervals/{filename}/slice_log.txt")
    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)

    if os.path.isfile(inp):
        input_files = [inp]
    elif os.path.isdir(inp):
        input_files = [os.path.join(inp, name) for name in sorted(list(os.listdir(inp)))]
    else:
        return "输入路径存在但既不是文件也不是文件夹"

    slicer = Slicer(
        sr=32000,  # 长音频采样率
        threshold=int(threshold),  # 静音阈值
        min_length=int(min_length),  # 每段最小长度
        min_interval=int(min_interval),  # 最短切割间隔
        hop_size=int(hop_size),  # 步长
        max_sil_kept=int(max_sil_kept),  # 最大静音保留长度
    )

    _max = float(_max)
    alpha = float(alpha)

    with open(output_log_path, 'w') as log_file:
        for inp_path in input_files[int(i_part)::int(all_part)]:
            try:
                name = os.path.basename(inp_path)
                audio = load_audio(inp_path, 32000)

                for chunk, start, end in slicer.slice(audio):  # start和end是帧数
                    # 计算对应的时间
                    start_time = start / 32000  # 转换为秒
                    end_time = end / 32000  # 转换为秒

                    tmp_max = np.abs(chunk).max()
                    if tmp_max > 1:
                        chunk /= tmp_max

                    chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
                    wavfile.write(
                        "%s/%s_%010d_%010d.wav" % (opt_root, name, start, end),
                        32000,
                        (chunk * 32767).astype(np.int16),
                    )

                    # 记录到日志
                    log_file.write(f"{start_time:.2f}/{end_time:.2f}\n")

            except Exception as e:
                print(inp_path, "->fail->", traceback.format_exc())

    return "执行完毕，请检查输出文件和日志"



