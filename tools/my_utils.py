import platform,os,traceback
import ffmpeg
import numpy as np


def load_audio(file, sr):#sr 是另一个输入参数，表示目标采样率（Sample Rate），即音频文件将被重新采样到的采样率。
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)#使用 ffmpeg-python 库加载音频文件，threads=0 表示让 ffmpeg 自动决定使用的线程数。
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            #"-": 将输出重定向到标准输出流（stdout），而不是文件。
            # format="f32le": 指定输出格式为 32 位浮点数，使用小端字节序（little-endian）。
            # acodec="pcm_f32le": 指定音频编解码器为 32 位浮点数的 PCM 格式。
            # ac=1: 将音频转换为单声道（mono）。
            # ar=sr: 将音频重新采样到指定的采样率 sr。
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
        #返回值 out 包含了音频的二进制数据（标准输出的内容），_ 则捕获了标准错误流内容（但这里没有进一步使用）。
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load audio: {e}")
    #traceback.print_exc() 打印完整的错误堆栈信息，这对于调试很有帮助。
    # raise RuntimeError(f"Failed to load audio: {e}") 抛出一个新的 RuntimeError，包含原始错误信息，以便进一步处理。
    return np.frombuffer(out, np.float32).flatten()
#使用 numpy.frombuffer 函数将二进制数据转换为 NumPy 数组，数据类型为 32 位浮点型。将多维数组展平成一维数组。


def clean_path(path_str:str):
    if path_str.endswith(('\\','/')):
        return clean_path(path_str[0:-1])
    if platform.system() == 'Windows':
        path_str = path_str.replace('/', '\\')
    return path_str.strip(" ").strip('\'').strip("\n").strip('"').strip(" ").strip("\u202a")