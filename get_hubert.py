import torch
import torch.nn.functional as F
import soundfile as sf
import os

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

def wav2hubert(wav_path,exp_name):
    model_path="models/chinese-hubert-large"
    wave_name=os.path.basename(wav_path)
    print(wave_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = HubertModel.from_pretrained(model_path)

    # for pretrain: Wav2Vec2ForPreTraining
    # model = Wav2Vec2ForPreTraining.from_pretrained(model_path)

    model = model.to(device)
    model = model.half()
    model.eval()

    wav, sr = sf.read(wav_path)
    input_values = feature_extractor(wav, return_tensors="pt",sampling_rate=16000).input_values
    input_values = input_values.half()
    input_values = input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values,)
        last_hidden_state = outputs.last_hidden_state
    os.makedirs(f"resources/wave_hubert/{exp_name}", exist_ok=True)
    torch.save(last_hidden_state, f"resources/wave_hubert/{exp_name}/{wave_name}.pt")
    return last_hidden_state
