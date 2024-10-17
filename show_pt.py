import torch
file=torch.load("resources/text2phonemes/shoulinrui_m4a_0000063040_0000325440_wav.pt",weights_only=True)
print(file)
print(file.shape)