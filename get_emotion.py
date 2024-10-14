'''
Using the finetuned emotion recognization model

rec_result contains {'feats', 'labels', 'scores'}
	extract_embedding=False: 9-class emotions with scores
	extract_embedding=True: 9-class emotions with scores, along with features

9-class emotions:
iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large (May. 2024 release)
iic/emotion2vec_base_finetuned (Jan. 2024 release)
    0: angry
    1: disgusted
    2: fearful
    3: happy
    4: neutral
    5: other
    6: sad
    7: surprised
    8: unknown
'''
from funasr import AutoModel
import numpy as np
model = AutoModel(model="iic/emotion2vec_base_finetuned")




def get_emotion_vec(file_path):
    rec_result= model.generate(file_path, granularity="utterance", extract_embedding=False)
    scores=np.round(rec_result[0]['scores'],2)
    return rec_result[0]['labels'], scores

def main():
    print(get_emotion_vec("resources/train/shoulinrui.m4a"))

if __name__ == '__main__':
    main()