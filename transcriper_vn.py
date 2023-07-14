from stt import Transcriber
transcriber = Transcriber(pretrain_model = '/home/tuannn/self-supervised-speech-recognition/1.pretrain/pretrain.pt', finetune_model = '/home/tuannn/self-supervised-speech-recognition/1.finetune/finetune.pt', 
                          dictionary = '/home/tuannn/self-supervised-speech-recognition/1.dictionary/dict.ltr.txt',
                          lm_type = 'kenlm',
                          lm_lexicon = '/home/tuannn/self-supervised-speech-recognition/1.vn_model/lexicon.txt', 
                          lm_model = '/home/tuannn/self-supervised-speech-recognition/1.vn_model/lm.bin',
                          lm_weight = 1.5, word_score = -1, beam_size = 50)
hypos = transcriber.transcribe(['/home/tuannn/self-supervised-speech-recognition/tuannn/record_data/Test1.wav'])
print(hypos)
