import codecs
import sacrebleu

ref_t = []
pred_t = []
path = "~/atmt/assignment3/"
ref_file = path + "baseline/raw_data/test.en"
pred_file = path + "translated/model_translations_bpe4000.txt"
with codecs.open(ref_file, "r") as ref:
    for i, line in enumerate(ref):
        if i < 500:
            ref_t.append(line)
with codecs.open(pred_file, "r") as pred:
    for i, line in enumerate(pred):
        if i < 500:
            pred_t.append(line)
ref_t_1 = []
ref_t_1.append(ref_t)
print(sacrebleu.corpus_bleu(pred_t, ref_t_1))