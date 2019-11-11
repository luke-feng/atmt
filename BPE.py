import os
import codecs
import tempfile
from collections import Counter
from subword_nmt import learn_bpe
from subword_nmt import apply_bpe

def get_dict(args):
    input_args = [args.train_prefix + '.' + args.source_lang, args.train_prefix + '.' + args.target_lang]
    path = '/Users/chaofeng/atmt/assignment3/baseline/raw_data_back20000/'
    vocab_args = [path+"dict" + '.' + args.source_lang, path+"dict" + '.' + args.target_lang]
    #input_args = [path+"train.de", path+"train.en"]
    #vocab_args = [path+"dict.de", path+"dict.en"]

    separator = '@@'
    symbols = 10000
    min_frequency = 1
    output =path +  "code"

    # read/write files as UTF-8
    input = [codecs.open(f, encoding='UTF-8') for f in input_args]
    vocab = [codecs.open(f, mode='w', encoding='UTF-8') for f in vocab_args]
    # get combined vocabulary of all input texts
    full_vocab = Counter()
    for f in input:
        full_vocab += learn_bpe.get_vocabulary(f)
        f.seek(0)

    vocab_list = ['{0} {1}'.format(key, freq) for (key, freq) in full_vocab.items()]

    # learn BPE on combined vocabulary
    with codecs.open(output, mode='w', encoding='UTF-8') as file:
        learn_bpe.learn_bpe(vocab_list, file, symbols, min_frequency)

    with codecs.open(output, encoding='UTF-8') as codes:
        bpe = apply_bpe.BPE(codes, separator=separator)

    # apply BPE to each training corpus and get vocabulary
    for train_file, vocab_file in zip(input, vocab):

        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()

        tmpout = codecs.open(tmp.name, 'w', encoding='UTF-8')

        train_file.seek(0)
        for line in train_file:
            tmpout.write(bpe.segment(line).strip())
            tmpout.write('\n')

        tmpout.close()
        tmpin = codecs.open(tmp.name, encoding='UTF-8')

        vocab = learn_bpe.get_vocabulary(tmpin)
        tmpin.close()
        os.remove(tmp.name)

        for key, freq in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
            vocab_file.write("{0} {1}\n".format(key, freq))
    return vocab_args, output

def generate_bpe_file(input, output, code):
    inputs = codecs.open(input, encoding='utf-8')
    outputs = codecs.open(output, mode='w', encoding='UTF-8')
    codes = codecs.open(code, encoding='utf-8')
    bpe = apply_bpe.BPE(codes)
    for line in inputs:
        outputs.write(bpe.process_line(line))
    outputs.close()