import argparse
import collections
import logging
import os
import sys
import re
import pickle
from BPE import get_dict
from BPE import generate_bpe_file

from seq2seq import utils
from seq2seq.data.dictionary import Dictionary

SPACE_NORMALIZER = re.compile("\s+")


def word_tokenize(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def get_args():
    parser = argparse.ArgumentParser('Data pre-processing)')
    parser.add_argument('--source-lang', default="de", metavar='SRC', help='source language')
    parser.add_argument('--target-lang', default="en", metavar='TGT', help='target language')

    parser.add_argument('--train-prefix', default='~/atmt/assignment3/baseline/raw_data_back20000/train', metavar='FP', help='train file prefix')
    parser.add_argument('--tiny-train-prefix', default='~/atmt/assignment3/baseline/raw_data_back20000/tiny_train', metavar='FP', help='tiny train file prefix')
    parser.add_argument('--valid-prefix', default='~/atmt/assignment3/baseline/raw_data_back20000/valid', metavar='FP', help='valid file prefix')
    parser.add_argument('--test-prefix', default='~/atmt/assignment3/baseline/raw_data_back20000/test', metavar='FP', help='test file prefix')
    parser.add_argument('--dest-dir', default='~/atmt/assignment3/baseline/prepared_data_back_add20000', metavar='DIR', help='destination dir')

    parser.add_argument('--threshold-src', default=1, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words-src', default=-1, type=int, help='number of source words to retain')
    parser.add_argument('--threshold-tgt', default=1, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--num-words-tgt', default=-1, type=int, help='number of target words to retain')
    parser.add_argument('--bpe', default=True, help='Use BPE')
    return parser.parse_args()


def make_bpe_files(lang, code_name):
    generate_bpe_file(args.train_prefix + '.' + lang, args.train_prefix + '1.' + lang, code_name)
    generate_bpe_file(args.tiny_train_prefix + '.' + lang, args.tiny_train_prefix + '1.' + lang, code_name)
    generate_bpe_file(args.valid_prefix + '.' + lang, args.valid_prefix + '1.' + lang, code_name)
    generate_bpe_file(args.test_prefix + '.' + lang, args.test_prefix + '1.' + lang, code_name)


def main(args):
    os.makedirs(args.dest_dir, exist_ok=True)
    if(args.bpe):
        dict_file, code = get_dict(args)
        dictionary = Dictionary()
        src_dict = dictionary.load(dict_file[0], 10000)
        tgt_dict = dictionary.load(dict_file[1], 10000)

        make_bpe_files(args.source_lang, code)
        make_bpe_files(args.target_lang, code)
    else:
        src_dict = build_dictionary([args.train_prefix + '.' + args.source_lang])
        tgt_dict = build_dictionary([args.train_prefix + '.' + args.target_lang])

    src_dict.finalize(threshold=args.threshold_src, num_words=args.num_words_src)
    src_dict.save(os.path.join(args.dest_dir, 'dict.' + args.source_lang))
    logging.info('Built a source dictionary ({}) with {} words'.format(args.source_lang, len(src_dict)))

    tgt_dict.finalize(threshold=args.threshold_tgt, num_words=args.num_words_tgt)
    tgt_dict.save(os.path.join(args.dest_dir, 'dict.' + args.target_lang))
    logging.info('Built a target dictionary ({}) with {} words'.format(args.target_lang, len(tgt_dict)))


    def make_split_datasets(lang, dictionary, bpe):
        key = "1" if bpe else ""

        if args.train_prefix is not None:
            make_binary_dataset(args.train_prefix + key + '.' + lang, os.path.join(args.dest_dir, 'train.' + lang),
                                dictionary)
        if args.tiny_train_prefix is not None:
            make_binary_dataset(args.tiny_train_prefix + key + '.' + lang, os.path.join(args.dest_dir, 'tiny_train.' + lang),
                                dictionary)
        if args.valid_prefix is not None:
            make_binary_dataset(args.valid_prefix + key + '.' + lang, os.path.join(args.dest_dir, 'valid.' + lang),
                                dictionary)
        if args.test_prefix is not None:
            make_binary_dataset(args.test_prefix + key + '.' + lang, os.path.join(args.dest_dir, 'test.' + lang), dictionary)

    make_split_datasets(args.source_lang, src_dict, args.bpe)
    make_split_datasets(args.target_lang, tgt_dict, args.bpe)
    ''' source_dict = Dictionary.load("/Users/chaofeng/atmt/assignment3/baseline/prepared_data/dict.en",4000)
    target_dict = Dictionary.load("/Users/chaofeng/atmt/assignment3/baseline/prepared_data/dict.de", 4000)
    path = "/Users/chaofeng/atmt/assignment3/baseline/pare/"
    source_file = path + "train.en"
    target_file = path + "train.de"
    sor_output =  path+"train1.en"
    tar_output = path + "train1.de"
    make_binary_dataset(source_file,sor_output, source_dict)
    make_binary_dataset(target_file, tar_output, target_dict)'''


def build_dictionary(filenames, tokenize=word_tokenize):
    dictionary = Dictionary()
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                for symbol in word_tokenize(line.strip()):
                    dictionary.add_word(symbol)
                dictionary.add_word(dictionary.eos_word)
    return dictionary


def make_binary_dataset(input_file, output_file, dictionary, tokenize=word_tokenize, append_eos=True):
    nsent, ntok = 0, 0
    unk_counter = collections.Counter()

    def unk_consumer(word, idx):
        if idx == dictionary.unk_idx and word != dictionary.unk_word:
            unk_counter.update([word])

    tokens_list = []
    with open(input_file, 'r') as inf:
        for line in inf:
            tokens = dictionary.binarize(line.strip(), word_tokenize, append_eos, consumer=unk_consumer)
            nsent, ntok = nsent + 1, ntok + len(tokens)
            tokens_list.append(tokens.numpy())

    with open(output_file, 'wb') as outf:
        pickle.dump(tokens_list, outf, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('Built a binary dataset for {}: {} sentences, {} tokens, {:.3f}% replaced by unknown token'.format(
            input_file, nsent, ntok, 100.0 * sum(unk_counter.values()) / ntok, dictionary.unk_word))


if __name__ == '__main__':
    args = get_args()
    utils.init_logging(args)
    logging.info('COMMAND: %s' % ' '.join(sys.argv))
    logging.info('Arguments: {}'.format(vars(args)))
    main(args)
