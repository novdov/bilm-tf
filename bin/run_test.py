import argparse
import os
import sys
import random

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.append(os.pardir)

from bilm.training import test, load_options_latest_checkpoint, load_vocab
from bilm.data import LMDataset, BidirectionalLMDataset
from train_config import config

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main(args):
    options, ckpt_file = load_options_latest_checkpoint(args.save_dir)

    # load the vocab
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    vocab = load_vocab(args.vocab_file, max_word_length)

    test_prefix = args.test_prefix

    kwargs = {
        'test': True,
        'shuffle_on_load': False,
    }

    if options.get('bidirectional'):
        data = BidirectionalLMDataset(test_prefix, vocab, **kwargs)
    else:
        data = LMDataset(test_prefix, vocab, **kwargs)

    test(options, ckpt_file, data, batch_size=args.batch_size)


if __name__ == '__main__':
    test_dir = '/media/scatter/scatterdisk/pingpong/raw_corpus/identified_corpus_20171212/textat/messages.org/00/'
    # test_candidates = random.sample(os.listdir(test_dir), 1)
    # test_prefix = [os.path.join(test_dir, cand, '*') for cand in test_candidates]
    test_sub_dir = os.path.join(test_dir, '73')
    fnames = os.listdir(test_sub_dir)[:5]
    test_prefix = [os.path.join(test_sub_dir, fname) for fname in fnames]

    parser = argparse.ArgumentParser(description='Compute test perplexity')
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file', default=config['vocab_file'])
    parser.add_argument('--test_prefix', help='Prefix for test files', default=test_prefix)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    args = parser.parse_args()
    main(args)
