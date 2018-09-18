import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.append(os.pardir)

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset
from train_config import config

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 32)

    # define the options
    batch_size = 64*2  # batch size for each GPU
    n_gpus = 1

    # number of tokens in training data (this for 1B Word Benchmark)
    # 연애의 과학 토크나이징된 카톡 데이터 (identified_corpus_20180105) 토큰 개수
    # 토큰 줄이기
    # n_train_tokens = 609518
    n_train_tokens = 80000000

    options = {
        'bidirectional': True,
        'char_cnn': {
            # 'activation': 'relu',
            'activation': 'tanh',
            'embedding': {'dim': 16},
            'filters': [[1, 32],
                        [2, 32],
                        [3, 64],
                        [4, 128],
                        [5, 256],
                        [6, 512],
                        [7, 1024]],
            'max_characters_per_token': 32,
            'n_characters': 62,
            'n_highway': 2,
        },
        'dropout': 0.2,

        'lstm': {
            'cell_clip': 3,
            'dim': 256,
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 256,
            'use_skip_connections': True,
        },

        'all_clip_norm_val': 10.0,
        'n_epochs': 10,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 10,
        'n_negative_samples_batch': 4096,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix,
                                  vocab,
                                  test=False,
                                  shuffle_on_load=True)
    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    from datetime import datetime

    ROOT_PATH = config['ROOT_PATH']
    sol_paths_201801015 = config['sol_paths_201801015']
    textat_paths_201801015 = config['textat_paths_201801015']
    sol_data_pattern = os.path.join(ROOT_PATH, sol_paths_201801015)
    textat_data_pattern = os.path.join(ROOT_PATH, textat_paths_201801015)
    filepattern = [sol_data_pattern, textat_data_pattern]

    now = datetime.now()
    date_fmt = '{:%m%d_%H%M}'.format(now)
    save_dir = config['save_dir'].format(date_fmt)
    vocab_file = config['vocab_file']

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files', default=save_dir)
    parser.add_argument('--vocab_file', help='Vocabulary file', default=vocab_file)
    # parser.add_argument('--train_prefix', help='Prefix for train files', default=train_prefix)
    parser.add_argument('--train_prefix', help='Prefix for train files', default=filepattern)
    args = parser.parse_args()

    main(args)
