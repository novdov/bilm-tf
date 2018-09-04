import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = 32  # batch size for each GPU
    n_gpus = 2

    # number of tokens in training data (this for 1B Word Benchmark)
    # n_train_tokens = 768648884
    # 연애의 과학 토크나이징된 카톡 데이터 (identified_corpus_20180105) 토큰 개수
    n_train_tokens = 605918

    options = {
        'bidirectional': True,
        'char_cnn': {
            'activation': 'relu',
            'embedding': {'dim': 16},
            'filters': [[1, 32],
                        [2, 32],
                        [3, 64],
                        [4, 128],
                        [5, 256],
                        [6, 512],
                        [7, 1024]],
            'max_characters_per_token': 50,
            'n_characters': 261,
            'n_highway': 2
        },

        'dropout': 0,
        'lstm': {
            'cell_clip': 3,
            'dim': 256,
            'n_layers': 3,
            'proj_clip': 3,
            'projection_dim': 256,
            'use_skip_connections': True
        },

        'all_clip_norm_val': 10.0,
        'n_epochs': 10,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 10,
        'n_negative_samples_batch': 30,
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

    ROOT_PATH = '/media/scatter/scatterdisk/pingpong/raw_corpus/'
    sol_paths_201801015 = 'identified_corpus_20180105/sol/messages/*/*'
    textat_paths_201801015 = 'identified_corpus_20180105/textat/messages.org/*/*/*'

    sol_data_pattern = os.path.join(ROOT_PATH, sol_paths_201801015)
    textat_data_pattern = os.path.join(ROOT_PATH, textat_paths_201801015)

    filepattern = [sol_data_pattern, textat_paths_201801015]

    now = datetime.now()
    date_fmt = '{:%m%d_%H%M}'.format(now)
    # train_prefix = '/media/scatter/scatterdisk/sandbox_temp/data/kakaotalk_sol_elmo/messages/*/*'
    save_dir = '/media/scatter/scatterdisk/elmo_ckpt/elmo_ckpt_{}'.format(date_fmt)
    vocab_file = '/media/scatter/scatterdisk/sandbox_temp/data/kakaotalk_sol_unique_tokens.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files', default=save_dir)
    parser.add_argument('--vocab_file', help='Vocabulary file', default=vocab_file)
    # parser.add_argument('--train_prefix', help='Prefix for train files', default=train_prefix)
    parser.add_argument('--train_prefix', help='Prefix for train files', default=filepattern)
    args = parser.parse_args()

    main(args)
