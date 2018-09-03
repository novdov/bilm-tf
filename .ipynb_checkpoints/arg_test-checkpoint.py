import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))

parser = argparse.ArgumentParser()
parser.add_argument('--train_prefix')
args = parser.parse_args()

file_patterns = glob.glob(args.train_prefix)

with open('args_text.txt', 'w') as fw:
    for file_p in file_patterns:
        fw.write('{}\n'.format(file_p))

print('writing finished')
