from argparse import ArgumentParser

import numpy as np

def main(path_array_one, path_array_two, save_path):
    array_one = np.load(path_array_one)
    array_two = np.load(path_array_two)
    new_array = np.concatenate(array_one, array_two)
    np.save(save_path, )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path_array_one', type=str)
    parser.add_argument('--path_array_two', type=str)
    parser.add_argument('--save_path', type=str)
    main(**vars(parser.parse_args()))