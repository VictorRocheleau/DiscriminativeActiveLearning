from pathlib import Path
from shutil import copyfile
from sklearn.model_selection import train_test_split
import numpy as np

# for path in Path('BreaKHis_v1').rglob('*.png'):
#     print(path)
#
# paths = [path for path in Path('BreaKHis_v1').rglob('*.png')]

class extractor:

    def __init__(self):
        pass

    def analyse(self, path, level, test_size=0.2):

        paths = [str(path) for path in Path(path).rglob('*.png')]

        benign_files = list(filter(lambda f: level in f and 'benign' in f, paths))
        malignant_files = list(filter(lambda f: level in f and 'malignant' in f, paths))

        print('Benign {} : {}'.format(level, len(benign_files)))
        print('Malignant {} : {}'.format(level, len(malignant_files)))

        return len(benign_files), len(malignant_files)


    def copy_to_dir(self, src, level):

        paths = [str(path) for path in Path(src).rglob('*.png')]

        benign_files = list(filter(lambda f: level in f and 'benign' in f, paths))
        malignant_files = list(filter(lambda f: level in f and 'malignant' in f, paths))

        Path('breakhis/train/{}/benign/'.format(level)).mkdir(parents=True, exist_ok=True)
        Path('breakhis/test/{}/benign/'.format(level)).mkdir(parents=True, exist_ok=True)
        # Path('breakhis/val/{}/benign/'.format(level)).mkdir(parents=True, exist_ok=True)

        Path('breakhis/train/{}/malignant/'.format(level)).mkdir(parents=True, exist_ok=True)
        Path('breakhis/test/{}/malignant/'.format(level)).mkdir(parents=True, exist_ok=True)
        # Path('breakhis/val/{}/malignant/'.format(level)).mkdir(parents=True, exist_ok=True)

        b_train_files, b_test_files = train_test_split(benign_files, test_size=0.2)
        m_train_files, m_test_files = train_test_split(malignant_files, test_size=0.2)

        # b_val_files, b_test_files = train_test_split(b_holdhout_files, test_size=0.5)
        # m_val_files, m_test_files = train_test_split(m_holdhout_files, test_size=0.5)

        # ________BENIGN________
        # BENIGN TRAIN SET
        for i, file in enumerate(b_train_files):
            print('Writing : {}'.format(file))
            out_path = 'breakhis/train/{}/benign/benign_{}.png'.format(level, i)
            copyfile(file, out_path)

        # BENIGN TEST SET
        for i, file in enumerate(b_test_files):
            print('Writing : {}'.format(file))
            out_path = 'breakhis/test/{}/benign/benign_{}.png'.format(level, i)
            copyfile(file, out_path)

        # BENIGN VAL SET
        # for i, file in enumerate(b_val_files):
        #     print('Writing : {}'.format(file))
        #     out_path = 'breakhis/val/{}/benign/benign_{}.png'.format(level, i)
        #     copyfile(file, out_path)

        # ________MALIGNANT________
        # MALIGNANT TRAIN SET
        for i, file in enumerate(m_train_files):
            print('Writing : {}'.format(file))
            out_path = 'breakhis/train/{}/malignant/malignant_{}.png'.format(level, i)
            copyfile(file, out_path)

        # MALIGNANT TEST SET
        for i, file in enumerate(m_test_files):
            print('Writing : {}'.format(file))
            out_path = 'breakhis/test/{}/malignant/malignant_{}.png'.format(level, i)
            copyfile(file, out_path)

        # MALIGNANT VAL SET
        # for i, file in enumerate(m_val_files):
        #     print('Writing : {}'.format(file))
        #     out_path = 'breakhis/val/{}/malignant/malignant_{}.png'.format(level, i)
        #     copyfile(file, out_path)


if __name__ == "__main__":

    extractor = extractor()

    zooms = ['40X', '100X', '200X', '400X']

    total = 0

    for level in zooms:
        b, m = extractor.analyse('BreaKHis_v1', level, 0.2)

        total += b
        total += m
    print("Total : {}".format(total))

    for level in zooms:
        extractor.copy_to_dir('BreaKHis_v1', level)

    total = 0

    for level in zooms:
        b, m = extractor.analyse('breakhis', level, 0.2)

        total += b
        total += m
    print("Total : {}".format(total))
