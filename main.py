import time
import datetime as dt
import os
import argparse
import solar_CNN_keras_multiplebands as solar

parser = argparse.ArgumentParser()
parser.add_argument(
    "-D", "--data_dir",
    default="../data/PV_IMAGE/",
    help="choose your data (image) directory"
)
parser.add_argument(
    "-T", "--target_dir",
    default="../data/PV_CSV/",
    help="choose your target dir"
)
parser.add_argument(
    "-S", "--save_dir",
    default="./RESULT/",
    help="choose save dir"
)

today_time = dt.datetime.today().strftime("%Y_%m_%d")

args = parser.parse_args()
DATA_DIR, TARGET_DIR, SAVE_DIR = \
    args.data_dir, args.target_dir, args.save_dir
SAVE_DIR = os.path.join(SAVE_DIR, today_time)
print(SAVE_DIR)
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def main():

    parameters = [50, 4, 3, 50, 32]
    DIRS = [DATA_DIR, TARGET_DIR, SAVE_DIR]
    solar.main(parameters, DIRS)


if __name__ == '__main__':

    start_main = time.time()
    main()
    elapsed_time = start_main - time.time()
    print("ELAPSED TIME : ", elapsed_time)
