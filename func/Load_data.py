# coding: utf-8
"""solar_projectにおける、データ整理用プログラム
1.発電量csvファイルから、必要な列を取り出して、整理する
2.画像の読み込み
"""

import numpy as np
import os
import pandas as pd
import datetime as dt
import argparse
from PIL import Image
from scipy import ndimage

def load_image(imgdir, size, norm=True, median=True, median_size=3, mean=True):
    """
    input: image directory
    output: numpy array
    """
    images = []
    imglist = os.listdir(imgdir)
    imglist.sort()
    for filename in imglist:
        if not filename.startswith('.'):
            if len(filename) > 10:
                img = Image.open(os.path.join(imgdir, filename))
                if img is not None:
                    img = img.resize(size)
                    img = np.array(img)
                    if norm is True:
                        img = img / 255.
                    if median is True:
                        img = ndimage.median_filter(img, median_size)
                    images.append(img)
    images = np.array(images, dtype=np.float32)
    print("---------- Image Load Done")
    return images


def load_target(csv, imgdir):
    """
    csvファイルを読み込む
    Pandasを用いて、処理する
    csv: csvファイルのありか　(./hoge.csv)
    imgdir: 対応するimgdirのありか
    interval: 撮影インターバル (sec)

    out: target [time, Generated Power, temperature]
    target[:, 0] : time
    target[:, 1] : power
    target[:, 2] : temperature
    $ target.shape
    -> (number, channel) ex) (164, 3)
    """
    # csvファイルの読み込み
    df = pd.read_csv(csv)

    # 必要な列(時刻、発電量、気温)を取り出し、必要のない行を取り除く
    df = df.iloc[:, [0, 11, 36]]
    df.columns = ["time", "power", "temperature"]
    df = df.drop([0, 1])
    # データフレームをnumpy配列にする
    csv_tmp = np.array(df)

    # 画像撮影時刻を補正する
    filelist = os.listdir(imgdir)
    # ソートする (これでかなり悩んだ)
    filelist.sort()
    time_start, time_end = check_target_time(filelist=filelist)
    # print(time_start, time_end)

    # 時間をつきあわせる
    # i_start = 0
    # i_end = 0
    for i in range(len(csv_tmp)):

        if csv_tmp[i][0].replace(':', '') == time_start:
            i_start = i
            # print(i)
            # print(csv_tmp[i])
        elif csv_tmp[i][0].replace(':', '') == time_end:
            i_end = i
            # print(i)
            # print(csv_tmp[i]

    # 画像の時間インターバルの計算
    time_a = filelist[10][9:15]
    time_b = filelist[11][9:15]
    delta_tmp = dt.datetime.strptime(time_b, "%H%M%S") - dt.datetime.strptime(time_a, "%H%M%S")
    interval = delta_tmp.seconds
    target = csv_tmp[i_start:i_end+int(1):int(interval/6)]

    # print("CSV Load Done")
    # print(len(target))
    # print(target)
    return target


def check_target_time(filelist):
    """
    画像の時刻と、ファイルの時刻が一致しないことがあるので、チェックする。
    """
    file_list = filelist

    # .DS_storeがとかがあるとエラー出るから回避
    if file_list[0][9:15] == "":
        time_tmp = file_list[1][9:15]
    else:
        time_tmp = file_list[0][9:15]

    if int(time_tmp[-2:]) % 6 >= 3:
        delta = 6 - (int(time_tmp[-2:]) % 6)
        timedelta = dt.timedelta(seconds=delta)
        time_start = dt.datetime.strptime(time_tmp, "%H%M%S")
        time_correct = time_start + timedelta
        time_correct_s = time_correct.strftime("%H%M%S")
    elif int(time_tmp[-2:]) % 6 < 3:
        delta = int(time_tmp[-2:]) % 6
        timedelta = dt.timedelta(seconds=delta)
        time_start = dt.datetime.strptime(time_tmp, "%H%M%S")
        time_correct = time_start - timedelta
        time_correct_s = time_correct.strftime("%H%M%S")

    # jpgのフォルダがあると数がずれてしまうからif文かいておく
    if file_list[-1][9:15] == "":
        time_tmp = file_list[-2][9:15]
    else:
        time_tmp = file_list[-1][9:15]

    if int(time_tmp[-2:]) % 6 >= 3:
        delta = 6 - (int(time_tmp[-2:]) % 6)
        timedelta = dt.timedelta(seconds=delta)
        time_start = dt.datetime.strptime(time_tmp, "%H%M%S")
        time_correct = time_start + timedelta
        time_correct_e = time_correct.strftime("%H%M%S")
    elif int(time_tmp[-2:]) % 6 < 3:
        delta = int(time_tmp[-2:]) % 6
        timedelta = dt.timedelta(seconds=delta)
        time_start = dt.datetime.strptime(time_tmp, "%H%M%S")
        time_correct = time_start - timedelta
        time_correct_e = time_correct.strftime("%H%M%S")

    return time_correct_s, time_correct_e

def compute_mean(image_array):
    """全画像の平均をとって、平均を返す
    入力：画像データセット (np配列を想定してる) [枚数, height, width, rgb]
    出力：平均画像
    """
    print("conmpute mean image")
    mean_image = np.ndarray.mean(image_array, axis=0)
    return mean_image

def main():
    """ 画像の日付リストの獲得
    img_20170101 = np.array
    みたいな感じで代入していく
    また、ディレクトリのパスをdictionalyに入れておくことで、targetのロードのときに役たてる
    """
    img_dir_path_dic = {}
    # img_name_list = []
    target_name_list = []
    img_tr = []
    target_tr = []

    for month_dir in os.listdir(DATA_DIR):
        if not month_dir.startswith("."):
            im_dir = os.path.join(DATA_DIR, month_dir)
            for day_dir in os.listdir(im_dir):
                if not day_dir.startswith("."):
                    dir_path = os.path.join(im_dir, day_dir)
                    img_dir_path_dic[day_dir[:8]] = dir_path
                    # img_name_list.append("img_"+day_dir[:8])

    """ ターゲットの読み込み
    target_20170101 = np.array
    みたいな感じで代入していく
    dictionalyに保存したpathをうまく利用
    """

    for month_dir in os.listdir(TARGET_DIR):
        if not month_dir.startswith("."):
            im_dir = os.path.join(TARGET_DIR, month_dir)
            for day_dir in os.listdir(im_dir):
                if not day_dir.startswith("."):
                    file_path = os.path.join(im_dir, day_dir)
                    # print(day_dir[3:11])
                    try:
                        target_tr.append(load_target(csv=file_path, imgdir=img_dir_path_dic[day_dir[3:11]]))
                        target_name_list.append("target_"+day_dir[3:11])
                        # img_tr.append(load_image(imgdir=dir_path, size=(224, 224), norm=True))
                    except:
                        print("Imageデータがありません at "+day_dir[3:11])
    print("Data Load Done")



# try:
#     img_dir_path_dic["20170111"]
# except:
#     print("unko")
    # np.savez(
    #     "image_from11to6_224.npz",
    #     *img_name_list[:]
    # )
    # np.savez(
    #     "target_from11to6.npz",
    #     *target_name_list[:]
    # )


if __name__ == '__main__':
    # mkdir
    SAVE_dir = "./RESULT/CNN_keras/"
    if not os.path.isdir(SAVE_dir):
        os.makedirs(SAVE_dir)

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="../data/PV_IMAGE/",
        help="choose your data (image) directory"
    )
    parser.add_argument(
        "--target_dir",
        default="../data/PV_CSV/",
        help="choose your target dir"
    )
    args = parser.parse_args()
    DATA_DIR, TARGET_DIR = args.data_dir, args.target_dir

    # main関数の実行
    main()


############################################
# DATA_DIR = "../data/PV_IMAGE/"
# TARGET_DIR = "../data/PV_CSV/"
# end
