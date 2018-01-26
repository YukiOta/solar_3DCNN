# coding: utf-8
""" Prediction with CNN
input: fisheye image
out: Generated Power
クロスバリデーションもする
とりあえずkeras
"""

# library
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import time
import seaborn as sns
import Load_data as ld
import argparse
import gc
# matplotlib.use('Agg')

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
# from keras.callbacks import EarlyStopping
from keras import optimizers


# {{{ model
def CNN_model1(activation="relu",
               loss="mean_squared_error",
               optimizer="Adadelta",
               layer=0, height=0, width=0):
    """
    INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*2 -> [FC -> RELU]*2 -> OUT
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(layer, height, width)))
    model.add(BatchNormalization())
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(BatchNormalization())
    model.add(Convolution2D(16, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def CNN_model2(activation="relu",
               loss="mean_squared_error",
               optimizer="Adadelta",
               layer=0,
               height=0,
               width=0):
    """
    INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> OUT
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(layer, height, width)))
    model.add(BatchNormalization())
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation=activation))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def CNN_model3(activation="relu",
               loss="mean_squared_error",
               optimizer="Adadelta",
               layer=0,
               height=0,
               width=0,
               out_num=3):
    """
    INPUT -> [CONV -> RELU] -> OUT
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(layer, height, width),
                            data_format="channels_first"))
    model.add(Convolution2D(16, 3, 3, border_mode='same',
                            data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(out_num))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model
# }}}


# {{{
def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    # top_k = history.history['top_k_categorical_accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    # val_top_k = history.history['val_top_k_categorical_accuracy']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n"
                     % (i, loss[i], acc[i], val_loss[i], val_acc[i]))
# }}}


# args = [50, 4, 3, 40, 64]


def main(parameters, DIRS):
    """
    img_dir_path_dic に，pathを各日のディレクトリを追加していく
    """
    im_size, band_num, predict_num, nb_epoch, batch_size = parameters
    DATA_DIR, TARGET_DIR, SAVE_DIR = DIRS
    optimizer = optimizers.RMSprop()
    # パラメータの吐き出し
    with open(SAVE_DIR + 'setting.txt', 'w') as f:
        f.write('save_dir: ' + SAVE_DIR + '\n')
        f.write('im_size: ' + str(im_size) + '\n')
        f.write('batch_size: ' + str(batch_size) + '\n')
        f.write('epochs: ' + str(nb_epoch) + '\n')
        f.write('optimizer: ' + str(optimizer) + '\n')
        f.write('predct__num :' + str(predict_num) + '\n')
        f.write('COMMENT: no Maxpooling, and shuffle added, and SGD\n')

    img_dir_path_dic = {}
    date_list = []
    error_date_list = []
    img_tr = []
    target_tr = []

    img_month_list = os.listdir(DATA_DIR)
    img_month_list.sort()

    for month_dir in img_month_list:
        if not month_dir.startswith("."):
            im_dir = os.path.join(DATA_DIR, month_dir)
            img_day_list = os.listdir(im_dir)
            img_day_list.sort()
            for day_dir in img_day_list:
                if not day_dir.startswith("."):
                    dir_path = os.path.join(im_dir, day_dir)
                    img_dir_path_dic[day_dir[:8]] = dir_path

    """ データの読み込み
    """

    target_month_list = os.listdir(TARGET_DIR)
    target_month_list.sort()

    # テストの時用
    COUNT = 0  # COUNTで読み込む日数を指定する

    for month_dir in target_month_list:
        if not month_dir.startswith("."):
            # if month_dir == "201705":
            im_dir = os.path.join(TARGET_DIR, month_dir)
            target_day_list = os.listdir(im_dir)
            target_day_list.sort()
            for day_dir in target_day_list:
                if not day_dir.startswith("."):
                    # 各日のディレクトリのpath
                    file_path = os.path.join(im_dir, day_dir)
                    if COUNT < 3:
                        COUNT += 1
                        print("---- TRY ----- " + day_dir[3:11])
                        try:
                            target_tmp = ld.load_target(
                                csv=file_path,
                                imgdir=img_dir_path_dic[day_dir[3:11]]
                            )
                            img_tmp = ld.load_image(
                                imgdir=img_dir_path_dic[day_dir[3:11]],
                                size=(im_size, im_size), norm=True
                            )
                            if len(target_tmp) == len(img_tmp):
                                target_tr.append(target_tmp)  # (number, channel)
                                date_list.append(day_dir[3:11])
                                img_tr.append(img_tmp)
                                print("   OKAY")
                            else:
                                print("   数が一致しません on " + day_dir[3:11])
                                print("   target: {}".format(len(target_tmp)))
                                print("   img: {}".format(len(img_tmp)))
                                error_date_list.append(day_dir[3:11])
                        except:
                            print("   Imageデータがありません on " + day_dir[3:11])

    # errorの日を保存
    with open(SAVE_DIR + "error_date.txt", "w") as f:
        f.write(str(error_date_list))

    print("Data Load Done. Starting traning.....")
    print("training on days " + str(date_list))

    LOSS_list = []
    ACC_list = []
    # traning
    for i in range(len(date_list)):

        print("-----Training on " + str(date_list[i]) + "-----")
        title = str(date_list[i])

        ts_img = 0
        ts_target = 0
        ts_img_pool = 0
        ts_target_pool = 0
        # i=1
        ts_img_pool = img_tr.pop(i)
        ts_target_pool = target_tr.pop(i)
        ts_img = ts_img_pool.copy()
        ts_target = ts_target_pool.copy()

        img_tr_all = 0
        target_tr_all = 0
        img_tr_all = np.concatenate((
            img_tr[:]
        ), axis=0)
        target_tr_all = np.concatenate((
            target_tr[:]
        ), axis=0)

        # テストデータと訓練データから、訓練データでとった平均を引く
        mean_img = ld.compute_mean(image_array=img_tr_all)
        img_tr_all -= mean_img
        ts_img -= mean_img

        """
        # 多バンド化の実行します
        # 1.トレーニング画像
        # 2.テスト画像
        # 3.トレーニングターゲット
        # 4.テストターゲット
        """
        # 1.トレーニング画像
        # リストの定義
        tmp = []  # 一時的に画像をプールする
        BAND_NUM = band_num  # まとめる枚数の定義 (共通)
        img_n_sec = []  # n枚バンド化した画像を追加するリスト
        img_band_array = 0  # 最終的に使うnumpy配列
        for j in range(len(img_tr_all)):
            if j <= len(img_tr_all) - BAND_NUM:
                tmp_nparray = np.concatenate((
                    img_tr_all[j:j + BAND_NUM]
                ), axis=2)  # 画像が(im_size, im_size, 3)の形で入っているので、axisは2
                img_n_sec.append(tmp_nparray)  # バンド化した画像をリストにappend
                tmp = []  # プールリストの初期化
            else:
                break
        img_band_array = np.array(img_n_sec, dtype=float)
        del img_n_sec
        print(img_band_array.shape)

        # 2.テスト画像
        # リストの定義
        tmp = []  # 一時的に画像をプールする
        img_n_sec_ts = []  # n枚バンド化した画像を追加するリスト
        img_band_array_ts = 0  # 最終的に使うnumpy配列
        for j in range(len(ts_img)):
            if j <= len(ts_img) - BAND_NUM:
                tmp_nparray = np.concatenate((
                    ts_img[j:j + BAND_NUM]
                ), axis=2)
                img_n_sec_ts.append(tmp_nparray)  # バンド化した画像をリストにappend
                tmp = []  # プールリストの初期化
            else:
                break
        img_band_array_ts = np.array(img_n_sec_ts, dtype=float)
        del img_n_sec_ts
        gc.collect()
        print(img_band_array_ts.shape)

        # 次にターゲット(発電量)
        # 3.トレーニングターゲット
        # リストの定義
        tmp = []  # 一時的に画像をプールする
        target_n_sec = []  # n枚バンド化した発電量を追加するリスト
        PREDICT_NUM = predict_num
        # target_band_array = 0  # 最終的に使うnumpy配列
        for j in range(len(target_tr_all)):
            if j <= len(target_tr_all) - BAND_NUM:
                index = j + BAND_NUM
                tmp = target_tr_all[index:(index+PREDICT_NUM), 1]
                target_n_sec.append(tmp)  # バンド化したターゲットをリストにappend
                # tmp = []  # プールリストの初期化
            else:
                break
        target_n_sec = np.array(target_n_sec, dtype=np.float32)  # ここでエラーでた
        target_band_array = target_n_sec
        target_train = target_band_array.copy()
        len_target = len(target_band_array)
        target_band_array = np.concatenate((
            target_tr_all[len_target, 0][:, np.newaxis],
            target_band_array[len_target, 1][:, np.newaxis],
            target_tr_all[len_target, 2][:, np.newaxis]
        ), axis=1)  # 配列を(data数, 3)に直さないと、データプロットのところでエラー出る
        print(target_band_array.shape)

        # 4.テストターゲット
        # リストの定義
        tmp = []  # 一時的に画像をプールする
        target_n_sec_ts = []  # n枚バンド化した発電量を追加するリスト
        # target_band_array_ts = 0  # 最終的に使うnumpy配列
        for j in range(len(ts_target)):
            if j <= len(ts_target) - BAND_NUM:
                index = j + BAND_NUM
                tmp = ts_target[index:index + PREDICT_NUM, 1]
                target_n_sec_ts.append(tmp)  # バンド化したターゲットをリストにappend
                # tmp = []  # プールリストの初期化
            else:
                break
        target_band_array_ts = np.array(target_n_sec_ts, dtype=np.float32)
        target_test = target_band_array_ts.copy()
        len_target = len(target_band_array_ts)
        target_band_array_ts = np.concatenate((
            ts_target[len_target, 0][:, np.newaxis],
            target_band_array_ts[len_target, 1][:, np.newaxis],
            ts_target[len_target, 2][:, np.newaxis]
        ), axis=1)
        print(target_band_array_ts.shape)

        print("Bandalized DONE")
        """
        バンド化おわり
        img_band_array = 0  # 最終的に使うnumpy配列
        target_band_array = 0  # 最終的に使うnumpy配列
        """

        # transpose for CNN INPUT shit
        img_band_array = img_band_array.transpose(0, 3, 1, 2)
        print(img_band_array.shape)
        # set image size
        layer = img_band_array.shape[1]
        height = img_band_array.shape[2]
        width = img_band_array.shape[3]

        print("Image and Target Ready")

        # model set
        model = None
        # model = CNN_model3(
        #     activation="relu",
        #     optimizer="Adadelta",
        #     layer=layer,
        #     height=height,
        #     width=width)
        model = CNN_model3(
            activation="relu",
            optimizer=optimizer,
            layer=layer,
            height=height,
            width=width,
            out_num=PREDICT_NUM)

        # initialize check
        # data_plot(model=model, target=target_band_array_ts,
        #           img=img_band_array_ts,
        #           batch_size=batch_size,
        #           date=date_list[i], save_csv=True)

        # early_stopping = EarlyStopping(patience=3, verbose=1)

        # Learning model
        history = model.fit(img_band_array, target_train,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            validation_split=0.1,
                            )
        # data_plot(
        #     model=model, target=target_band_array_ts,
        #     img=img_band_array_ts, batch_size=batch_size,
        #     date=date_list[i], save_csv=True)
        # evaluate
        try:
            x_test = img_band_array_ts.transpose(0, 3, 1, 2)
            t_test = target_test
            loss_test, acc_test = \
                model.evaluate(x_test, t_test, verbose=1)
            print('TEST LOSS:  ', loss_test)
            print('TEST ACC :  ', acc_test)
            # print('TEST TOP5: ', topk_test)
            LOSS_list.append(loss_test)
            ACC_list.append(acc_test)
        except:
            print('fail to evaluate')

        try:
            # model.save_weights(os.path.join(save_dir, 'equi_model.h5'))
            save_history(history,
                         os.path.join(SAVE_DIR, 'history_'+title+'.txt'))
            model.save(os.path.join(SAVE_DIR, 'whole_model.h5'))
        except:
            print('fail to save')

        # put back data
        img_tr.insert(i, ts_img_pool)
        target_tr.insert(i, ts_target_pool)

        sns.set()
        plt.subplot(1, 2, 1)
        plt.figure(figsize=(8, 8))
        plt.plot(history.history['acc'], label='train')
        plt.plot(history.history['val_acc'], label='val')
        plt.legend()
        plt.title("ACC " + title)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        # plt.savefig(save_dir+'accuracy.png')

        plt.subplot(1, 2, 2)
        plt.figure(figsize=(8, 8))
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.legend()
        plt.title("LOSS "+title)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(SAVE_DIR+'acc_loss_' + title + '.png')


if __name__ == '__main__':
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

    # 時間の表示
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + " [sec]")
