import argparse
import time

from model_maker import ModelMaker

NUM_CLASSES = 10  # 分類数、数字なので10種類
SAMPLERATE = 8000

INFO_FILE = "keras_model/my_model_info.txt"
GRAPH_FILE = "keras_model/my_model_graph.pdf"
EST_FILE = "keras_model/my_model.h5"  # 推定器のモデルの保存先
# INPUT_SIZE = (,)
LR = 1e-4
DATA_SIZE = 4096
BATCH_SIZE = 128  # 訓練データを128ずつのデータに分ける
EPOCH = 30  # 訓練データを繰り返し学習させる数
VALID_RATE = 0.2

def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument("-p", "--path_to_truth", type=str, help='テストデータの正解ファイルCSVのパス')
    args = parser.parse_args()
    start_time = time.time()

    maker = ModelMaker(
        samplerate = SAMPLERATE,
        
        info_file= INFO_FILE,
        graph_file=GRAPH_FILE,
        est_file = EST_FILE,
        lr = LR,
        data_size = DATA_SIZE,
        batch_size = BATCH_SIZE,
        epochs = EPOCH,
        valid_rate = VALID_RATE
    )
    
    maker.execute()


if __name__ == "__main__":
    main()