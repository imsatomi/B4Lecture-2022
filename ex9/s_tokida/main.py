from model_maker import ModelMaker

SAMPLERATE = 8000

INFO_FILE = "keras_model/my_model_info.txt"
GRAPH_FILE = "keras_model/my_model_graph.pdf"
EST_FILE = "keras_model/my_model.h5"  # 推定器のモデルの保存先
# INPUT_SIZE = (,)
LR = 1e-4
DATA_SIZE = 4096
BATCH_SIZE = 128
EPOCH = 30
VALID_RATE = 0.2

def main():

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