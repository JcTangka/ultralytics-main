# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model # 三选一
    # model = YOLO('../cfg/models/v8/yolov8-BoT3.yaml')  # build a new model from YAML
    # model = YOLO('D://Work//PYworks//ultralytics-main//ultralytics//yolov8n.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('../cfg/models/v8/yolov8-BoT3.yaml').load('D://Work//PYworks//ultralytics-main//ultralytics//yolov8n.pt')  # BoT3
    # model = YOLO('../cfg/models/v8/yolov8-MHSA.yaml').load('D://Work//PYworks//ultralytics-main//ultralytics//yolov8n.pt')  # MHSA
    model = YOLO("../cfg/models/v8/yolov8-GAM.yaml").load(
        "D://Work//PYworks//ultralytics-main//ultralytics//yolov8n.pt"
    )  # GAM

    # Train the model
    # model.train(data='../cfg/datasets/All_my_school_dataset.yaml', epochs=500,project='./runs/detect',name='GAMTest2_mydataset_mypt_parameterchange',
    #             imgsz=960,conf=0.1,iou=0.6,patience=200)  #  涨4个点
    model.train(
        data="../cfg/datasets/All_my_school_dataset.yaml",
        epochs=500,
        project="./runs/detect",
        name="GAMTest3_mydataset_mypt_NetLocChange",
        imgsz=640,
        conf=0.1,
        iou=0.6,
        patience=200,
    )
