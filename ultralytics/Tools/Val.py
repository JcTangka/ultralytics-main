# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(
        "D://Work//PYworks//ultralytics-main//ultralytics//Tools//runs//detect//GAMTest1_mydataset_mypt_500//weights//best.pt"
    )  # GAM

    metrics = model.val(
        data="../cfg/datasets/All_my_school_dataset.yaml", split="val", plots=True, project="./runs/Val", name="GAMVal"
    )

    print(metrics.results_dict)
