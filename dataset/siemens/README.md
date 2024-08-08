
Start LabelMe

```
    python -m venv venv
    source ./venv/Scripts/activate

    pip install labelme

    labelme # start up
```

Convert *.json files to COCO format: 

```
python .\datasets\convert_coco.py C:\Users\z004c84d\Documents\DEV\detectron2\datasets\siemens\annotations 
    
    python3 dataset/convert_coco.py /mnt/c/Users/z004c84d/Documents/DEV/detectron2/dataset/siemens/train/images --output "/mnt/c/Users/z004c84d/Documents/DEV/detectron2/dataset/siemens/train/train2021.json"


        python3 dataset/convert_coco.py /mnt/c/Users/z004c84d/Documents/DEV/detectron2/dataset/siemens/val/images --output "/mnt/c/Users/z004c84d/Documents/DEV/detectron2/dataset/siemens/val/val2021.json"
```


COCO Dataset structure in project

```
└── coco
    ├── annotations
    │   ├── instances_minival2014_100.json
    │   ├── instances_train2017.json
    │   ├── instances_val2017_100.json
    │   ├── person_keypoints_minival2014_100.json
    │   └── person_keypoints_val2017_100.json
    ├── train2017
    │   ├──  00XXXXXXXXXX.jpg
    │   ├──  00XXXXXXXXXX.jpg
    │   ├──  00......
    ├── val2017
    │   ├──  00XXXXXXXXXX.jpg
    │   ├──  00XXXXXXXXXX.jpg
    │   ├──  00......
```
