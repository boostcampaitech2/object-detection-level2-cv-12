## ultralytics yolov5 패키지
https://github.com/ultralytics/yolov5

### yoloDataSet생성. 
trashUtil/createDatase.py


### train 
```
train.py --weights yolov5x6.pt --data ./dataset/trash/trash.yaml --hyp data/hyps/hyp.scratch-high.yaml --batch-size 4 --imgsz 1024 --project ./work_dir/ --name yolo --save-period 50
```

### inference 
```
val.py --weights ./workdir/yolo/weights/best.pt --augment --project ./output/ --name yolo --save-txt --save-conf
```

CV Score 0.601 LB Score 0.594
