## ultralytics yolov5 패키지
https://github.com/ultralytics/yolov5

### yoloDataSet생성. 
trashUtil/createDatase.py


## train 
```
train.py --weights yolov5x6.pt --data ./content/trashl/trash.yaml --hyp data/hyps/hyp.scratch-high.yaml --batch-size 4 --imgsz 1024 --project ./ultra_workdir/ --name v.6-last --noval --save-period 50
```
