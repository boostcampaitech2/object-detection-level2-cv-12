import json
import csv
import shutil
import tqdm

#
sumissionDir='./ensemble.csv'
trainJsonDir='./train.json'
testJosonDir='./dataset/test.json'
pseudoCsv = csv.reader(open('./ensemble.csv','r'))
train_dict = json.load(open('./train.json','r'))
test_dict = json.load(open('./dataset/test.json','r'))
outputDir='./ultralytics/yolov5/content/trashp/labels/train/pesudo.json'
confidence_thres=0.7
#num lableing
numLabel=1000

numLine=0
numAnn=0

# train set 개수 train set annotation 개수
numTrainimage=len(train_dict['images'])
numTrainAnno=len(train_dict['annotations'])

new_train_dict = { 'info': train_dict['info'], 'licenses': train_dict['licenses'], 'images': train_dict['images'], 'categories': train_dict['categories'], 'annotations': train_dict['annotations'] }
for line in tqdm.tqdm(pseudoCsv):
    if numLine>numLabel:
        break
    if numLine>=1:
        splitedLine = line[0].split(' ')
        image_id = line[1]
        boxes = [list(map(float,splitedLine[i*6:i*6+6])) for i in range(len(splitedLine)//6)]
        # nms_boxes= nms(boxes,0.65,0.8)
        nms_boxes = boxes
        for box in nms_boxes:

            category_id=int(box[0])
            confidence, x1,y1,x2,y2=map(float,box[1:])
            w = int(x2-x1)
            h = int(y2-y1)
            if confidence>confidence_thres:
                annodict = {'image_id' : numLine-2+numTrainimage, 'category_id':category_id, 'area':w*h, 'bbox':[x1,y1,w,h],'iscrowd':0 ,'id':numTrainAnno+numAnn}
                new_train_dict['annotations'].append(annodict)
                numAnn+=1
        test_dict['images'][numLine-1]['id']=numLine+numTrainimage-2
        new_train_dict['images'].append(test_dict['images'][numLine-1])
        
        
    numLine+=1
print(numLine-1+numTrainimage-1)
print(len(new_train_dict['images']))
print(new_train_dict['annotations'][-1]['image_id'])
for idx,ann in enumerate(new_train_dict['annotations']):
    if idx !=int(ann['id']):
        print(ann['id'])
        break
with open(outputDir, 'w', encoding='utf-8') as f:

    json.dump(new_train_dict, f, indent=4)   
    