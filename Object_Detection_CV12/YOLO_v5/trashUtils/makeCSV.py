import numpy as np
import pandas as pd
import tqdm
import os 
import glob
import configparser
config = configparser.ConfigParser()
config.read('config.ini',encoding='UTF-8')
inference_path = config['path']['inference_path']
save_path = config['path']['csv_save_path']

file_names=[]
for i in range(4871):
    file_names.append(('0'* (4-len(str(i))) +str(i)))
file_id = list(map(lambda x : 'test/'+x+'.jpg',file_names))
output=[ [[] for _ in range(10)] for i in range(4871) ]
idx=0


for file_name in tqdm.tqdm(file_names):
    
    with open(os.path.join(inference_path,file_name+'.txt'),'r') as f :
        result=''
        category_box=[[] for _ in range(10)]
        boxes=[]
        for line in f:
            box=list(map(float,line.split(' ')))
            category_id=int(box[0])
            xc,yc,w,h = map(lambda x :x*1024,box[1:-1])
            confidence = box[-1]
            xmin= max(0,xc - int(np.round(w/2)))
            ymin = max(0,yc - int(np.round(h/2)))
            xmax= min(xc + int(np.round(w/2)),1024)
            ymax= min(yc + int(np.round(h/2)),1024)
            box_list = [category_id,confidence,xmin,ymin,xmax,ymax]
            boxes.append(box_list)
 
        nms_box =boxes
        for box in nms_box:
            category_id = box[0]
            confidence = box[1]
            x1,y1,x2,y2 = box[2:]
            xmin= x1
            ymin = y1 
            box_list = [category_id, x1,y1,x2,y2, confidence]
            output[idx][box_list[0]].append([box_list[1],box_list[2],box_list[3],box_list[4],box_list[5]])
        idx+=1

prediction_strings = []

class_num = 10
for i, out in enumerate(tqdm.tqdm(output)):
    prediction_string = ''
    for j in range(class_num):
        for o in out[j]:
            prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                o[2]) + ' ' + str(o[3]) + ' '
        
    prediction_strings.append(prediction_string)
submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_id
submission.to_csv(save_path, index=None)
submission.head()
            
