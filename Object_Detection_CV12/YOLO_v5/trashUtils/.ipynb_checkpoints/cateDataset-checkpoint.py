# https://github.com/alexmihalyk23/COCO2YOLO.git
import json
import os
import shutil

import configparser


config = configparser.ConfigParser()
config['tgt_path']= '../dataset/trash/'
config['src_img_path'] = '/opt/ml/detection/dataset/'
config['train_json_path']='/opt/ml/detection/dataset/train.json'
config['val_json_path']='/opt/ml/detection/dataset/train.json'
tgt_path='../dataset/trash/'
with open('config.ini', 'w', encoding='utf-8') as configfile:
    config.write(configfile)

class COCO2YOLO:
  # 소스 이미지 디렉토리와 Json annotation 파일, 타겟 이미지 디렉토리, 타겟 annotation 디렉토리를 생성자로 입력 받음. 
  def __init__(self, src_img_dir, json_file, tgt_img_dir, tgt_anno_dir):
    self.json_file = json_file
    self.src_img_dir = src_img_dir
    self.tgt_img_dir = tgt_img_dir
    self.tgt_anno_dir = tgt_anno_dir
    # json 파일과 타겟 디렉토리가 존재하는지 확인하고, 디렉토리의 경우는 없으면 생성. 
    self._check_file_and_dir(json_file, tgt_img_dir, tgt_anno_dir)
    # json 파일을 메모리로 로딩. 
    self.labels = json.load(open(json_file, 'r', encoding='utf-8'))
    # category id와 이름을 매핑하지만, 실제 class id는 이를 적용하지 않고 별도 적용. 
    self.coco_id_name_map = self._categories()
    self.coco_name_list = list(self.coco_id_name_map.values())
    print("total images", len(self.labels['images']))
    print("total categories", len(self.labels['categories']))
    print("total labels", len(self.labels['annotations']))
  
  # json 파일과 타겟 디렉토리가 존재하는지 확인하고, 디렉토리의 경우는 없으면 생성. 
  def _check_file_and_dir(self, file_path, tgt_img_dir, tgt_anno_dir):
    if not os.path.exists(file_path):
        raise ValueError("file not found")
    if not os.path.exists(tgt_img_dir):
        os.makedirs(tgt_img_dir)
    if not os.path.exists(tgt_anno_dir):
        os.makedirs(tgt_anno_dir)

  # category id와 이름을 매핑하지만, 추후에 class 명만 활용. 
  def _categories(self):
    categories = {}
    for cls in self.labels['categories']:
        categories[cls['id']] = cls['name']
    return categories
  
  # annotation에서 모든 image의 파일명(절대 경로 아님)과 width, height 정보 저장. 
  def _load_images_info(self):
    images_info = {}
    for image in self.labels['images']:
        id = image['id']
        file_name = image['file_name']
        if file_name.find('\\') > -1:
            file_name = file_name[file_name.index('\\')+1:]
        w = image['width']
        h = image['height']
  
        images_info[id] = (file_name, w, h)

    return images_info

  # ms-coco의 bbox annotation은 yolo format으로 변환. 좌상단 x, y좌표, width, height 기반을 정규화된 center x,y 와 width, height로 변환. 
  def _bbox_2_yolo(self, bbox, img_w, img_h):
    # ms-coco는 좌상단 x, y좌표, width, height
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    # center x좌표는 좌상단 x좌표에서 width의 절반을 더함. center y좌표는 좌상단 y좌표에서 height의 절반을 더함.  
    centerx = bbox[0] + w / 2
    centery = bbox[1] + h / 2
    # centerx, centery, width, height를 이미지의 width/height로 정규화. 
    dw = 1 / img_w
    dh = 1 / img_h
    centerx *= dw
    w *= dw
    centery *= dh
    h *= dh
    return centerx, centery, w, h
  
  # image와 annotation 정보를 기반으로 image명과 yolo annotation 정보 가공. 
  # 개별 image당 하나의 annotation 정보를 가지도록 변환. 
  def _convert_anno(self, images_info):
    anno_dict = dict()
    for anno in self.labels['annotations']:
      bbox = anno['bbox']
      image_id = anno['image_id']
      category_id = anno['category_id']

      image_info = images_info.get(image_id)
      image_name = image_info[0]
      img_w = image_info[1]
      img_h = image_info[2]
      yolo_box = self._bbox_2_yolo(bbox, img_w, img_h)

      anno_info = (image_name, category_id, yolo_box)
      anno_infos = anno_dict.get(image_id)
      if not anno_infos:
        anno_dict[image_id] = [anno_info]
      else:
        anno_infos.append(anno_info)
        anno_dict[image_id] = anno_infos
    return anno_dict

  # class 명을 파일로 저장하는 로직. 사용하지 않음. 
  def save_classes(self):
    sorted_classes = list(map(lambda x: x['name'], sorted(self.labels['categories'], key=lambda x: x['id'])))
    print('coco names', sorted_classes)
    with open('coco.names', 'w', encoding='utf-8') as f:
      for cls in sorted_classes:
          f.write(cls + '\n')
    f.close()
  # _convert_anno(images_info)로 만들어진 anno 정보를 개별 yolo anno txt 파일로 생성하는 로직. 
  # coco2yolo()에서 anno_dict = self._convert_anno(images_info)로 만들어진 anno_dict를 _save_txt()에 입력하여 파일 생성
  def _save_txt(self, anno_dict):
    # 개별 image별로 소스 image는 타겟이미지 디렉토리로 복사하고, 개별 annotation을 타겟 anno 디렉토리로 생성. 
    for k, v in anno_dict.items():
      # 소스와 타겟 파일의 절대 경로 생성. 
      src_img_filename = os.path.join(self.src_img_dir, v[0][0])
      tgt_anno_filename = os.path.join(self.tgt_anno_dir,v[0][0].split("/")[-1].split(".")[0] + ".txt")
      #print('source image filename:', src_img_filename, 'target anno filename:', tgt_anno_filename)
      # 이미지 파일의 경우 타겟 디렉토리로 단순 복사. 
      shutil.copy(src_img_filename, self.tgt_img_dir)
      # 타겟 annotation 출력 파일명으로 classid, bbox 좌표를 object 별로 생성. 
      with open(tgt_anno_filename, 'w', encoding='utf-8') as f:
        #print(k, v)
        # 여러개의 object 별로 classid와 bbox 좌표를 생성. 
        for obj in v:
          cat_name = self.coco_id_name_map.get(obj[1])
          # category_id는 class 명에 따라 0부터 순차적으로 부여. 
          category_id = self.coco_name_list.index(cat_name)
          #print('cat_name:', cat_name, 'category_id:', category_id)
          box = ['{:.6f}'.format(x) for x in obj[2]]
          box = ' '.join(box)
          line = str(category_id) + ' ' + box
          f.write(line + '\n')

  # ms-coco를 yolo format으로 변환. 
  def coco2yolo(self):
    print("loading image info...")
    images_info = self._load_images_info()
    print("loading done, total images", len(images_info))

    print("start converting...")
    anno_dict = self._convert_anno(images_info)
    print("converting done, total labels", len(anno_dict))

    print("saving txt file...")
    self._save_txt(anno_dict)
    print("saving done")

    
os.makedirs(tgt_path)
os.makedirs(os.path.join(tgt_path,'images'))
os.makedirs(os.path.join(tgt_path'/iamges/train')
os.makedirs(os.path.join(tgt_path,'images/val'))
os.makedirs(os.path.join(tgt_path,'iamges/test'))
os.makedirs(os.path.join(tgt_path,'labels'))
os.makedirs(os.path.join(tgt_path,'labels/train'))
os.makedirs(os.path.join(tgt_path,'labels/val'))
os.makedirs(os.path.join(tgt_path,'labels/test'))

            
train_yolo_converter = COCO2YOLO(src_img_dir='/opt/ml/detection/dataset/', json_file='../../../dataset/train.json',
                                 tgt_img_dir=os.path.join(tgt_path,'/iamges/train'), tgt_anno_dir=os.path.join(tgt_path,'labels/train'))
train_yolo_converter.coco2yolo()
val_yolo_converter = COCO2YOLO(src_img_dir='/opt/ml/detection/dataset/', json_file='../../../dataset/train.json',
                                 tgt_img_dir=os.path.join(tgt_path,'images/val'), tgt_anno_dir=os.path.join(tgt_path,'labels/val'))
val_yolo_converter.coco2yolo()
shutil.copy('./train.yaml', tgt_path)