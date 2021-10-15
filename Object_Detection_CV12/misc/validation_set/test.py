import json
import random

train_dict = {}
dict_path = '/opt/ml/validation_set/dataset/train.json'
img_path = '/opt/ml/validation_set/dataset/train/'
save_path = '/opt/ml/validation_set/dataset/validation.json'

with open(dict_path, 'r') as f:
    train_dict = json.load(f)

"""
train_dict = {
    'info': { ... },
    'licenses': [ ... ],
    'categories': [ ... ],
    'images': [
        {
            'width': ~,
            'height': ~,
            'file_name': ~,
            'id': ~,
        },
    ],
    'annotations': [
        {
            'image_id': ~,
            'category_id': ~,
            'area': ~,
            'bbox': [x, y, w, h],
        },
    ]
}
"""

bbox = [[] for _ in range(4883)] # bbox[i] : image_i의 bbox 정보, { 'pos': [x, y, w, h], 'area': ~, 'category': ~ }
bbox_cnt = [0] * 4883 # bbox_cnt[i] : image_i의 bbox 개수
category = [0] * 10 # category[i] : 전체 image의 i번 카테고리인 bbox 개수
area_sorted = [] # bbox들의 area

for i in train_dict['annotations']:
    bbox[i['image_id']].append({ 'pos': i['bbox'], 'area': i['area'], 'category': i['category_id'] })
    bbox_cnt[i['image_id']] += 1
    category[i['category_id']] += 1
    area_sorted.append(i['area'])

area_sorted.sort()
bbox_cnt_sorted = sorted(bbox_cnt)

def GetImageID():
    image_id = [0] * 4883
    for idx in range(4883):
        cnt = [0] * 10
        for bbox_info in bbox[idx]:
            cnt[bbox_info['category']] += 1
        v, mx = [], max(cnt)
        for i in range(10):
            if cnt[i] == mx: v.append(i)
        image_id[idx] = v[random.randint(0, len(v) - 1)]
    return image_id

img_to_id = GetImageID()
id_to_img = [[] for _ in range(10)]

for idx in range(4883):
    id_to_img[img_to_id[idx]].append(idx)

print("univ distribution")
for i in range(10):
    print(f'class : {i}, percent : {len(id_to_img[i]) / 48.83:.4f}%')

bucket = [[] for _ in range(5)]
for imgs in id_to_img:
    sz = len(imgs)
    random.shuffle(imgs)
    for i in range(sz):
        bucket[i % 5].append(imgs[i])
    random.shuffle(bucket)

for i in range(5):
    bucket[i].sort()

d = { 'bucket': bucket }
with open('/opt/ml/validation_set/validation_idx.json', 'w') as f:
    json.dump(d, f)