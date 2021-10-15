import json
import random
import copy
import math

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

"""
Goal. split img(4883) to 5 pieces

1. bbox 개수 분포
2. 카테고리 분포
3. area 분포
"""

def GetSample(v, sz):
    ret = random.sample(v, sz)
    return sorted(ret)

def GetDiff(v1, v2):
    v1.sort()
    v2.sort()
    ret = 0
    for i in range(len(v1)):
        ret += (v1[i] - v2[i]) * (v1[i] - v2[i])
    return ret / len(v1) / 100000000

def GetBboxScore(bucket):
    ret = 0
    for i in bucket:
        univ = GetSample(bbox_cnt, len(i))
        samp = sorted([bbox_cnt[j] for j in i])
        ret += GetDiff(univ, samp)
    return ret

def GetCategoryScore(bucket):
    ret = 0
    for i in bucket:
        samp = [0] * 10
        for idx in i:
            for bbox_info in bbox[idx]:
                samp[bbox_info['category']] += 1
        ret += GetDiff(category, samp)
    return ret

def GetAreaScore(bucket):
    ret = 0
    for i in bucket:
        samp = []
        for idx in i:
            for bbox_info in bbox[idx]:
                samp.append(bbox_info['area'])
        univ = GetSample(area_sorted, len(samp))
        ret += GetDiff(univ, samp)
    return ret

def GetScore(bucket):
    bbox_score = GetBboxScore(bucket)
    category_score = GetCategoryScore(bucket)
    area_score = GetAreaScore(bucket)
    return bbox_score + category_score + area_score

def SimulatedAnnealing(epochs: int, T: float, lr: float):
    bucket = [[] for _ in range(5)]
    for i in range(4883):
        bucket[random.randint(0, 4)].append(i)
    
    bucket_best = copy.deepcopy(bucket)
    score = GetScore(bucket)
    score_best = score

    for epoch in range(1, epochs + 1):
        t1 = random.randint(0, 4)
        t2 = random.randint(0, 4)
        while t1 == t2: t2 = random.randint(0, 4)
        t1_idx = random.randint(0, len(bucket[t1]) - 1)
        t2_idx = random.randint(0, len(bucket[t2]) - 1)

        bucket_nxt = copy.deepcopy(bucket)
        bucket_nxt[t1][t1_idx], bucket_nxt[t2][t2_idx] = bucket[t2][t2_idx], bucket[t1][t1_idx]
        
        nxt_score = GetScore(bucket_nxt)
        # print(score, nxt_score)
        p = math.exp((score - nxt_score) / T)
        if p > random.random():
             bucket = copy.deepcopy(bucket_nxt)
             score = nxt_score
        
        if score_best < score:
            bucket_best = copy.deepcopy(bucket)
            score_best = score
            
        T *= lr
        if epoch % 10 == 0: print(f'epoch : {epoch}, T : {T}, score : {score}')
    
    return bucket_best

validation_set = SimulatedAnnealing(1000, 1.0, 0.999)
for i in validation_set:
    print(i)