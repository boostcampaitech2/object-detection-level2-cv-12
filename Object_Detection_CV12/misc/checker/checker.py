import csv

# csv file path
csv_path = "/opt/ml/detection/mmdetection/work_dirs/swin_1024_adam_pseudo/submission_epoch_14.csv"
csv_reader = csv.reader(open(csv_path, 'r'))
data = list(csv_reader)

# error checker
def Check(v):
    if v[0] < 0 or 10 <= v[0]: return 4
    if v[1] < 0.0 or 1.0 < v[1]: return 3
    if v[2] > v[4] or v[3] > v[5]: return 1
    for i in range(2, 6):
        if v[i] < 0.0 or 1024.0 < v[i]: return 2
    return 0

# info
error_cnt = 0
category_cnt = [0] * 10
min_confidence = 1e9
bbox_cnt = [0] * (len(data))

# check data
for i in range(1, len(data)):
    v = list(map(float, data[i][0].split()))
    for j in range(0, len(v), 6):
        cur = v[j:j+6]
        res = Check(cur)
        if res: error_cnt += 1

        if res == 1: print(f"x1 > x2 or y1 > y2 error in image no.{i - 1}")
        elif res == 2: print(f"out of bound error in image no.{i - 1}, {cur[2]}, {cur[3]}, {cur[4]}, {cur[5]}")
        elif res == 3: print(f"confidence > 1.0 or confidence < 0.0 error in image no.{i - 1}")
        elif res == 4: print(f"label out of index error in image no.{i - 1}")

        category_cnt[int(cur[0])] += 1
        min_confidence = min(min_confidence, cur[1])
        bbox_cnt[i] += 1

# print error result
print()
if error_cnt: print(f"total {error_cnt} error found!")
else: print("no error founded :)")

# print data result
print(f"category_cnt : ", category_cnt)
print(f"min_confidence : {min_confidence:0.6f}")
print(f"max_bbox_cnt : {max(bbox_cnt)}")