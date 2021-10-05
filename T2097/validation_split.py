import json
from copy import deepcopy

train_dict = {}
validation_dict = {}

validation_idx_path = '/opt/ml/detection/dataset_with_validation/validation_idx.json'
dict_path = '/opt/ml/detection/dataset_with_validation/train.json'
save_train_path = '/opt/ml/detection/dataset_with_validation/new_train.json'
save_validation_path = '/opt/ml/detection/dataset_with_validation/validation.json'

with open(dict_path, 'r') as f:
    train_dict = json.load(f)

with open(validation_idx_path, 'r') as f:
    validation_dict = json.load(f)

new_train_dict = { 'info': train_dict['info'], 'licenses': train_dict['licenses'], 'images': [], 'categories': train_dict['categories'], 'annotations': [] }
new_validation_dict = { 'info': train_dict['info'], 'licenses': train_dict['licenses'], 'images': [], 'categories': train_dict['categories'], 'annotations': [] }

for i in train_dict['images']:
    # print(type(i['id']), i['id'], i['id'] in validation_idx)
    if i['id'] in validation_dict['bucket'][0]:
        new_validation_dict['images'].append(i)
    else:
        new_train_dict['images'].append(i)

for i in train_dict['annotations']:
    if i['image_id'] in validation_dict['bucket'][0]:
        new_validation_dict['annotations'].append(i)
    else:
        new_train_dict['annotations'].append(i)

with open(save_train_path, 'w') as f:
    json.dump(new_train_dict, f, indent=4)

with open(save_validation_path, 'w') as f:
    json.dump(new_validation_dict, f, indent=4)