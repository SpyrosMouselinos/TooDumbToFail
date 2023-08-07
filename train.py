import json
import pandas
import seaborn as sns

sns.set_style('darkgrid')


def open_json(file):
    with open(file, 'r') as fin:
        data = json.load(fin)
    return data


PATH = 'data/mc_question_'
TRAIN_PATH = PATH + 'train.json'
VALID_PATH = PATH + 'valid.json'
TEST_PATH = PATH + 'test.json'
train_data = open_json(TRAIN_PATH)
valid_data = open_json(VALID_PATH)
test_data = open_json(TEST_PATH)

# VIDEO OVERLAP #

train_video_names = set(train_data.keys())

valid_video_names = set(valid_data.keys())

test_video_names = set(test_data.keys())

train_valid_overlap = train_video_names.intersection(valid_video_names)
train_test_overlap = train_video_names.intersection(test_video_names)
valid_test_overlap = valid_video_names.intersection(test_video_names)

print(f"Train-Valid Video Overlap: {len(train_valid_overlap)}")
print(f"Train-Test Video Overlap: {len(train_test_overlap)}")
print(f"Valid-Test Video Overlap: {len(valid_test_overlap)}")

# QUESTION OVERLAP #
train_q = []
train_o = []

valid_q = []
valid_o = []

test_q = []
test_o = []


def get_qo(data_split, q_list, o_list):
    for _, v in data_split.items():
        for i in v['mc_question']:
            q_list.append(i['question'])
            for j in i['options']:
                o_list.append(j)


get_qo(train_data, train_q, train_o)
get_qo(valid_data, valid_q, valid_o)
get_qo(test_data, test_q, test_o)

train_q = set(train_q)
train_o = set(train_o)
valid_q = set(valid_q)
valid_o = set(valid_o)
test_q = set(test_q)
test_o = set(test_o)

test_train_overlap = test_q.difference(train_q)
test_valid_overlap = test_q.difference(valid_q)

print(f"Questions in Test but not in Train: {len(test_train_overlap)}")
print(f"Questions in Test but not in Valid: {len(test_valid_overlap)}")

test_train_overlap = test_o.difference(train_o)
test_valid_overlap = test_o.difference(valid_o)

print(f"Options in Test but not in Train: {len(test_train_overlap)}")
print(f"Options in Test but not in Valid: {len(test_valid_overlap)}")

print((len(test_o) - len(test_valid_overlap)) / len(test_o))
#### Lets see them ###
# for i in list(test_valid_overlap):
#     print(i)
#     print('\n')