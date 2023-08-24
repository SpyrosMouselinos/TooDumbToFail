import json

GLOBAL_IDX = 0

def transform_format(perception_data):
    global GLOBAL_IDX
    target_data = []
    for video_id, video_info in perception_data.items():
        mc_questions = video_info["mc_question"]
        metadata = video_info['metadata']
        for question in mc_questions:
            transformed_question = {
                "question_id": GLOBAL_IDX,
                "video_id": video_id,
                "question_content": question["question"],
                "templ_values": "[]",
                "anser": question["options"][question["answer_id"]] if 'answer_id' in question else None,
                "answer_id": question["answer_id"] if 'answer_id' in question else None,
                "options": question["options"],
                "split": metadata['split']
            }

            target_data.append(transformed_question)
            GLOBAL_IDX += 1

    return target_data


def process_json_files(input_files, output_files):
    # Process each JSON file
    final_data = []
    for gg, ff in zip(input_files, output_files):
        if gg is not None:
            this_file_data = []
            with open(gg, 'r') as fin:
                perception_data = json.load(fin)
            for f in transform_format(perception_data):
                final_data.append(f)
                this_file_data.append(f)
            with open(ff, 'w') as fout:
                json.dump(this_file_data, fout)
        else:
            with open(ff, 'w') as fout:
                json.dump(final_data, fout)

def check_longest(file):
    with open(file, 'r') as fin:
        data = json.load(fin)

    largest_q_len = 0
    largest_a_len = 0
    for k in data:
        a_len = max([len(f) for f in k['options']])
        q_len = len(k['question_content'])
        if a_len > largest_a_len:
            largest_a_len = a_len
        if q_len > largest_q_len:
            largest_q_len = q_len

    print(largest_a_len)
    print(largest_q_len)
    return


# Example usage
if __name__ == '__main__':
    input_files = ["../data/mc_question_train.json","../data/mc_question_valid.json", "../data/mc_question_test.json", None]
    output_file = ["./dataset/split_que_id/perception_train.json","./dataset/split_que_id/perception_valid.json","./dataset/split_que_id/perception_test.json","./dataset/split_que_id/perception.json"]

    process_json_files(input_files, output_file)
    #check_longest(output_file)