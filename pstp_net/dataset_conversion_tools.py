# Some tools necessary to convert a perception dataset to mavqa format #
import json
import os


def transform_format(perception_data, global_idx):
    target_data = []
    for video_id, video_info in perception_data.items():
        mc_questions = video_info["mc_question"]
        metadata = video_info['metadata']
        for question in mc_questions:
            transformed_question = {
                "question_id": global_idx,
                "video_id": video_id,
                "question_content": question["question"],
                "templ_values": "[]",
                "anser": question["options"][question["answer_id"]] if 'answer_id' in question else None,
                "answer_id": question["answer_id"] if 'answer_id' in question else None,
                "options": question["options"],
                "split": metadata['split']
            }

            target_data.append(transformed_question)
            global_idx += 1

    return target_data


def process_json_files(input_files, output_file):
    # Process each JSON file
    global_idx = 0
    final_data = []
    for file in input_files:
        with open(file, 'r') as fin:
            perception_data = json.load(fin)
        for f in transform_format(perception_data, global_idx):
            final_data.append(f)
    # Massive output #
    with open(output_file, 'w') as fout:
        json.dump(fout, output_file)


# Example usage
if __name__ == '__main__':
    input_files = ["../data/mc_question_train.json", "../data/mc_question_test.json", "../data/mc_question_valid.json"]
    output_file = "./dataset/split_que_id/perception.json"

    process_json_files(input_files, output_file)
