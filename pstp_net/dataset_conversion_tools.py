# Some tools necessary to convert a perception dataset to mavqa format #
import json
import os


def transform_format(original_data):
    target_data = []
    global_idx = 0
    for video_id, video_info in original_data.items():
        mc_questions = video_info["mc_question"]

        for question in mc_questions:
            transformed_question = {
                "question_id": global_idx,
                "video_id": video_id,
                "question_content": question["question"],
                "templ_values": "[]",
                "anser": question["options"][question["answer_id"]],
                "answer_id": question["answer_id"],
                "options": question["options"]
            }

            target_data.append(transformed_question)
            global_idx += 1

    return target_data



def process_json_files(input_folder, output_folder):
    # Ensure output folder exists, create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each JSON file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_filepath = os.path.join(input_folder, filename)
            output_filename = f"transformed_{filename}"
            output_filepath = os.path.join(output_folder, output_filename)

            with open(input_filepath, 'r') as input_file:
                original_data = json.load(input_file)

            target_data = transform_format(original_data)

            with open(output_filepath, 'w') as output_file:
                json.dump(target_data, output_file, indent=2)

            print(f"Transformed '{filename}' and saved as '{output_filename}'.")


# Example usage
input_folder = "/path/to/input/json/files"
output_folder = "/path/to/output/json/files"

process_json_files(input_folder, output_folder)
