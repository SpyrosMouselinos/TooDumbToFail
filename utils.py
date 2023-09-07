import json
import os
import zipfile
import moviepy.editor as mvp
import requests
from typing import Dict, Any
import decord as de
from nltk import pos_tag, word_tokenize
from tqdm import tqdm
import numpy as np

ctx = de.cpu(0)

AREA = ['physics', 'semantics', 'abstraction', 'memory']
REASONING = ['descriptive', 'counterfactual', 'explanatory', 'predictive']
TAG = ['motion', 'place recognition', 'action counting', 'spatial relations',
       'object recognition', 'action recognition', 'distractor action',
       'distractor object', 'state recognition', 'object counting',
       'change detection', 'sequencing', 'task completion',
       'adversarial action', 'collision', 'object attributes',
       'pattern breaking', 'feature matching', 'pattern discovery',
       'part recognition', 'material', 'event recall', 'containment',
       'occlusion', 'solidity', 'object permanence', 'colour recognition',
       'conservation', 'quantity', 'event counting', 'language',
       'visual discrimination', 'general knowledge', 'stability']

TAG_AREA = {
    'change detection': 'memory',
    'event recall': 'memory',
    'sequencing': 'memory',
    'visual discrimination': 'memory',
    'action counting': 'abstraction',
    'event counting': 'abstraction',
    'feature matching': 'abstraction',
    'object counting': 'abstraction',
    'pattern breaking': 'abstraction',
    'pattern discovery': 'abstraction',
    'collision': 'physics',
    'conservation': 'physics',
    'containment': 'physics',
    'material': 'physics',
    'motion': 'physics',
    'occlusion': 'physics',
    'object attributes': 'physics',
    'object permanence': 'physics',
    'quantity': 'physics',
    'solidity': 'physics',
    'spatial relations': 'physics',
    'stability': 'physics',
    'action recognition': 'semantics',
    'adversarial action': 'semantics',
    'colour recognition': 'semantics',
    'distractor action': 'semantics',
    'distractor object': 'semantics',
    'general knowledge': 'semantics',
    'language': 'semantics',
    'object recognition': 'semantics',
    'part recognition': 'semantics',
    'place recognition': 'semantics',
    'state recognition': 'semantics',
    'task completion': 'semantics',
}

CAT = AREA + REASONING + TAG
DATA_PATH = data_path = './data/'
CLIP_Q_DATA_PATH = DATA_PATH + '/PERCEPTION/clip_word'
CLIP_A_DATA_PATH = DATA_PATH + '/PERCEPTION/clip_word_ans'
TRAIN_DATA_PATH = train_data_path = DATA_PATH + 'train/'
VALID_DATA_PATH = valid_data_path = DATA_PATH + 'valid/'
ANSWER_DATA_PATH = answer_data_path = DATA_PATH + 'answer_keysword_overlap.json'  # Use gather answers to make this
OCR_RELEVANT_VIDEO_IDS_PATH = ocr_relevant_video_ids_path = DATA_PATH + 'ocr_rel_videos.json'  # Use gather ocr videos to make this


def download_and_unzip(url: str, destination: str):
    """Downloads and unzips a .zip file to a destination.

    Downloads a file from the specified URL, saves it to the destination
    directory, and then extracts its contents.

    If the file is larger than 1GB, it will be downloaded in chunks,
    and the download progress will be displayed.

    Args:
      url (str): The URL of the file to download.
      destination (str): The destination directory to save the file and
        extract its contents.
    """
    if not os.path.exists(destination):
        os.makedirs(destination)

    filename = url.split('/')[-1]
    file_path = os.path.join(destination, filename)

    if os.path.exists(file_path):
        print(f'{filename} already exists. Skipping download.')
        return

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    gb = 1024 * 1024 * 1024

    if total_size / gb > 1:
        print(f'{filename} is larger than 1GB, downloading in chunks')
        chunk_flag = True
        chunk_size = int(total_size / 100)
    else:
        chunk_flag = False
        chunk_size = total_size

    with open(file_path, 'wb') as file:
        for chunk_idx, chunk in enumerate(
                response.iter_content(chunk_size=chunk_size)):
            if chunk:
                if chunk_flag:
                    print(f"""{chunk_idx}% downloading
          {round((chunk_idx * chunk_size) / gb, 1)}GB
          / {round(total_size / gb, 1)}GB""")
                file.write(chunk)
    print(f"'{filename}' downloaded successfully.")

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination)
    print(f"'{filename}' extracted successfully.")

    os.remove(file_path)


def load_db_json(db_file: str) -> Dict[str, Any]:
    """Loads a JSON file as a dictionary.

    Args:
      db_file (str): Path to the JSON file.

    Returns:
      Dict: Loaded JSON data as a dictionary.

    Raises:
      FileNotFoundError: If the specified file doesn't exist.
      TypeError: If the JSON file is not formatted as a dictionary.
    """
    if not os.path.isfile(db_file):
        raise FileNotFoundError(f'No such file: {db_file}')

    with open(db_file, 'r') as f:
        db_file_dict = json.load(f)
        if not isinstance(db_file_dict, dict):
            raise TypeError('JSON file is not formatted as a dictionary.')
        return db_file_dict


def gather_answers(db_files):
    """
    Generates ANSWER_DATA_PATH
    """
    if isinstance(db_files, str):
        db_files = [db_files]
    else:
        pass
    global_list = []
    for db in db_files:
        dict = load_db_json(db)
        for q, i in dict.items():
            for qas in i['mc_question']:
                for answer in qas['options']:
                    global_list.append(answer)

    # Sort Alphabetically #
    global_list = list(set(global_list))
    global_list.sort()

    MAX_ITEMS = len(global_list)
    new_dict = {}
    for index, item in enumerate(global_list):
        one_hot = [0] * MAX_ITEMS
        one_hot[index] = 1
        new_dict.update({item: {'int_index': index, 'one_hot_index': one_hot}})

    with open(ANSWER_DATA_PATH, 'w') as fout:
        json.dump(new_dict, fout)
    return


def gather_ocr_videos(db_files):
    """
    Generates OCR_RELEVANT_VIDEO_IDS_PATH
    """
    if isinstance(db_files, str):
        db_files = [db_files]
    else:
        pass
    marked_for_ocr = {}
    for db in db_files:
        dict = load_db_json(db)
        for q, i in dict.items():
            for qas in i['mc_question']:
                if 'letter' in qas['question'] or 'letters' in qas['question']:
                    if q not in marked_for_ocr:
                        marked_for_ocr.update({q: [qas['question']]})
                    else:
                        marked_for_ocr[q].append(qas['question'])
    with open(OCR_RELEVANT_VIDEO_IDS_PATH, 'w') as fout:
        json.dump(marked_for_ocr, fout)
    return marked_for_ocr


def load_mp4_to_frames(filename: str, indices=None, resize_to=None) -> np.array:
    """Loads an MP4 video file and returns its frames as a NumPy array.

    Args:
      filename (str): Path to the MP4 video file.

    Returns:
      np.array: Frames of the video as a NumPy array.
    """

    vr = de.VideoReader(filename, ctx=ctx, width=resize_to, height=resize_to)
    if indices is None:
        indices = list(range(len(vr)))
    frames = vr.get_batch(indices).asnumpy()
    return frames


def load_mp4_to_audioframes(filename: str, indices=None) -> np.array:
    """Loads an MP4 video file and returns its audio frames as a NumPy array.

    Args:
      filename (str): Path to the MP4 video file.

    Returns:
      np.array: Frames of the video as a NumPy array.
    """

    ar = de.AudioReader(filename, ctx=ctx, sample_rate=16000, mono=True)
    if indices is None:
        indices = list(range(len(ar)))
    frames = ar.get_batch(indices).asnumpy()
    return frames


def load_mp4_to_ocr_frames(filename: str, indices=None, resize_to=None) -> np.array:
    return load_mp4_to_frames(filename=filename, indices=indices, resize_to=resize_to)


def get_video_frames(data_item: Dict[str, Any],
                     video_folder_path: str,
                     override_video_name: bool = False,
                     resize_to=None,
                     num_samples=16,
                     n_segments=1) -> np.array:
    """Loads frames of a video specified by an item dictionary.

    Assumes format of annotations used in the Perception Test Dataset.

    Args:
      data_item (Dict): Item from dataset containing metadata.
      video_folder_path (str): Path to the directory containing videos.

    Returns:
      np.array: Frames of the video as a NumPy array.
      :param override_video_name: Add one random video from the pre-existing ones
    """

    if override_video_name:
        video_file = os.path.join(video_folder_path,
                                  'video_874') + '.mp4'
    else:
        video_file = os.path.join(video_folder_path,
                                  data_item['metadata']['video_id']) + '.mp4'

    parts = []
    vr = de.VideoReader(video_file, ctx=ctx, width=resize_to, height=resize_to)
    t = len(vr)
    for i in range(n_segments):
        indices = np.linspace(i * (t // n_segments), (i + 1) * ((t // n_segments) - 2), num_samples)
        indices = np.clip(indices, i * (t // n_segments), (i + 1) * ((t // n_segments) - 2)).astype(int)
        parts.append(vr.get_batch(indices).asnumpy())
    return parts


def get_audio_frames(data_item: Dict[str, Any],
                     video_folder_path: str,
                     override_video_name: bool = False, num_samples=16, n_segments=1) -> np.array:
    """Loads audio frames of a video specified by an item dictionary.

    Assumes format of annotations used in the Perception Test Dataset.

    Args:
      data_item (Dict): Item from dataset containing metadata.
      video_folder_path (str): Path to the directory containing videos.

    Returns:
      np.array: Frames of the video as a NumPy array.
      :param override_video_name: Add one random video from the pre-existing ones
    """

    if override_video_name:
        video_file = os.path.join(video_folder_path,
                                  'video_8241') + '.mp4'
    else:
        video_file = os.path.join(video_folder_path,
                                  data_item['metadata']['video_id']) + '.mp4'

    parts = []

    # You can specify the desired sample rate and channel layout
    ar = de.AudioReader(video_file, ctx=ctx, sample_rate=16000, mono=True)
    t = ar.shape[1]
    indices = list(range(0, t))
    return ar.get_batch(indices=indices).asnumpy()


def get_ocr_frames(data_item: Dict[str, Any],
                   video_folder_path: str,
                   override_video_name: bool = False,
                   resize_to=None,
                   num_samples=16,
                   n_segments=1) -> np.array:
    return get_video_frames(data_item=data_item, video_folder_path=video_folder_path,
                            override_video_name=override_video_name, resize_to=resize_to, num_samples=num_samples,
                            n_segments=n_segments)


def get_qo_frames(data_item: Dict[str, Any],
                  q_folder_path: str,
                  a_folder_path: str,
                  override_video_name: bool = False, num_samples=16, n_segments=1) -> np.array:
    """
    Loads clip processed question / answers
    """

    if override_video_name:
        question_file = os.path.join(q_folder_path,
                                     '8241_sent') + '.npy'
        answer_file = os.path.join(a_folder_path,
                                   '8241_sent') + '.npy'
    else:
        question_file = os.path.join(q_folder_path,
                                     data_item['metadata']['video_id']) + '_sent.npy'
        answer_file = os.path.join(a_folder_path,
                                   data_item['metadata']['video_id']) + '_sent.npy'

    question_feat = np.load(question_file)
    answer_feat = np.load(answer_file)
    return question_feat, answer_feat


def atest_video(valid_db_dict, video_id='video_8241'):
    video_path = 'data/sample/videos/'
    video_item = valid_db_dict[video_id]
    video_path = os.path.join(video_path,
                              video_item['metadata']['video_id']) + '.mp4'
    mvp.ipython_display(video_path)

    print('Video ID: ', video_item['metadata']['video_id'])
    print('Video Length: ', video_item['metadata']['num_frames'])
    print('Video FPS: ', video_item['metadata']['frame_rate'])
    print('Video  Resolution: ', video_item['metadata']['resolution'])

    if video_item['mc_question']:
        for example_q in video_item['mc_question']:
            print('---------------------------------')
            print('Question: ', example_q['question'])
            print('Options: ', example_q['options'])
            print('Answer ID: ', example_q['answer_id'])
            print('Answer: ', example_q['options'][example_q['answer_id']])
            print('Question info: ')
            print('Reasoning: ', example_q['reasoning'])
            print('Tag: ', example_q['tag'])
            print('area: ', example_q['area'])
            print('---------------------------------')


def calc_top(answers_dict: Dict[str, Any],
             db_dict: Dict[str, Any], k=1) -> float:
    """Calculates the top-k accuracy and results for each category.

    Args:
      answers_dict (Dict): Dictionary containing the answers.
      db_dict (Dict): Dictionary containing the database.

    Returns:
      float: Top-k accuracy.

    Raises:
      KeyError: Raises error if the results contain a question ID that does not
        exist in the annotations.
      ValueError: Raises error if the answer ID is outside the expected range
        [0,2].
      ValueError: Raises error if the text of the answer in the results does not
        match the expected string answer in the annotations.
      ValueError: If answers are missing from the results.

    """
    expected_total = 0
    total_correct = 0
    total = 0

    for v in db_dict.values():
        expected_total += len(v['mc_question'])

    for vid_id, vid_answers in answers_dict.items():
        for answer_info in vid_answers:
            answer_id = answer_info['answer_id']
            question_idx = answer_info['id']

            try:
                ground_truth = db_dict[vid_id]['mc_question'][question_idx]
            except KeyError as exc:
                print(f'Unexpected question ID in {vid_id}.')
                continue
            if not isinstance(answer_id, list):
                answer_id = [answer_id]
            if any(f > 2 or f < 0 for f in answer_id):
                raise ValueError(f'Answer ID must be in range [0:2], got {answer_id}.')
            # if ground_truth['options'][answer_id] != answer_info['answer']:
            #     raise ValueError('Answer text is not as expected.')

            gt_answer_id = ground_truth['answer_id']
            ### Here is where Magic Happens ###
            if k == 1:
                val = int(gt_answer_id == answer_id[0])
            else:
                val = int(any(f == gt_answer_id for f in answer_id))
            total_correct += val
            total += 1

    if expected_total != total:
        print('Missing answers in results.')

    return total_correct / total


def calc_top_by_cat(answers_dict: Dict[str, Any],
                    db_dict: Dict[str, Any], k=1) -> Dict[str, Any]:
    """Calculates the top-k accuracy and results for each category.

    Args:
      answers_dict (Dict): Dictionary containing the answers.
      db_dict (Dict): Dictionary containing the database.

    Returns:
      Dict: Top-k accuracy and results for each category.
    """
    results_dict = {k: v for k, v in zip(CAT, np.zeros((len(CAT), 2)))}

    for vid_id, vid_answers in answers_dict.items():
        for answer_info in vid_answers:
            answer_id = answer_info['answer_id']
            question_idx = answer_info['id']
            ground_truth = db_dict[vid_id]['mc_question'][question_idx]
            gt_answer_id = ground_truth['answer_id']
            val = int(gt_answer_id == answer_id)

            used_q_areas = []
            q_areas = [TAG_AREA[tag] for tag in ground_truth['tag']]
            for a in q_areas:
                if a not in used_q_areas:
                    results_dict[a][0] += val
                    results_dict[a][1] += 1
                    used_q_areas.append(a)

            results_dict[ground_truth['reasoning']][0] += val
            results_dict[ground_truth['reasoning']][1] += 1
            for t in ground_truth['tag']:
                results_dict[t][0] += val
                results_dict[t][1] += 1

    results_dict = {k: v[0] / v[1] for k, v in results_dict.items()}
    return results_dict


def analyse_test_results(results, category='8'):
    shot_results = results[category]
    print('8-shot Frequency Results')
    print('---------------------------------')

    print('Areas')
    print('---------------------------------')
    for area in AREA:
        print(f'{area}: {shot_results[area] * 100:.2f}%')
    print('---------------------------------')

    print('Reasonings')
    print('---------------------------------')
    for reasoning in REASONING:
        print(f'{reasoning}: {shot_results[reasoning] * 100:.2f}%')
    print('---------------------------------')

    print('Tags')
    print('---------------------------------')
    for example_tag in TAG:
        print(f'{example_tag}: {shot_results[example_tag] * 100:.2f}%')


def atest_download_samples():
    sample_annot_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sample_annotations.zip'
    download_and_unzip(sample_annot_url, data_path)
    sample_videos_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sample_videos.zip'
    download_and_unzip(sample_videos_url, data_path)
    sample_audios_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sample_audios.zip'
    download_and_unzip(sample_audios_url, data_path)

    # validation set annotations to perform tracking
    valid_annot_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_valid_annotations.zip'
    download_and_unzip(valid_annot_url, data_path)
    # validation set annotations to perform tracking
    train_annot_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_train_annotations.zip'
    download_and_unzip(train_annot_url, data_path)

    # validation videos not downloaded because they are too big (approx 70GB).
    # not needed for this baseline since we are choosing answers based on the
    # frequency of oocurence we do not actually need the videos to calculate
    # the performance.

    train_videos_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/train_videos.zip'
    download_and_unzip(train_videos_url, data_path + '/train')
    # valid_videos_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/valid_videos.zip'
    # download_and_unzip(valid_videos_url, data_path)


def get_nouns_and_verbs(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    nouns = []
    verbs = []

    for token, tag in tagged_tokens:
        if tag.startswith("NN"):  # Noun
            nouns.append(token)
        elif tag.startswith("VB"):  # Verb
            verbs.append(token)

    return nouns + verbs


def word_overlap(str1, str2):
    words1 = set(get_nouns_and_verbs(str1))
    words2 = set(get_nouns_and_verbs(str2))
    # words1 = set(str1.split())
    # words2 = set(str2.split())
    return len(words1 & words2)  # Returns the count of common words


def find_largest_overlap(y, x):
    max_overlap = -1
    max_overlap_item = None

    for item in y:
        overlap = word_overlap(item, x)
        if overlap > max_overlap:
            max_overlap = overlap
            max_overlap_item = item
        elif overlap == max_overlap:
            if len(item.split()) < len(max_overlap_item.split()):
                max_overlap = overlap
                max_overlap_item = item

    return max_overlap_item


def find_closest_index(json_file, option_frames, method='word_overlap2'):
    """
    Finds the closest OOD option entry based on a chosen method
    Args:
        json_file: The json file with the option listing
        option_frames: The OOD options
        method: Method to choose [Word_overlap, character_overlap, soft_cosine]

    Returns: The respective embedding

    """

    ood_entries = []
    closest_entries = []
    for f in tqdm(option_frames):
        if f not in json_file:
            ### Retrieval Starts Here ###
            if method == 'word_overlap2':
                maybe_entry = find_largest_overlap(json_file, f)
                ### Retrieval Ends Here ###
                ood_entries.append(f)
                closest_entries.append(maybe_entry)
    return {a: b for a, b in zip(ood_entries, closest_entries)}


def generate_answer_keys_for_test(train_val_answer_keys, test_questions, method='word_overlap2'):
    org_name = train_val_answer_keys.split('.json')[0]
    with open(train_val_answer_keys, 'r') as fin:
        train_val_data = json.load(fin)

    with open(test_questions, 'r') as fin:
        test_questions_data = json.load(fin)

    o_list = []
    for k, v in test_questions_data.items():
        for i in v['mc_question']:
            for j in i['options']:
                o_list.append(j)
    o_list = list(set(o_list))
    pair_dict = find_closest_index(train_val_data, o_list)
    new_dict = train_val_data
    for k, v in pair_dict.items():
        new_dict.update({k: new_dict[v]})
    with open(org_name + method + '.json', 'w') as fout:
        json.dump(new_dict, fout)
