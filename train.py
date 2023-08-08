from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
import requests
from PIL import Image
from utils import get_ocr_frames
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR, draw_ocr
from skimage.color import rgb2gray


def crop_image(img, set_of_coordinates, scale=1.5):
    cropped_images = []
    for i in range(len(set_of_coordinates)):
        set_ = set_of_coordinates  # 4 pairs of polygon
        x1 = set_[0][0]
        y1 = set_[0][1]
        x2 = set_[1][0]
        y2 = set_[1][1]
        x3 = set_[2][0]
        y3 = set_[2][1]
        x4 = set_[3][0]
        y4 = set_[3][1]
        top_left_x = min([x1, x2, x3, x4])
        top_left_y = min([y1, y2, y3, y4])
        bot_right_x = max([x1, x2, x3, x4])
        bot_right_y = max([y1, y2, y3, y4])

        width_of_cut = (bot_right_y - top_left_y) * scale / 2
        height_of_cut = (bot_right_x - top_left_x) * scale / 2
        cropped = img[int(max(0, top_left_y - width_of_cut)): int(bot_right_y + width_of_cut),
                  int(max(0, top_left_x - height_of_cut)): int(bot_right_x + height_of_cut)]
        cropped_images.append(cropped)

    return cropped_images


processor = MgpstrProcessor.from_pretrained('alibaba-damo/mgp-str-base')
model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base')
ocr = PaddleOCR(use_angle_cls=True, lang='en', type='ocr', rec=True, det=True,
                det_db_thresh=0.1,
                label_list=['0'])
video = 'data/sample/videos/video_1589.mp4'
frames = get_ocr_frames(data_item=None,
                        video_folder_path='./data/sample/videos/',
                        override_video_name=True,
                        resize_to=300,
                        num_samples=16,
                        n_segments=1)
for j in range(len(frames[0])):
    img = frames[0][j]
    result = ocr.ocr(img, cls=True, rec=False, det=True)
    for entry in result[0]:
        if len(entry) > 0:
            cropped = crop_image(img, set_of_coordinates=entry)
            plt.figure(1)
            plt.clf()
            plt.imshow(cropped[0])
            plt.title('Number ' + str(j))
            plt.pause(1)
            result = ocr.ocr(cropped[0], cls=False, rec=True, det=True)
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    print(line)
    #         pixel_values = processor(images=cropped[0], return_tensors="pt").pixel_values
    #         outputs = model(pixel_values)
    #
    #         generated_text = processor.batch_decode(outputs.logits)['generated_text']
    #         print(generated_text)
