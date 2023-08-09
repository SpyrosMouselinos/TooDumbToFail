import pickle
import numpy as np
from paddleocr import PaddleOCR


class OCRmodel:
    def __init__(self, use_angle_cls=True, det_db_thresh=0.1, label_list=['0'], dual_pass=False, crop_scale=1.5):
        self.use_angle_cls = use_angle_cls
        self.det_db_thresh = det_db_thresh
        self.label_list = label_list
        self.dual_pass = dual_pass
        self.model = PaddleOCR(use_angle_cls=use_angle_cls, lang='en', type='ocr', rec=True, det=True,
                               det_db_thresh=det_db_thresh,
                               label_list=label_list, show_log = False)
        self.crop_scale = crop_scale

    def __repr__(self):
        return (f"PaddleOCR-based model that does {'two' if self.dual_pass else 'one'} steps."
                f" It reads at {self.label_list} degrees.")

    def __call__(self, video_frames: np.ndarray, *args, **kwargs):
        BS, H, W, C = video_frames.shape
        identified_strings = []
        for frame_idx in range(BS):
            img = video_frames[frame_idx]
            if self.dual_pass:
                result = self.model.ocr(img, cls=True, rec=False, det=True)[0]
                for entry in result:
                    if len(entry) > 0:
                        cropped = self.crop_image(img, set_of_coordinates=entry)
                        updated_result = self.model.ocr(cropped[0], cls=False, rec=True, det=True)[0]
                        if len(updated_result) > 0:
                            identified_chars = updated_result[0][1][0]
                            identified_strings.append(identified_chars)
            else:
                result = self.model.ocr(img, cls=True, rec=True, det=True)[0]
                for entry in result:
                    if len(entry) > 0:
                        identified_chars = entry[1][0]
                        identified_strings.append(identified_chars)
        return identified_strings

    def crop_image(self, img, set_of_coordinates):
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

            width_of_cut = (bot_right_y - top_left_y) * self.crop_scale / 2
            height_of_cut = (bot_right_x - top_left_x) * self.crop_scale / 2
            cropped = img[int(max(0, top_left_y - width_of_cut)): int(bot_right_y + width_of_cut),
                      int(max(0, top_left_x - height_of_cut)): int(bot_right_x + height_of_cut)]
            cropped_images.append(cropped)

        return cropped_images

    def save_model(self, path):
        state = {
            'use_angle_cls': self.use_angle_cls,
            'det_db_thresh': self.det_db_thresh,
            'label_list': self.label_list,
            'dual_pass': self.dual_pass,
            'crop_scale': self.crop_scale
        }
        with open(path, 'wb') as fout:
            pickle.dump(state, fout)

    def load_model(self, path):
        with open(path, 'rb') as fin:
            state = pickle.load(fin)

        self.use_angle_cls = state['use_angle_cls']
        self.det_db_thresh = state['det_db_thresh']
        self.label_list = state['label_list']
        self.dual_pass = state['dual_pass']
        self.crop_scale = state['crop_scale']
        self.model = PaddleOCR(use_angle_cls=self.use_angle_cls, lang='en', type='ocr', rec=True, det=True,
                               det_db_thresh=self.det_db_thresh,
                               label_list=self.label_list, show_log = False)
