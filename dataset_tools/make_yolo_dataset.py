import os
import shutil

DATA_ALL_DIR = os.path.join('.', 'images')
DATA_OUT_DIR = os.path.join('.', 'data')

if os.path.exists(DATA_OUT_DIR):
    shutil.rmtree(DATA_OUT_DIR)
os.mkdir(DATA_OUT_DIR)

for set_ in ['train', 'val', 'test']:
    os.mkdir(os.path.join(DATA_OUT_DIR, set_))
    os.mkdir(os.path.join(DATA_OUT_DIR, set_, 'imgs'))
    os.mkdir(os.path.join(DATA_OUT_DIR, set_, 'anns'))

alpaca_id = '/m/0pcr'

train_bboxes_filename = os.path.join('.', 'oidv6-train-annotations-bbox.csv')
validation_bboxes_filename = os.path.join('.', 'validation-annotations-bbox.csv')
test_bboxes_filename = os.path.join('.', 'test-annotations-bbox.csv')

for j, filename in enumerate([train_bboxes_filename, validation_bboxes_filename, test_bboxes_filename]):
    set_ = ['train', 'val', 'test'][j]
    print(filename)

    with open(filename, 'r') as f:
        line = f.readline()
        while len(line) != 0:
            parts = line.strip().split(',')

            if len(parts) >= 8:
                image_id = parts[0]
                class_name = parts[2]
                x1, x2, y1, y2 = parts[4], parts[5], parts[6], parts[7]

                if class_name == alpaca_id:
                    src_img = os.path.join(DATA_ALL_DIR, f'{image_id}.jpg')
                    dst_img = os.path.join(DATA_OUT_DIR, set_, 'imgs', f'{image_id}.jpg')

                    if os.path.exists(src_img):
                        if not os.path.exists(dst_img):
                            shutil.copy(src_img, dst_img)

                        with open(os.path.join(DATA_OUT_DIR, set_, 'anns', f'{image_id}.txt'), 'a') as f_ann:
                            x1, x2, y1, y2 = [float(v) for v in [x1, x2, y1, y2]]
                            xc = (x1 + x2) / 2
                            yc = (y1 + y2) / 2
                            w = x2 - x1
                            h = y2 - y1

                            f_ann.write(f'0 {xc} {yc} {w} {h}\n')

            line = f.readline()