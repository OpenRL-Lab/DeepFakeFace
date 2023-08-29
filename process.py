from scipy.io import loadmat
from os.path import join
# import cv2
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import threading


def generate(root, mask_dir, faces_dir, paths, face_locations):
    result = []
    for path, face_location in zip(paths, face_locations):
        try:
            dir = os.path.split(path[0])[0]
            image = Image.open(join(root, path[0]))
            image = image.convert("RGB")
            mask = np.zeros_like(image)
            if len(mask.shape) == 3:
                mask[int(face_location[0][1]):int(face_location[0][3]), int(
                    face_location[0][0]):int(face_location[0][2]), :] = 255
            else:

                mask[int(face_location[0][1]):int(face_location[0][3]), int(
                    face_location[0][0]):int(face_location[0][2])] = 255
            image = image.resize((512, 512))
            mask = Image.fromarray(mask).resize((512, 512))
            if not os.path.exists(join(faces_dir, dir)):
                os.makedirs(join(faces_dir, dir), exist_ok=True)
            image.save(join(faces_dir, path[0]))
            if not os.path.exists(join(mask_dir, dir)):
                os.makedirs(join(mask_dir, dir), exist_ok=True)
            mask.save(join(mask_dir, path[0]))
        except Exception as e:
            print('\n error is:', e)


if __name__ == '__main__':
    total_threads = 200
    root = './wiki'
    mask_dir = './result/wiki_masks'
    faces_dir = './result/wiki_faces'
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    mat = loadmat(join(root, 'wiki.mat'))['wiki'][0][0]
    gap = int(len(mat[0][0])/total_threads)
    threads = []
    for idx in range(total_threads):
        if idx == total_threads-1:
            th = threading.Thread(target=generate, args=(
                root, mask_dir, faces_dir, mat[2][0][idx*gap:], mat[5][0][idx*gap:]))
            th.start()
        else:
            th = threading.Thread(target=generate, args=(
                root, mask_dir, faces_dir, mat[2][0][idx*gap:(idx+1)*gap], mat[5][0][idx*gap:(idx+1)*gap]))
            th.start()
        threads.append(th)
    for th in threads:
        th.join()
