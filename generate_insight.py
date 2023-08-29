
# from multiprocessing import Pool
import threading
from prompt.wiki import WIKI
import os
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from PIL import Image
import numpy as np


def generate(paths, root, target_dir):

    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(
        './inswapper_128.onnx', download=False, download_zip=False)

    for src_file, dst_file in paths:
        try:
            src_img = np.array(Image.open(
                os.path.join(root, src_file)).convert('RGB'))
            faces = app.get(src_img)
            faces = sorted(faces, key=lambda x: x.bbox[0])
            src_face = faces[0]

            dst_img = np.array(Image.open(
                os.path.join(root, dst_file)).convert('RGB'))
            faces = app.get(dst_img)
            faces = sorted(faces, key=lambda x: x.bbox[0])
            dst_face = faces[0]

            res = dst_img.copy()
            res = swapper.get(res, dst_face, src_face, paste_back=True)

            dir = src_file.split('/')[0]
            if not os.path.exists(os.path.join(target_dir, dir)):
                os.makedirs(os.path.join(target_dir, dir))
            Image.fromarray(res).save(os.path.join(target_dir, src_file))
        except Exception as e:
            print(e)
    return


if __name__ == '__main__':
    root = './wiki_faces'
    target_dir = './insight'
    total_threads = 8
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    dataset = WIKI(prompt_pth='./data/wiki.pickle',
                   data_pth='./wiki/wiki.mat')
    _, _, paths = dataset.prompt
    gap = int(len(paths)/total_threads)
    print(gap)
    threads = []
    for idx in range(total_threads):
        if idx == total_threads-1:
            path = paths[idx*gap:]
        else:
            path = paths[idx*gap:(idx+1)*gap]

        th = threading.Thread(target=generate, args=(path, root, target_dir,))
        th.start()
        threads.append(th)

    for th in threads:
        th.join()
