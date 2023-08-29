import argparse
import os
# from multiprocessing import Pool
import threading
from prompt.wiki import WIKI
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image


def generate(method, device, length, batch, path, prompt, root, face_root, mask_root):
    if length % batch == 0:
        iteration = int(length/batch)
    else:
        iteration = int(length/batch)+1
    if method == 'sd':
        model_id = "runwayml/stable-diffusion-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id)
        print("cuda:"+str(device))
        pipe = pipe.to("cuda:"+str(device))
        print('pipe is ok')

        # for idx in range(iteration-1):
        #     image_tmp = []
        #     mask_tmp = []
        #     for pth in path[idx*batch:(idx+1)*batch]:
        #         dir = pth.split('/')[0]
        #         if not os.path.exists(os.path.join(root, dir)):
        #             os.makedirs(os.path.join(root, dir))
        #         image_tmp.append(Image.open(
        #             os.path.join(face_root, pth)).convert('RGB'))
        #         mask_tmp.append(Image.open(
        #             os.path.join(mask_root, pth)).convert('RGB'))

        #     # image = pipeline(prompt=prompt, image=init_images,mask_image=mask_images).images
        #     images = pipe(prompt=prompt[idx*batch:(idx+1)*batch],
        #                   image=image_tmp, mask_image=mask_tmp).images
        #     for i, image in enumerate(images):
        #         image.save(os.path.join(root, path[idx*batch+i]))
        image_tmp = []
        mask_tmp = []
        for pth in path[(iteration-1)*batch:]:
            dir = pth.split('/')[0]
            if not os.path.exists(os.path.join(root, dir)):
                os.makedirs(os.path.join(root, dir))
            image_tmp.append(Image.open(
                os.path.join(face_root, pth)).convert('RGB'))
            mask_tmp.append(Image.open(
                os.path.join(mask_root, pth)).convert('RGB'))
        images = pipe(prompt=prompt[(iteration-1)*batch:],
                      image=image_tmp, mask_image=mask_tmp).images
        for i, image in enumerate(images):
            image.save(os.path.join(root, path[idx*batch+i]))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo of argparse')

    # 2. 添加命令行参数
    # parser.add_argument('--devices', '-d', type=list, default=[0])
    parser.add_argument('--devices', '-d', type=list,
                        default=[0, 1, 2, 3, 4, 5, 6, 7])
    # parser.add_argument('--total_device', type=int, default=8)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--num', type=int, default=70)
    parser.add_argument('--each', type=int, default=2)
    parser.add_argument('--root', type=str,
                        default='./deepfakes')
    parser.add_argument('--face_root', type=str,
                        default='./wiki_faces')
    parser.add_argument('--mask_root', type=str,
                        default='./wiki_masks')
    parser.add_argument('--method', type=str, default='sd', choices=['sd'])
    # 3. 从命令行中结构化解析参数
    args = parser.parse_args()
    total_device = len(args.devices)
    if not os.path.exists(args.root):
        os.makedirs(args.root)
    dataset = WIKI(prompt_pth='./data/wiki.pickle',
                   data_pth='./wiki/wiki.mat', num=args.num, each=args.each)
    paths, prompts, pair = dataset.prompt
    gap = int(len(prompts)/total_device)
    print(gap)
    threads = []
    for idx, device in enumerate(args.devices):
        if idx == total_device-1:
            path = paths[idx*gap:]
            prompt = prompts[idx*gap:]
            length = len(prompts)-idx*gap
        else:
            path = paths[idx*gap:(idx+1)*gap]
            prompt = prompts[idx*gap:(idx+1)*gap]
            length = gap

        th = threading.Thread(target=generate, args=(
            args.method, device, length, args.batch, path, prompt, args.root, args.face_root, args.mask_root))
        th.start()
        threads.append(th)

    for th in threads:
        th.join()
