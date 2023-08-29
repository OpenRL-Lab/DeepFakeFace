import os
# from multiprocessing import Pool
import threading
from prompt.wiki import WIKI
from diffusers import StableDiffusionPipeline
import argparse


def generate(method, device, length, batch, path, prompt, root):
    if length % batch == 0:
        iteration = int(length/batch)
    else:
        iteration = int(length/batch)+1
    if method == 'sd':
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        print("cuda:"+str(device))
        pipe = pipe.to("cuda:"+str(device))
        print('pipe is ok')
        for pth in path:
            dir = pth.split('/')[0]
            if not os.path.exists(os.path.join(root, dir)):
                os.makedirs(os.path.join(root, dir))
        for idx in range(iteration-1):
            images = pipe(prompt[idx*batch:(idx+1)*batch]).images
            for i, image in enumerate(images):
                image.save(os.path.join(root, path[idx*batch+i]))
        images = pipe(prompt[(iteration-1)*batch:]).images
        for i, image in enumerate(images):
            image.save(os.path.join(root, path[(iteration-1)*batch+i]))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--devices', '-d', type=list,
                        default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--num', type=int, default=70)
    parser.add_argument('--each', type=int, default=2)
    parser.add_argument('--root', type=str,
                        default='./result/text2img')
    parser.add_argument('--method', type=str, default='sd', choices=['sd'])
    args = parser.parse_args()
    total_device = len(args.devices)
    if not os.path.exists(args.root):
        os.makedirs(args.root)
    dataset = WIKI(prompt_pth='./data/wiki.pickle',
                   data_pth='./wiki/wiki.mat', num=args.num, each=args.each)
    paths, prompts, pairs = dataset.prompt

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
            args.method, device, length, args.batch, path, prompt, args.root))
        th.start()
        threads.append(th)

    for th in threads:
        th.join()
