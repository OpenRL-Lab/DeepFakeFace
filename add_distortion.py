import argparse
from distortion import color_contrast, color_saturation, jpeg_compression, gaussian_blur
import os
import cv2
from tqdm import tqdm

func_dict = {'CS': color_saturation, 'CC': color_contrast,
             'JPEG': jpeg_compression, 'GB': gaussian_blur}
param_dict = {'CS': 0.4, 'CC': 0.85,
              'JPEG': 2, 'GB': 7}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add a distortion to video.')
    parser.add_argument('--img_in', '-i',
                        type=str,
                        default='./faces/inpainting_crop',
                        help='path to the input video')
    parser.add_argument('--img_out', '-o',
                        type=str,
                        default=None,
                        help='path to the output video (defaults to ./perturbations/[type])')
    parser.add_argument(
        '--type', '-t',
        type=str,
        default='CS',
        help='distortion type: CS | CC | JPEG | GB')
    args = parser.parse_args()

    # Dynamic feature: Automatically matches output folder name to the chosen type if not specified
    output_root = args.img_out or os.path.join('./perturbations', args.type)

    for dir in tqdm(os.listdir(args.img_in)):
        for img in os.listdir(os.path.join(args.img_in, dir)):
            img_path = os.path.join(args.img_in, dir, img)
            ori_img = cv2.imread(img_path)
            per_img = func_dict[args.type](ori_img, param_dict[args.type])
            
            save_dir = os.path.join(output_root, dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(os.path.join(save_dir, img), per_img)
