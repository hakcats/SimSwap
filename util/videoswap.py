'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:52
Description: 
'''
import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import  time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import time
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils import img2tensor, tensor2img
from cl import test

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def rev(tensor):
    tensor = (tensor * 255).int()
    a = tensor.transpose(2, 0).transpose(1, 0)
    return a.cpu().detach().numpy()


def video_swap(video_path, id_vetor, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False):
    video_forcheck = VideoFileClip(video_path)
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    video = cv2.VideoCapture(video_path)
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    ret = True
    frame_index = 0

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = video.get(cv2.CAP_PROP_FPS)
    if  os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)

    spNorm =SpecificNorm()
    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net =None

    # while ret:
    i = 0

    face_helper = test(
        1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device='cuda',
        model_rootpath='weights')

    f = 'ttt/masks/1.pth'
    mask = torch.load(f, map_location='cuda')

    for frame_index in tqdm(range(frame_count)):
        i += 1
        f = 'ttt/masks/{}.pth'.format(i)
        mask = torch.load(f, map_location='cuda')
        # for frame_index in tqdm(range(3)):
        # if i > 10:
        #     exit(0)

        start_time = time.perf_counter()

        ret, frame = video.read()

        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f'function video read took {total_time:.3f} seconds')
        if  ret:
            start_time = time.perf_counter()

            detect_results = detect_model.get(frame,crop_size)

            end_time = time.perf_counter()
            total_time = end_time - start_time
            # print(f'function detect_results took {total_time:.3f} seconds')

            start_time = time.perf_counter()
            if detect_results is not None:
                # print(frame_index)
                if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                frame_align_crop_list = detect_results[0]
                swap_result_list = []
                frame_align_crop_tenor_list = []
                for frame_align_crop in frame_align_crop_list:
                    # BGR TO RGB
                    # frame_align_crop_RGB = frame_align_crop[...,::-1]
                    # cv2.imwrite(os.path.join(temp_results_dir, 'frame_a_{:0>7d}.jpg'.format(frame_index)), frame_align_crop)
                    # print(frame_align_crop)
                    frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

                    swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                    # print(tensor2img(swap_result.squeeze(0), rgb2bgr=True, min_max=(-1, 1)))
                    # cv2.imwrite(os.path.join(temp_results_dir, 'frame_d_{:0>7d}.jpg'.format(frame_index)), tensor2img(swap_result.squeeze(0), rgb2bgr=True))
                    # print(frame_align_crop_tenor)
                    # exit(0)
                    # cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                    # cv2.imwrite(os.path.join('ttt', 'aframe_{:0>7d}.jpg'.format(frame_index)), swap_result.cpu().detach().numpy().astype(np.uint8))
                    swap_result_list.append(swap_result)
                    frame_align_crop_tenor_list.append(frame_align_crop_tenor)

                dd = face_helper.generate(frame, restored_face=tensor2img(swap_result_list[0].squeeze(0), rgb2bgr=True).astype('uint8'), i=i, mask=mask)
                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), dd)
                # reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
                #     os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,use_mask=use_mask, norm = spNorm)

            else:
                start_time = time.perf_counter()
                if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)
                frame = frame.astype(np.uint8)
                if not no_simswaplogo:
                    frame = logoclass.apply_frames(frame)
                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                end_time = time.perf_counter()
                total_time = end_time - start_time
                print(f'function else took {total_time:.3f} seconds')


        else:
            break



    video.release()

    # image_filename_list = []
    path = os.path.join(temp_results_dir,'*.jpg')
    image_filenames = sorted(glob.glob(path))

    clips = ImageSequenceClip(image_filenames,fps = fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)


    clips.write_videofile(save_path,audio_codec='aac')

