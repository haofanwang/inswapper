"""
This project is developed by Haofan Wang to support face swap in single frame. Multi-frame will be supported soon!

It is highly built on the top of insightface, sd-webui-roop and CodeFormer.
"""

import os
import cv2
import copy
import argparse
import insightface
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Set, Tuple


def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model


def getFaceAnalyser(model_path: str,
                    det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints")
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_one_face(face_analyser,
                 frame:np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None

    
def get_many_faces(face_analyser,
                   frame:np.ndarray):
    """
    get faces from left to right by order
    """
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None

    
def swap_face(face_swapper,
              source_face, 
              target_face, 
              temp_frame):
    """
    paste source_face on target image
    """
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)
 
    
def process(source_img: Union[Image.Image, List], 
            target_img: Image.Image, 
            model: str):
        
    # load face_analyser
    face_analyser = getFaceAnalyser(model)
    
    # load face_swapper
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    face_swapper = getFaceSwapModel(model_path)
    
    # read target image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    
    # detect faces that will be replaced in target_img
    target_faces = get_many_faces(face_analyser, target_img)
    if target_faces is not None:
        temp_frame = copy.deepcopy(target_img)
        if isinstance(source_img, list) and len(source_img) == len(target_faces):
            # replace faces in target image from the left to the right by order
            for i in range(len(target_faces)):
                target_face = target_faces[i]
                source_face = get_one_face(face_analyser, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR))
                if source_face is None:
                    raise Exception("No source face found!")
                temp_frame = swap_face(face_swapper, source_face, target_face, temp_frame)
        else:
            # replace all faces in target image to same source_face
            source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
            source_face = get_one_face(face_analyser, source_img)
            if source_face is None:
                raise Exception("No source face found!")
            for target_face in target_faces:
                temp_frame = swap_face(face_swapper, source_face, target_face, temp_frame)
        result = temp_frame
    else:
        print("No target faces found!")
    
    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image


def parse_args():
    parser = argparse.ArgumentParser(description="Face swap.")
    parser.add_argument("--source_img", type=str, required=True, help="The path of source image, it can be multiple images, dir;dir2;dir3.")
    parser.add_argument("--target_img", type=str, required=True, help="The path of target image.")
    parser.add_argument("--face_restore", action="store_true", help="The flag for face restoration.")
    parser.add_argument("--background_enhance", action="store_true", help="The flag for background enhancement.")
    parser.add_argument("--face_upsample", action="store_true", help="The flag for face upsample.")
    parser.add_argument("--upscale", type=int, default=1, help="The upscale value, up to 4.")
    parser.add_argument("--codeformer_fidelity", type=float, default=0.5, help="The codeformer fidelity.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_args()
    
    source_img_paths = args.source_img.split(';')
    print(source_img_paths)
    target_img_path = args.target_img
    
    source_img = [Image.open(img_path) for img_path in source_img_paths]
    target_img = Image.open(target_img_path)

    # download from https://huggingface.co/deepinsight/inswapper/tree/main
    model = "./checkpoints/inswapper_128.onnx"
    result_image = process(source_img, target_img, model)
    
    if args.face_restore:
        from restoration import *
        
        # make sure the ckpts downloaded successfully
        check_ckpts()
        
        # https://huggingface.co/spaces/sczhou/CodeFormer
        upsampler = set_realesrgan()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                         codebook_size=1024,
                                                         n_head=8,
                                                         n_layers=9,
                                                         connect_list=["32", "64", "128", "256"],
                                                        ).to(device)
        ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
        checkpoint = torch.load(ckpt_path)["params_ema"]
        codeformer_net.load_state_dict(checkpoint)
        codeformer_net.eval()
        
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        result_image = face_restoration(result_image, 
                                        args.background_enhance, 
                                        args.face_upsample, 
                                        args.upscale, 
                                        args.codeformer_fidelity,
                                        upsampler,
                                        codeformer_net,
                                        device)
        result_image = Image.fromarray(result_image)
    
    # save result
    result_image.save("result.png")
