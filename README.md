# inswapper

One-click Face Swapper and Restoration powered by [insightface](https://github.com/deepinsight/insightface). We don't use the name ROOP here, as the credit should be given to the group that develops this great face swap model.

## News
🔥 We release [InstantID](https://github.com/InstantID/InstantID) as a state-of-the-art ID preservering generation method.

<left><img src="https://github.com/haofanwang/inswapper/raw/main/data/mans1.jpeg" width="49%" height="49%"></left> 
<right><img src="https://github.com/haofanwang/inswapper/raw/main/result.png" width="49%" height="49%"></right> 

## Installation

```bash
# git clone this repository
git clone https://github.com/haofanwang/inswapper.git
cd inswapper

# create a Python venv
python3 -m venv venv

# activate the venv
source venv/bin/activate

# install required packages
pip install -r requirements.txt
```

You have to install ``onnxruntime-gpu`` manually to enable GPU inference, install ``onnxruntime`` by default to use CPU only inference.

## Download Checkpoints

First, you need to download [face swap model](https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx) and save it under `./checkpoints`. To obtain better result, it is highly recommended to improve image quality with face restoration model. Here, we use [CodeFormer](https://github.com/sczhou/CodeFormer). You can finish all as following, required models will be downloaded automatically when you first run the inference.

```bash
mkdir checkpoints
wget -O ./checkpoints/inswapper_128.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx

cd ..
git lfs install
git clone https://huggingface.co/spaces/sczhou/CodeFormer
```


## Quick Inference

```bash
from swapper import *

source_img = [Image.open("./data/man1.jpeg"),Image.open("./data/man2.jpeg")]
target_img = Image.open("./data/mans1.jpeg")

model = "./checkpoints/inswapper_128.onnx"
result_image = process(source_img, target_img, -1, -1, model)
result_image.save("result.png")
```

To improve to quality of face, we can further do face restoration as shown in the full script.

```bash
python swapper.py \
--source_img="./data/man1.jpeg;./data/man2.jpeg" \
--target_img "./data/mans1.jpeg" \
--face_restore \
--background_enhance \
--face_upsample \
--upscale=2 \
--codeformer_fidelity=0.5
```
You will obtain the exact result as above.

## Acknowledgement
This project is inspired by [inswapper](https://huggingface.co/deepinsight/inswapper/tree/main), thanks [insightface.ai](https://insightface.ai/) for releasing their powerful face swap model that makes this happen. Our codebase is built on the top of [sd-webui-roop](https://github.com/s0md3v/sd-webui-roop) and [CodeFormer](https://huggingface.co/spaces/sczhou/CodeFormer).

## Contact
If you have any issue, feel free to contact me via haofanwang.ai@gmail.com.
