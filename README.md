# inswapper

## One-click training-free Face Swapper.


<left><img src="https://github.com/haofanwang/roop-for-diffusers/raw/main/data/mans1.jpeg" width="49%" height="49%"></left> 
<right><img src="https://github.com/haofanwang/roop-for-diffusers/raw/main/result.png" width="49%" height="49%"></right> 

## Installation

```bash
# git clone this repository
git clone https://github.com/haofanwang/inswapper.git
cd inswapper
```

## Download Checkpoints
First, you need to download [face swap model](https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx) and save it under `./checkpoints`. To obtain better result, it is highly recommendated to improve image quality with face restoration model. Here, we use [CodeFormer](https://github.com/sczhou/CodeFormer). You can finish all as following, required models will be downloaded automatically when you first run the inference.

```bash
mkdir checkpoints
wget -O ./checkpoints/inswapper_128.onnx https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx 

cd ..
git lfs install
git clone https://huggingface.co/spaces/sczhou/CodeFormer
```


## Quick Inference

```bash
source_img = [Image.open("./data/man1.jpeg"),Image.open("./data/man2.jpeg")]
target_img = Image.open("./data/mans1.jpeg")

model = "./checkpoints/inswapper_128.onnx"
result_image = process(source_img,target_img, model)
```

## Acknowledgement
This project is inspired by [inswapper](https://huggingface.co/deepinsight/inswapper/tree/main), thanks [insightface.ai](https://insightface.ai/) for releasing their powful swap model that makes this happen. Our codebase is built on the top of [sd-webui-roop](https://github.com/s0md3v/sd-webui-roop) and [CodeFormer](https://huggingface.co/spaces/sczhou/CodeFormer).

## Contact
If you have any issue, feel free to contact me via haofanwang.ai@gmail.com.
