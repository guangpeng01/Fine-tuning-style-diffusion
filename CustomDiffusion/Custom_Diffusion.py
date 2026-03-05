import os  
import random  
from pathlib import Path  

import numpy as np  
import safetensors  # 
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from safetensors.torch import load_file

import diffusers
# from diffusers.pipelines import BlipDiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import CustomDiffusionAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import load_image
import streamlit as st

import io

import streamlit as st  # 用于创建交互式网页UI
import io  # 处理文件流(后面用来生成下载按钮)

# 设置页面标题和布局
st.set_page_config(page_title="Fine-tuning style diffusion", layout="wide")

st.title("Fine-tuning style diffusion 推理 Demo")

st.write("支持 **A <new1> reference.(风格) + 文本*")

st.write("只是训练了一个提示词 'A <new1> reference.'")

st.write("即使用该提示词时以十二生肖为主要元素进行新年图片风格的生成，例如使用一下提示词")

st.write("A <new1> reference. New Year image with a  rabbit as the main element, in a 2D or anime style, and a festive background")


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16


# ==========================
# 模型加载（缓存）
# ==========================

@st.cache_resource
def load_models():

    model_path = "./stable-diffusion-v1-5"

    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16 
    ).to(device)

    vae = AutoencoderKL.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float16 
    ).to(device)

    unet = UNet2DConditionModel.from_pretrained(
        model_path,
        subfolder="unet",
        torch_dtype=torch.float16  
    ).to(device)

    attn_path = "output/pytorch_custom_diffusion_weights.bin"

    state_dict = torch.load(attn_path, map_location="cpu")
    unet.load_attn_procs(state_dict)

    token_path = "output/learned_embeds.safetensors"


    try:

        new_embed = torch.load(token_path)

        token_id = tokenizer.convert_tokens_to_ids("<new1>")

        text_encoder.get_input_embeddings().weight.data[token_id] = new_embed

        print("Loaded <new1> token embedding")

    except:
        print("No trained <new1> token found")

    scheduler = DDPMScheduler.from_pretrained(
        model_path,
        subfolder="scheduler"
    )
    
    unet.enable_xformers_memory_efficient_attention()

    return tokenizer, text_encoder, vae, unet, scheduler

tokenizer, text_encoder, vae, unet, scheduler = load_models()


prompt = st.text_input(
    "Prompt",
    "A <new1> reference."
)

# 调整参数
steps = st.slider("Steps", 10, 320, 100)

guidance = st.slider("Guidance", 1.0, 18.0, 6.0)


# ==========================
# 图像预处理
# ==========================

def preprocess(image):
    # 调整图像，转换为tensor（张量）并归一化到[-1,1]
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    # 增加batch维度
    return transform(image).unsqueeze(0)


# ==========================
# diffusion 推理
# ==========================

def generate(prompt):

    with torch.no_grad():
        # 文本向量化
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        text_emb = text_encoder(text_input.input_ids)[0]

        # 无条件 embedding;
        uncond_input = tokenizer(
            "",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).to(device)

        uncond_emb = text_encoder(uncond_input.input_ids)[0]


        text_emb = torch.cat([uncond_emb, text_emb], dim=0)

        # 初始化噪声潜变量
        latents = torch.randn(
            (1,4,64,64),
            device=device,
            dtype=torch.float16
        )

        # 设置diffusion时间步
        scheduler.set_timesteps(steps)

        # ----------------
        # diffusion loop
        # ----------------
        # 采用
        for t in scheduler.timesteps:
            # 为什么要拼接两份
            latent_model_input = torch.cat([latents]*2)

            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_emb
            ).sample


            noise_uncond, noise_text = noise_pred.chunk(2)

            noise_pred = noise_uncond + guidance * (
                noise_text - noise_uncond
            )
            # 调度程序/潜在的
            latents = scheduler.step(
                noise_pred,
                t,
                latents
            ).prev_sample


        # ----------------
        # decode image；解码图像
        # ----------------
        # 解码生成图像；将latent解码成[0,1]的RGB图像
        latents = latents / vae.config.scaling_factor

        image = vae.decode(latents).sample

        image = (image/2 + 0.5).clamp(0,1)
        # 转成numpy数组，再用PIL转成可展示的图像
        image = image.cpu().permute(0,2,3,1).numpy()[0]

        image = (image*255).astype(np.uint8)

        return Image.fromarray(image)


if st.button("Generate"):

    with st.spinner("Generating..."):

        image = generate(prompt)

    st.image(image,caption="Result",width=512)

    buf = io.BytesIO()

    image.save(buf,format="PNG")

    st.download_button(
        "Download",
        buf.getvalue(),
        "result.png"
    )

