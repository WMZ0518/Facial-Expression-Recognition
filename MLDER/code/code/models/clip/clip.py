import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

# 从当前目录下的 model.py 导入 build_model 函数
from .model import build_model

# 从 simple_tokenizer.py 导入 SimpleTokenizer 并重命名为 _Tokenizer
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

# 尝试导入 torchvision.transforms.InterpolationMode.BICUBIC（双三次插值）
# 如果失败则使用 PIL.Image.BICUBIC 作为替代
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# PyTorch 版本推荐提示（当前被注释掉）
# if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
#     warnings.warn("建议使用 PyTorch 1.7.1 或更高版本")

# 定义该模块对外暴露的接口函数
__all__ = ["available_models", "load", "tokenize"]

# 初始化一个全局分词器实例
_tokenizer = _Tokenizer()

# 支持的 CLIP 模型及其对应的下载链接
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    """
    下载模型权重文件，并验证 SHA256 校验和。

    参数:
        url: 模型权重文件的 URL 地址
        root: 下载目标路径

    返回:
        download_target: 下载完成后的本地路径
    """
    os.makedirs(root, exist_ok=True)  # 创建下载目录（如果不存在）
    filename = os.path.basename(url)  # 获取文件名

    expected_sha256 = url.split("/")[-2]  # 从URL提取期望的SHA256校验码
    download_target = os.path.join(root, filename)  # 构造完整下载路径

    # 如果路径存在但不是文件，抛出错误
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} 存在但不是一个普通文件")

    # 如果文件已存在且校验通过，则直接返回路径
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} 存在，但SHA256校验失败；将重新下载")

    # 开始下载文件并显示进度条
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    # 再次校验SHA256，失败则抛出异常
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("模型已下载，但SHA256校验失败")

    return download_target


def _convert_image_to_rgb(image):
    """
    将图像转换为RGB格式。
    
    参数:
        image: 输入图像（PIL.Image）

    返回:
        RGB格式图像
    """
    return image.convert("RGB")


def _transform(n_px):
    """
    构建图像预处理管道。
    
    参数:
        n_px: 输出图像尺寸

    返回:
        transform: 图像预处理操作序列
    """
    return Compose([
        Resize(n_px, interpolation=BICUBIC),  # 双三次插值缩放至指定大小
        CenterCrop(n_px),  # 中心裁剪为正方形
        _convert_image_to_rgb,  # 转换为RGB图像
        ToTensor(),  # 转换为PyTorch张量
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # 使用CLIP均值方差归一化
    ])


def available_models() -> List[str]:
    """
    获取所有可用的CLIP模型名称。
    
    返回:
        模型名称列表
    """
    return list(_MODELS.keys())  # 返回所有支持的模型名称列表


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """
    加载CLIP模型。
    
    参数:
        name: 模型名称或自定义路径
        device: 运行设备（CPU/GPU）
        jit: 是否加载JIT优化模型
        download_root: 模型下载路径

    返回:
        model: 加载好的CLIP模型
        preprocess: 图像预处理函数
    """
    if name in _MODELS:
        # 如果是内置模型，下载模型权重
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        # 如果是已有文件，直接使用
        model_path = name
    else:
        # 否则报错
        raise RuntimeError(f"模型 {name} 未找到；可用模型：{available_models()}")

    # 打开模型文件尝试加载
    with open(model_path, 'rb') as opened_file:
        try:
            # 先尝试作为JIT模型加载
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # 失败则尝试作为state_dict加载
            if jit:
                warnings.warn(f"文件 {model_path} 不是JIT模型，尝试作为state_dict加载")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        # 非JIT模式下构建模型
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()  # CPU上使用float32
        return model, _transform(model.visual.input_resolution)

    # JIT模式下需要修补设备信息
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # 如果运行在CPU上，修补数据类型为float32
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype可以是第二个或第三个参数
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    对输入的文本字符串或字符串列表进行分词处理，返回其对应的token化表示。

    参数说明:
    ----------
    texts : Union[str, List[str]]
        输入文本，可以是一个字符串或者多个字符串组成的列表。每个字符串会被独立处理。

    context_length : int，默认值为 77
        每个文本的最大长度限制（包含特殊标记）。CLIP模型的标准上下文长度为77。

    truncate : bool，默认值为 False
        是否截断超过 context_length 的文本。如果为 True，则超出部分会被截断；否则会抛出异常。

    返回值:
    -------
    Union[torch.IntTensor, torch.LongTensor]
        一个二维张量，形状为 [输入字符串数量, context_length]，表示 token 化后的结果。
        如果 PyTorch 版本 <1.8.0，返回 LongTensor；否则返回 IntTensor。
    """
    # 将输入转换为列表形式，确保统一处理
    if isinstance(texts, str):
        texts = [texts]

    # 获取起始 (SOT) 和结束 (EOT) 标记的编码
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["
