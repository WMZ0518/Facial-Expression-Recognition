# 导入必要的库
import torchvision  # PyTorch 提供的视觉变换工具
import random       # 随机数生成器
from PIL import Image, ImageOps  # 图像处理库
import numpy as np  # 数值计算库
import numbers      # 数字类型检测
import math         # 数学函数
import torch        # PyTorch 核心库
import PIL          # Python Imaging Library

class GroupRandomCrop(object):
    """
    对一组图像进行随机裁剪（所有图像使用相同位置）

    参数:
        size (int or tuple): 裁剪的目标尺寸 (height, width)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        """
        执行裁剪操作
        
        参数:
            img_group (list[PIL.Image]): 多帧图像列表

        返回:
            list[PIL.Image]: 裁剪后的图像组
        """
        w, h = img_group[0].size
        th, tw = self.size

        out_images = []

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert img.size[0] == w and img.size[1] == h
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    """
    对一组图像进行中心裁剪
    
    参数:
        size (int or tuple): 裁剪目标尺寸
    """

    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        """
        对每张图像执行中心裁剪
        
        参数:
            img_group (list[PIL.Image]): 多帧图像列表

        返回:
            list[PIL.Image]: 裁剪后的图像组
        """
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    """
    对一组图像以 50% 概率进行水平翻转
    
    参数:
        is_flow (bool): 是否为光流数据（翻转时需要反转奇偶通道）
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # 光流图像翻转需反向像素值
            return ret
        else:
            return img_group


class GroupNormalize(object):
    """
    对一组图像进行归一化（减均值除标准差）
    
    参数:
        mean (list[float]): 均值
        std (list[float]): 标准差
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        对输入图像组进行标准化
        
        参数:
            tensor (Tensor): 输入图像张量（N × C × H × W）

        返回:
            Tensor: 归一化后的图像组
        """
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupScale(object):
    """
    对一组图像统一缩放至指定大小
    
    参数:
        size (int or tuple): 缩放目标尺寸
        interpolation: 插值方式，默认为 BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    """
    多尺度裁剪 + 翻转增强组合
    
    参数:
        crop_size (int or tuple): 裁剪尺寸
        scale_size (int): 缩放目标尺寸
    """

    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):
        if self.scale_worker:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = []
        for o_w, o_h in offsets:
            normal_group = []
            flip_group = []
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):
    """
    多尺度随机裁剪（训练常用数据增强策略）
    
    参数:
        input_size (tuple): 输出尺寸
        scales (list): 可选的缩放比例
        max_distort (int): 最大宽高比扭曲程度
        fix_crop (bool): 是否使用固定采样点
        more_fix_crop (bool): 是否增加更多采样点
    """

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):
        im_size = img_group[0].size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        """
        生成多个固定裁剪点用于增强采样
        """
        w_step = (image_w - crop_w) / 4
        h_step = (image_h - crop_h) / 4

        ret = [(0, 0), (4 * w_step, 0), (0, 4 * h_step), (4 * w_step, 4 * h_step), (2 * w_step, 2 * h_step)]

        if more_fix_crop:
            ret += [
                (0, 2 * h_step),
                (4 * w_step, 2 * h_step),
                (2 * w_step, 4 * h_step),
                (2 * w_step, 0 * h_step),
                (1 * w_step, 1 * h_step),
                (3 * w_step, 1 * h_step),
                (1 * w_step, 3 * h_step),
                (3 * w_step, 3 * h_step)
            ]

        return ret

class GroupResize(object):
    """
    对一组图像统一调整尺寸（所有图像保持相同大小）

    参数:
        size (int): 目标边长
        interpolation: 插值方式，默认为双线性插值 BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        """
        执行 resize 操作
        
        参数:
            img_group (list[PIL.Image]): 多帧图像列表

        返回:
            list[PIL.Image]: 缩放后的图像组
        """
        out_group = list()
        for img in img_group:
            out_group.append(img.resize((self.size, self.size), self.interpolation))
        return out_group


class GroupRandomSizedCrop(object):
    """
    随机尺寸和宽高比裁剪 + 缩放至固定尺寸（训练常用的数据增强方法）
    
    参考 Inception 网络使用的策略：
        - 裁剪面积范围：原图面积的 8% ~ 100%
        - 宽高比范围：3/4 ~ 4/3
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        """
        尝试进行随机裁剪，若失败则回退使用缩放 + 中心裁剪
        
        参数:
            img_group (list[PIL.Image]): 图像组

        返回:
            list[PIL.Image]: 经过裁剪和缩放后的图像组
        """
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w  # 随机交换宽高

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))  # 裁剪
                assert img.size == (w, h)
                out_group.append(img.resize((self.size, self.size), self.interpolation))  # 缩放
            return out_group
        else:
            # 回退方案：先缩放再随机裁剪
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class ColorJitter(object):
    """
    对图像组进行随机颜色扰动（亮度、对比度、饱和度、色相）

    参数:
        brightness (float): 亮度扰动幅度
        contrast (float): 对比度扰动幅度
        saturation (float): 饱和度扰动幅度
        hue (float): 色相扰动幅度
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        """
        获取随机参数因子
        
        返回:
            brightness_factor, contrast_factor, saturation_factor, hue_factor
        """
        brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness) if brightness > 0 else None
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast) if contrast > 0 else None
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation) if saturation > 0 else None
        hue_factor = random.uniform(-hue, hue) if hue > 0 else None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        对图像组应用颜色变换
        
        参数:
            clip (list[PIL.Image]): 图像帧序列

        返回:
            list[PIL.Image]: 变换后的图像组
        """
        if isinstance(clip[0], np.ndarray):
            raise TypeError('Color jitter not yet implemented for numpy arrays')
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)  # 打乱顺序以增加多样性

            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    img = func(img)
                jittered_clip.append(img)

            return jittered_clip
        else:
            raise TypeError(f"Expected numpy.ndarray or PIL.Image, got {type(clip[0])}")


class RandomRotation(object):
    """
    对图像组进行随机角度旋转
    
    参数:
        degrees (int or tuple): 旋转角度范围
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number, must be positive')
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence, it must have length 2.')
            self.degrees = degrees

    def __call__(self, clip):
        """
        对图像组执行旋转
        
        参数:
            clip (list[PIL.Image]): 图像帧序列

        返回:
            list[PIL.Image]: 旋转后的图像组
        """
        angle = random.uniform(*self.degrees)
        if isinstance(clip[0], np.ndarray):
            # numpy 支持未实现
            exit()
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError(f"Expected numpy.ndarray or PIL.Image, got {type(clip[0])}")

        return rotated


class Stack(object):
    """
    将图像组堆叠成一个张量（按通道拼接）

    参数:
        roll (bool): 是否将 RGB 转为 BGR（用于 OpenCV 格式兼容）
    """

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        """
        堆叠多帧图像为一个张量
        
        参数:
            img_group (list[PIL.Image]): 图像帧列表

        返回:
            np.ndarray: 堆叠后的图像张量
        """
        if img_group[0].mode == 'L' or img_group[0].mode == 'F':
            # 灰度图：添加通道维度后堆叠
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                # RGB -> BGR（用于 OpenCV 兼容）
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """
    将图像组从 PIL.Image 或 numpy.ndarray 转换为 PyTorch Tensor

    输入范围: [0, 255]
    输出范围: [0.0, 1.0]

    参数:
        div (bool): 是否除以 255 归一化
    """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        """
        转换函数
        
        参数:
            pic (PIL.Image or numpy.ndarray): 输入图像组

        返回:
            torch.Tensor: CHW 格式的浮点张量
        """
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # 转换为 ByteTensor 并重排为 CHW
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()

        return img.to(torch.float32).div_(255) if self.div else img.to(torch.float32)


class IdentityTransform(object):
    """
    恒等变换：不执行任何操作，占位用
    """

    def __call__(self, data):
        return data