# 导入必要的库
import cv2  # OpenCV，用于图像处理
import numpy as np  # NumPy，用于数组和数学运算


## 数据增强函数定义

def identity_func(img):
    """
    恒等变换：不进行任何操作，直接返回原图。
    
    参数:
        img: 输入图像 (np.array)
        
    返回:
        原始图像
    """
    return img


def autocontrast_func(img, cutoff=0):
    """
    自动对比度调整，模拟 PIL.ImageOps.autocontrast 的行为。

    参数:
        img: 输入图像 (np.array)
        cutoff: 直方图裁剪比例，用于去除极值点

    返回:
        对比度增强后的图像
    """
    n_bins = 256  # 图像直方图的bin数量（0-255）

    def tune_channel(ch):
        """
        对单个通道进行自动对比度调整。
        """
        n = ch.size  # 当前通道像素总数
        cut = cutoff * n // 100  # 根据百分比计算裁剪像素数

        if cut == 0:
            high, low = ch.max(), ch.min()  # 不裁剪时取最大最小值
        else:
            hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])  # 计算直方图
            low = np.argwhere(np.cumsum(hist) > cut)  # 累积直方图第一个超过cut的位置
            low = 0 if low.shape[0] == 0 else low[0]
            high = np.argwhere(np.cumsum(hist[::-1]) > cut)  # 反向累积直方图第一个超过cut的位置
            high = n_bins - 1 if high.shape[0] == 0 else n_bins - 1 - high[0]

        if high <= low:
            table = np.arange(n_bins)  # 如果high <= low，则不需要映射
        else:
            scale = (n_bins - 1) / (high - low)  # 缩放因子
            offset = -low * scale
            table = np.arange(n_bins) * scale + offset  # 构建映射表
            table[table < 0] = 0
            table[table > n_bins - 1] = n_bins - 1
        table = table.clip(0, 255).astype(np.uint8)  # 限制在0~255范围内并转为uint8
        return table[ch]  # 应用映射表到当前通道

    channels = [tune_channel(ch) for ch in cv2.split(img)]  # 分别对每个通道进行处理
    out = cv2.merge(channels)  # 合并通道
    return out


def equalize_func(img):
    """
    直方图均衡化，模拟 PIL.ImageOps.equalize 的行为。

    参数:
        img: 输入图像 (np.array)

    返回:
        直方图均衡化后的图像
    """
    n_bins = 256  # 图像直方图的bin数量（0-255）

    def tune_channel(ch):
        """
        对单个通道进行直方图均衡化。
        """
        hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])  # 计算直方图
        non_zero_hist = hist[hist != 0].reshape(-1)  # 过滤掉零值
        step = np.sum(non_zero_hist[:-1]) // (n_bins - 1)  # 步长，用于构建查找表
        if step == 0: return ch  # 如果step为0则跳过处理

        n = np.empty_like(hist)
        n[0] = step // 2
        n[1:] = hist[:-1]

        table = (np.cumsum(n) // step).clip(0, 255).astype(np.uint8)  # 构建映射表
        return table[ch]  # 应用映射表

    channels = [tune_channel(ch) for ch in cv2.split(img)]  # 分别对每个通道进行处理
    out = cv2.merge(channels)  # 合并通道
    return out


def rotate_func(img, degree, fill=(0, 0, 0)):
    """
    图像旋转，模拟 PIL.Image.rotate 行为。

    参数:
        img: 输入图像 (np.array)
        degree: 旋转角度（不是弧度）
        fill: 旋转后空白区域填充颜色

    返回:
        旋转后的图像
    """
    H, W = img.shape[0], img.shape[1]  # 获取图像高宽
    center = W / 2, H / 2  # 中心坐标
    M = cv2.getRotationMatrix2D(center, degree, 1)  # 获取旋转矩阵
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill)  # 执行仿射变换
    return out


def solarize_func(img, thresh=128):
    """
    图像反色处理，模拟 PIL.ImageOps.posterize 行为。

    参数:
        img: 输入图像 (np.array)
        thresh: 阈值，高于该值的像素取反

    返回:
        反色处理后的图像
    """
    table = np.array([el if el < thresh else 255 - el for el in range(256)])  # 构建映射表
    table = table.clip(0, 255).astype(np.uint8)  # 限制范围并转换类型
    out = table[img]  # 应用映射表
    return out


def color_func(img, factor):
    """
    调整图像色彩饱和度，模拟 PIL.ImageEnhance.Color 行为。

    参数:
        img: 输入图像 (np.array)
        factor: 色彩增强因子，1.0表示不变，>1.0增强，<1.0减弱

    返回:
        色彩调整后的图像
    """
    M = (
        np.float32([
            [0.886, -0.114, -0.114],
            [-0.587, 0.413, -0.587],
            [-0.299, -0.299, 0.701]]) * factor
        + np.float32([[0.114], [0.587], [0.299]])
    )  # 构造变换矩阵
    out = np.matmul(img, M).clip(0, 255).astype(np.uint8)  # 矩阵乘法并转换类型
    return out


def contrast_func(img, factor):
    """
    调整图像对比度，模拟 PIL.ImageEnhance.Contrast 行为。

    参数:
        img: 输入图像 (np.array)
        factor: 对比度增强因子，1.0表示不变，>1.0增强，<1.0减弱

    返回:
        对比度调整后的图像
    """
    mean = np.sum(np.mean(img, axis=(0, 1)) * np.array([0.114, 0.587, 0.299]))  # 计算加权平均灰度
    table = np.array([(el - mean) * factor + mean for el in range(256)]).clip(0, 255).astype(np.uint8)  # 构建映射表
    out = table[img]  # 应用映射表
    return out


def brightness_func(img, factor):
    """
    调整图像亮度，模拟 PIL.ImageEnhance.Contrast 行为。

    参数:
        img: 输入图像 (np.array)
        factor: 亮度增强因子，1.0表示不变，>1.0变亮，<1.0变暗

    返回:
        亮度调整后的图像
    """
    table = (np.arange(256, dtype=np.float32) * factor).clip(0, 255).astype(np.uint8)  # 构建映射表
    out = table[img]  # 应用映射表
    return out


def sharpness_func(img, factor):
    """
    调整图像锐度，模拟 PIL.ImageEnhance.Sharpness 行为。

    参数:
        img: 输入图像 (np.array)
        factor: 锐度增强因子，1.0表示不变，>1.0增强，<1.0减弱

    返回:
        锐度调整后的图像
    """
    kernel = np.ones((3, 3), dtype=np.float32)  # 定义卷积核
    kernel[1][1] = 5  # 中心权重加强
    kernel /= 13  # 归一化
    degenerate = cv2.filter2D(img, -1, kernel)  # 卷积模糊图像

    if factor == 0.0:
        out = degenerate  # 锐度为0则返回模糊图像
    elif factor == 1.0:
        out = img  # 锐度为1则返回原图
    else:
        out = img.astype(np.float32)
        degenerate = degenerate.astype(np.float32)[1:-1, 1:-1, :]  # 截取中间部分
        out[1:-1, 1:-1, :] = degenerate + factor * (out[1:-1, 1:-1, :] - degenerate)  # 混合
        out = out.astype(np.uint8)  # 转换回uint8
    return out


def shear_x_func(img, factor, fill=(0, 0, 0)):
    """
    X方向剪切变换。

    参数:
        img: 输入图像 (np.array)
        factor: 剪切因子
        fill: 填充颜色

    返回:
        剪切变换后的图像
    """
    H, W = img.shape[0], img.shape[1]  # 获取图像尺寸
    M = np.float32([[1, factor, 0], [0, 1, 0]])  # 构造变换矩阵
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)  # 仿射变换
    return out


def translate_x_func(img, offset, fill=(0, 0, 0)):
    """
    X方向平移变换，模拟 PIL.Image.transform 行为。

    参数:
        img: 输入图像 (np.array)
        offset: 平移偏移量
        fill: 填充颜色

    返回:
        平移变换后的图像
    """
    H, W = img.shape[0], img.shape[1]  # 获取图像尺寸
    M = np.float32([[1, 0, -offset], [0, 1, 0]])  # 构造变换矩阵
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)  # 仿射变换
    return out


def translate_y_func(img, offset, fill=(0, 0, 0)):
    """
    Y方向平移变换，模拟 PIL.Image.transform 行为。

    参数:
        img: 输入图像 (np.array)
        offset: 平移偏移量
        fill: 填充颜色

    返回:
        平移变换后的图像
    """
    H, W = img.shape[0], img.shape[1]  # 获取图像尺寸
    M = np.float32([[1, 0, 0], [0, 1, -offset]])  # 构造变换矩阵
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)  # 仿射变换
    return out

def posterize_func(img, bits):
    """
    图像色彩位数降低（色调分离），模拟 PIL.ImageOps.posterize 行为。

    参数:
        img: 输入图像 (np.array)
        bits: 保留的位数，范围是 0~8。值越小图像颜色越少

    返回:
        色调减少后的图像
    """
    # 使用按位与操作将每个像素值的低 (8 - bits) 位置零，从而减少颜色数量
    out = np.bitwise_and(img, np.uint8(255 << (8 - bits)))
    return out


def shear_y_func(img, factor, fill=(0, 0, 0)):
    """
    Y方向剪切变换。

    参数:
        img: 输入图像 (np.array)
        factor: 剪切因子
        fill: 填充颜色

    返回:
        剪切变换后的图像
    """
    H, W = img.shape[0], img.shape[1]  # 获取图像高宽
    M = np.float32([[1, 0, 0], [factor, 1, 0]])  # 构造Y方向剪切矩阵
    # 执行仿射变换并使用指定填充色填充空白区域
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def cutout_func(img, pad_size, replace=(0, 0, 0)):
    """
    随机遮挡图像的一部分（CutOut 数据增强方法）。

    参数:
        img: 输入图像 (np.array)
        pad_size: 遮挡区域的一半大小（即边长的一半）
        replace: 替换遮挡区域的颜色，默认为黑色 (0, 0, 0)

    返回:
        经过随机遮挡处理的图像
    """
    replace = np.array(replace, dtype=np.uint8)  # 将替换颜色转换为 numpy 数组
    H, W = img.shape[0], img.shape[1]  # 获取图像尺寸
    rh, rw = np.random.random(2)  # 随机生成中心点坐标比例
    pad_size = pad_size // 2  # 计算实际要遮挡区域的大小
    ch, cw = int(rh * H), int(rw * W)  # 计算遮挡区域中心坐标

    # 确定遮挡区域边界，防止超出图像范围
    x1, x2 = max(ch - pad_size, 0), min(ch + pad_size, H)
    y1, y2 = max(cw - pad_size, 0), min(cw + pad_size, W)

    out = img.copy()  # 复制原图以避免修改原始图像
    out[x1:x2, y1:y2, :] = replace  # 填充遮挡区域
    return out


### 每种数据增强操作对应的参数映射函数 ###

def enhance_level_to_args(MAX_LEVEL):
    """
    将增强级别转换为增强因子。
    
    参数:
        MAX_LEVEL: 最大增强级别
        
    返回:
        level_to_args 函数，输入level返回对应参数
    """
    def level_to_args(level):
        return ((level / MAX_LEVEL) * 1.8 + 0.1,)  # 映射到 [0.1, 1.9] 区间
    return level_to_args


def shear_level_to_args(MAX_LEVEL, replace_value):
    """
    将剪切级别转换为剪切因子和填充颜色。
    
    参数:
        MAX_LEVEL: 最大剪切级别
        replace_value: 填充颜色
        
    返回:
        level_to_args 函数，输入level返回对应参数
    """
    def level_to_args(level):
        level = (level / MAX_LEVEL) * 0.3  # 映射到 [0, 0.3] 区间
        if np.random.random() > 0.5: level = -level  # 有50%概率反向剪切
        return (level, replace_value)  # 返回剪切因子和填充色
    return level_to_args


def translate_level_to_args(translate_const, MAX_LEVEL, replace_value):
    """
    将平移级别转换为平移偏移量和填充颜色。
    
    参数:
        translate_const: 平移常数，表示最大偏移量
        MAX_LEVEL: 最大增强级别
        replace_value: 填充颜色
        
    返回:
        level_to_args 函数，输入level返回对应参数
    """
    def level_to_args(level):
        level = (level / MAX_LEVEL) * float(translate_const)  # 映射到 [0, translate_const]
        if np.random.random() > 0.5: level = -level  # 有50%概率反向平移
        return (level, replace_value)  # 返回偏移量和填充色
    return level_to_args


def cutout_level_to_args(cutout_const, MAX_LEVEL, replace_value):
    """
    将 CutOut 级别转换为遮挡大小和填充颜色。
    
    参数:
        cutout_const: CutOut 的最大遮挡大小
        MAX_LEVEL: 最大增强级别
        replace_value: 填充颜色
        
    返回:
        level_to_args 函数，输入level返回对应参数
    """
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * cutout_const)  # 映射为整数大小
        return (level, replace_value)  # 返回遮挡大小和填充色
    return level_to_args


def solarize_level_to_args(MAX_LEVEL):
    """
    将 Solarize 级别转换为阈值参数。
    
    参数:
        MAX_LEVEL: 最大增强级别
        
    返回:
        level_to_args 函数，输入level返回对应参数
    """
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 256)  # 映射到 [0, 256] 区间
        return (level,)  # 返回阈值
    return level_to_args


def none_level_to_args(level):
    """
    不需要额外参数的操作的参数映射函数。
    
    参数:
        level: 忽略此参数
        
    返回:
        空元组 ()
    """
    return ()


def posterize_level_to_args(MAX_LEVEL):
    """
    将 Posterize 级别转换为保留位数。
    
    参数:
        MAX_LEVEL: 最大增强级别
        
    返回:
        level_to_args 函数，输入level返回对应参数
    """
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 4)  # 映射到 [0, 4] 区间
        return (level,)  # 返回保留位数
    return level_to_args


def rotate_level_to_args(MAX_LEVEL, replace_value):
    """
    将旋转级别转换为角度和填充颜色。
    
    参数:
        MAX_LEVEL: 最大旋转级别
        replace_value: 填充颜色
        
    返回:
        level_to_args 函数，输入level返回对应参数
    """
    def level_to_args(level):
        level = (level / MAX_LEVEL) * 30  # 映射到 [0, 30] 度
        if np.random.random() < 0.5: level = -level  # 有50%概率反向旋转
        return (level, replace_value)  # 返回角度和填充色
    return level_to_args


# 定义所有支持的数据增强函数字典
func_dict = {
    'Identity': identity_func,
    'AutoContrast': autocontrast_func,
    'Equalize': equalize_func,
    'Rotate': rotate_func,
    'Solarize': solarize_func,
    'Color': color_func,
    'Contrast': contrast_func,
    'Brightness': brightness_func,
    'Sharpness': sharpness_func,
    'ShearX': shear_x_func,
    'TranslateX': translate_x_func,
    'TranslateY': translate_y_func,
    'Posterize': posterize_func,
    'ShearY': shear_y_func,
}

# 设置默认参数
translate_const = 10  # 平移变换的最大偏移量
MAX_LEVEL = 10  # 最大增强级别
replace_value = (128, 128, 128)  # 默认填充颜色为灰色

# 定义每种操作对应的参数生成函数字典
arg_dict = {
    'Identity': none_level_to_args,
    'AutoContrast': none_level_to_args,
    'Equalize': none_level_to_args,
    'Rotate': rotate_level_to_args(MAX_LEVEL, replace_value),
    'Solarize': solarize_level_to_args(MAX_LEVEL),
    'Color': enhance_level_to_args(MAX_LEVEL),
    'Contrast': enhance_level_to_args(MAX_LEVEL),
    'Brightness': enhance_level_to_args(MAX_LEVEL),
    'Sharpness': enhance_level_to_args(MAX_LEVEL),
    'ShearX': shear_level_to_args(MAX_LEVEL, replace_value),
    'TranslateX': translate_level_to_args(translate_const, MAX_LEVEL, replace_value),
    'TranslateY': translate_level_to_args(translate_const, MAX_LEVEL, replace_value),
    'Posterize': posterize_level_to_args(MAX_LEVEL),
    'ShearY': shear_level_to_args(MAX_LEVEL, replace_value),
}


class RandomAugment(object):
    """
    实现 RandAugment 数据增强策略。
    
    参数:
        N: 每次应用的增强操作数量
        M: 增强强度级别
        isPIL: 是否输入 PIL 图像
        augs: 自定义增强操作列表
    """

    def __init__(self, N=2, M=10, isPIL=False, augs=[]):
        self.N = N  # 每次选择的增强操作数量
        self.M = M  # 增强级别
        self.isPIL = isPIL  # 是否为 PIL 图像
        if augs:
            self.augs = augs  # 使用自定义增强列表
        else:
            self.augs = list(arg_dict.keys())  # 否则使用默认增强操作列表

    def get_random_ops(self):
        """
        随机选择一组增强操作。
        
        返回:
            ops: 增强操作列表，格式为 [(op_name, prob, level)]
        """
        sampled_ops = np.random.choice(self.augs, self.N)  # 随机选取 N 个操作
        return [(op, 0.5, self.M) for op in sampled_ops]  # 每个操作的概率设为 0.5

    def __call__(self, img):
        """
        对图像执行一组随机增强操作。
        
        参数:
            img: 输入图像
            
        返回:
            增强后的图像
        """
        if self.isPIL:
            img = np.array(img)  # 如果是 PIL 图像，先转为 numpy 数组
            
        ops = self.get_random_ops()  # 获取随机增强操作
        
        for name, prob, level in ops:
            if np.random.random() > prob:  # 根据概率决定是否执行该操作
                continue
            args = arg_dict[name](level)  # 获取操作参数
            img = func_dict[name](img, *args)  # 执行增强操作
        return img


if __name__ == '__main__':
    a = RandomAugment()
    img = np.random.randn(32, 32, 3)  # 生成一个随机图像用于测试
    a(img)  # 测试数据增强