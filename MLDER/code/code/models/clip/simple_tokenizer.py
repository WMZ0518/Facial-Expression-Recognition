import gzip  # 导入gzip模块，用于读取压缩文件
import html  # 导入html模块，用于处理HTML转义字符
import os  # 导入os模块，用于操作操作系统路径和文件
from functools import lru_cache  # 导入lru_cache装饰器，用于缓存函数调用结果

import ftfy  # 导入ftfy库，用于修复损坏的UTF-8字符
import regex as re  # 导入regex模块并重命名为re，支持更强大的正则表达式功能


@lru_cache()  # 使用lru_cache缓存default_bpe()函数的结果，避免重复计算
def default_bpe():
    """
    获取默认BPE词汇表的路径。
    
    返回:
        str: BPE词汇表文件路径
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()  # 使用lru_cache缓存bytes_to_unicode()函数的结果，避免重复计算
def bytes_to_unicode():
    """
    构建UTF-8字节到Unicode字符的映射表。

    这个映射表用于将原始字节转换为Unicode字符串以便进行可逆的BPE编码。
    避免映射到空白/控制字符（这些字符会导致BPE处理出错）。
    
    返回:
        dict: 字节值到Unicode字符的映射字典
    """
    # 创建基础字符集：ASCII标点符号和扩展拉丁字符
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    
    # 扩展字符集以覆盖所有可能的256个字节值
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)  # 使用高位Unicode码位避免冲突
            n += 1
            
    # 将数字转换为对应的Unicode字符
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))  # 返回字节到Unicode字符的映射字典


def get_pairs(word):
    """
    获取单词中相邻字符对的集合。
    
    参数:
        word (tuple): 表示单词的字符元组
        
    返回:
        set: 相邻字符对的集合
    """
    pairs = set()
    prev_char = word[0]  # 初始化前一个字符
    for char in word[1:]:  # 遍历后续字符
        pairs.add((prev_char, char))  # 添加字符对到集合
        prev_char = char  # 更新前一个字符
    return pairs


def basic_clean(text):
    """
    基础文本清理。
    
    参数:
        text (str): 输入文本
        
    返回:
        str: 清理后的文本
    """
    text = ftfy.fix_text(text)  # 修复损坏的UTF-8字符
    text = html.unescape(html.unescape(text))  # 双重HTML解码
    return text.strip()  # 去除首尾空格


def whitespace_clean(text):
    """
    空格清理，将任意长度的空白字符替换为单个空格。
    
    参数:
        text (str): 输入文本
        
    返回:
        str: 处理后的文本
    """
    text = re.sub(r'\s+', ' ', text)  # 替换连续空格为单个空格
    return text.strip()  # 去除首尾空格



class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        """
        初始化BytePairEncoding(BPE)处理器。
    
        参数:
        bpe_path: str - BPE合并表的文件路径。默认为通过default_bpe()函数获取的路径。
        """
        # 初始化字节编码器，将字节映射到对应的unicode字符
        self.byte_encoder = bytes_to_unicode()
        # 初始化字节解码器，将unicode字符映射回对应的字节
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    
        # 读取BPE合并表文件，解压并按行分割
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        # 选取合并表中的特定部分，用于构建词汇表
        merges = merges[1:49152-256-2+1]
        # 将合并表中的每一行分割成元组，用于后续处理
        merges = [tuple(merge.split()) for merge in merges]
    
        # 初始化词汇表，起始为所有字节对应的unicode字符
        vocab = list(bytes_to_unicode().values())
        # 扩展词汇表，添加每个字符后跟'</w>'的项，表示字词结束
        vocab = vocab + [v+'</w>' for v in vocab]
        # 根据合并表进一步扩展词汇表，添加新的合并项
        for merge in merges:
            vocab.append(''.join(merge))
        # 在词汇表中添加特殊标记，表示文本的开始和结束
        vocab.extend(['<|startoftext|>', ''])
    
        # 初始化编码器，将词汇表中的每一项映射到一个唯一的整数
        self.encoder = dict(zip(vocab, range(len(vocab))))
        # 初始化解码器，将整数映射回词汇表中的项
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 初始化BPE合并表的排名，用于快速查找合并项的顺序
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # 初始化缓存，用于存储特殊标记的映射，以提高处理速度
        self.cache = {'<|startoftext|>': '<|startoftext|>', '': ''}
        # 初始化正则表达式模式，用于文本的分词处理
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        """
        使用字节对编码（BPE）算法对给定的token进行编码。
        
        如果token在缓存中，则直接返回缓存中的结果。
        否则，将token处理为一个元组形式的单词，并获取其所有字符对。
        如果没有字符对，则返回token加上结束符号'</w>'。
        如果有字符对，則进入循环，找到频率最低的字符对，并用新的字符替换它们。
        重复此过程，直到不能再合并字符对为止。
        最后，将编码后的token存储在缓存中并返回。
        
        参数:
        token (str): 需要编码的字符串token。
        
        返回:
        str: 编码后的字符串。
        """
        # 检查缓存中是否已有该token的编码结果，如果有则直接返回
        if token in self.cache:
            return self.cache[token]
        
        # 将token转换为元组形式的单词，并在最后一个字符上添加结束符号'</w>'
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        
        # 获取单词中所有的字符对
        pairs = get_pairs(word)
    
        # 如果没有字符对，则直接返回token加上结束符号'</w>'
        if not pairs:
            return token+'</w>'
    
        # 进入循环，寻找频率最低的字符对并合并它们
        while True:
            # 找到频率最低的字符对
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            # 如果最低频率的字符对不在bpe_ranks中，则退出循环
            if bigram not in self.bpe_ranks:
                break
            
            # 分解字符对为两个字符
            first, second = bigram
            
            # 初始化新单词列表和索引i
            new_word = []
            i = 0
            
            # 遍历单词中的每个字符，尝试合并字符对
            while i < len(word):
                try:
                    # 寻找第一个字符在单词中的位置
                    j = word.index(first, i)
                    # 将i到j之间的字符添加到新单词中
                    new_word.extend(word[i:j])
                    i = j
                except:
                    # 如果找不到字符，则将剩余的字符添加到新单词中，并退出循环
                    new_word.extend(word[i:])
                    break
                
                # 如果当前位置的字符是第一个字符，且下一个字符是第二个字符，则合并它们
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    # 否则，将当前字符添加到新单词中，并移动到下一个字符
                    new_word.append(word[i])
                    i += 1
            
            # 将新单词转换为元组形式，并更新单词变量
            new_word = tuple(new_word)
            word = new_word
            
            # 如果单词只有一个字符，则退出循环
            if len(word) == 1:
                break
            else:
                # 否则，获取新单词中的所有字符对，并继续循环
                pairs = get_pairs(word)
        
        # 将编码后的单词转换为字符串，并存储在缓存中
        word = ' '.join(word)
        self.cache[token] = word
        
        # 返回编码后的字符串
        return word

    def encode(self, text):
        """
        将输入文本转换为BPE（Byte Pair Encoding）标记。
    
        该方法首先对文本进行基本的清理和格式化，然后使用正则表达式根据特定模式(self.pat)分割文本。
        对于每个分割出的标记(token)，将其编码为UTF-8字节序列，并使用字节编码器(self.byte_encoder)将每个字节转换为相应的BPE标记。
        最后，使用BPE编码器(self.encoder)将转换后的标记编码为最终的BPE标记序列。
    
        参数:
        text (str): 需要编码的文本。
    
        返回:
        list: 编码后的BPE标记列表。
        """
        # 初始化BPE标记列表
        bpe_tokens = []
        
        # 对文本进行基本清理、格式化并转换为小写
        text = whitespace_clean(basic_clean(text)).lower()
        
        # 使用正则表达式根据特定模式分割文本，并对每个标记进行处理
        for token in re.findall(self.pat, text):
            # 将标记编码为UTF-8字节序列，并使用字节编码器将每个字节转换为相应的BPE标记
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            
            # 使用BPE编码器将转换后的标记编码为最终的BPE标记序列，并添加到列表中
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        
        # 返回编码后的BPE标记列表
        return bpe_tokens

    def decode(self, tokens):
        """
        将给定的tokens解码为文本字符串。
    
        参数:
        tokens (list): 一个包含编码后tokens的列表。
    
        返回:
        str: 解码后的文本字符串。
        """
        # 将tokens列表中每个token转换为对应的字符，使用self.decoder进行映射
        text = ''.join([self.decoder[token] for token in tokens])
        
        # 将转换后的字符列表转换为bytearray，并解码为UTF-8字符串
        # 使用self.byte_decoder将每个字符映射到其对应的字节值，然后解码
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        
        # 返回解码后的文本
        return text
