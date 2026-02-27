import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('/home/et24-liax/new_fer/pretrain/static_dataset/affectnet/affectnet.csv')  # 请使用实际的文件路径
print(df.columns)
# 读取 RAF-label.txt
emotion_dict = {}
emotion_mapping = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt"
}
with open('/home/et24-liax/new_fer/pretrain/static_dataset/affectnet/8cls_train.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        image_name = parts[0]
        emotion_label = int(parts[1])  # 对应的情感标签（数字）
        emotion = emotion_mapping.get(emotion_label, "Unknown Emotion")  # 获取情感标签
        emotion_dict[image_name] = emotion

# 更新的 AU 的语义描述（根据你提供的描述）
AU_descriptions = {
    " AU01_r": ["Inner brow raiser", "Frown", "Eyebrow raised", "Head lifting wrinkles", "Lift eyebrows"],
    " AU02_r": ["Outer brow raiser", "Outer brow lift", "Elevate outer brow", "Outer brow arch"],
    " AU04_r": ["Brow Lowerer", "Frowns furrowed", "Lower eyebrows", "A look of disapproval"],
    " AU05_r": ["Upper Lid Raiser", "Pupil enlargement", "Eyes widened", "Lift upper eyelids", "Raise upper eyelids"],
    " AU06_r": ["Cheek Raiser", "Smile", "Pleasure", "Slight decrease in eyebrows", "Eyes narrowing", "Slightly lower eyebrows"],
    " AU07_r": ["Lid Tightener", "Facial tightness", "Tightening of eyelids"],
    " AU09_r": ["Nose Wrinkler", "Wrinkle the nose", "Curl the nose", "Make a face", "Pucker the nose"],
    " AU10_r": ["Upper Lip Raiser", "Curl the lips upwards", "Upper lip lift", "Lips apart showing teeth"],
    " AU12_r": ["Lip Corner Puller", "Toothy smile", "Grinning", "Big smile", "Show teeth"],
    " AU14_r": ["Dimpler", "Cheek dimple", "Indentation when smiling", "Hollow on the face when smiling"],
    " AU15_r": ["Lip Corner Depressor", "Downturned corners of the mouth", "Downward mouth curvature", "Lower Lip Depressor"],
    " AU17_r": ["Chin Raiser", "Lift the chin", "Chin held high", "Lips arching", "Lips forming an upward curve"],
    " AU20_r": ["Lip stretcher", "Tense lips stretched", "Anxiously stretched lips", "Nasal flaring", "Nostrils enlarge"],
    " AU23_r": ["Lip Tightener", "Tighten the lips", "Purse the lips", "Press the lips together"],
    " AU25_r": ["Lips part", "Open the lips", "Slightly puzzled", "Lips slightly parted"],
    " AU26_r": ["Jaw Drop", "Mouth Stretch", "Open mouth wide", "Wide-mouthed", "Lips elongated"],
    " AU45_r": ["Lip Suck", "Purse lips", "Pucker lips", "Draw in lips", "Bring lips together"]
}

# 筛选 success 为 1 的行
df_success = df[df[' success'] == 1]

# 存储最终的输出
output_data = []

# 遍历每一行
for index, row in df_success.iterrows():
    # 获取当前图像的名称
    frame =int(row['frame'])
    image_name = f"{frame}.jpg"  # 补全为4位数
    '''
    if frame <= 3068:
        image_name = f"test_{frame:04d}_aligned.jpg"  # 补全为4位数
    else:
        image_name = f"train_{frame:05d}_aligned.jpg"  # 补全为5位数
    # 获取对应的情感标签
    '''
    emotion = emotion_dict.get(image_name, "Unknown Emotion")
    
    # 找出强度最高的 3 个 AU
    aus = row[5:22]  # 从 AU01_r 到 AU15_r 列
    top_3_aus = aus.nlargest(3)
    
    # 创建描述
    description_parts = [f"a person with {emotion} emotions, with"]
    
    for au, intensity in top_3_aus.items():
        au_description = AU_descriptions.get(au, ["Unknown AU"])[0]  # 获取 AU 的描述
        description_parts.append(f"{au_description}")
    
    # 合成完整的描述
    description = ", ".join(description_parts) + "."
    
    # 将图像名称和描述写入 output_data
    output_data.append(f'{image_name} {description}')

# 将结果保存到 CSV 文件
output_df = pd.DataFrame(output_data, columns=["Image_Description"])
output_df.to_csv('affectnet_AU_descriptions.csv', index=False, header=False)

print("Descriptions have been saved to affectnet1_AU_descriptions.csv.")