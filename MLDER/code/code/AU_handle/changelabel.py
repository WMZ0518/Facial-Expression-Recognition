# 读取原始文件并修改
input_file = '/home/et24-liax/new_fer/pretrain/code/AU_handle/RAF_AU_descriptions.csv'  # 输入文件路径
output_file = '/home/et24-liax/new_fer/pretrain/code/AU_handle/RAF_AU_descriptions.csv'  # 输出文件路径

# 打开原始文件进行读取
with open(input_file, 'r') as f:
    lines = f.readlines()

# 处理每一行，修改文件名
modified_lines = []
for line in lines:
    

    modified_image_name = line.replace(' person', 'a person')

    # 拼接修改后的行
    modified_line = f"{modified_image_name}"
    modified_lines.append(modified_line)

# 将修改后的内容写入新的文件
with open(output_file, 'w') as f:
    f.writelines(modified_lines)

print(f"File has been modified and saved to {output_file}")
