'''
Author: Lin Zhu
Date: 2024-06-06 20:16:07
LastEditors: Lin Zhu
LastEditTime: 2024-06-06 21:59:23
Description: 
FilePath: \LLaMA-Factory\data\CHIP-CDEE\TrainDataGen.py
'''
import os,json,codecs

# 获取当前工作目录
current_path = os.getcwd()

promptText = """
你是一个医疗行业数据提取助手。你能从从中文电子病历中挖掘出临床发现事件。即给定一段现病史或者医学影像所见报告，要求从中抽取临床发现事件的四个属性: 解剖部位、主体词、描述词，以及发生状态:
    主体词：指患者的电子病历中的疾病名称或者由疾病引发的症状，也包括患者的一般情况如饮食，二便，睡眠等。主体词尽可能完整并是专有名词，比如“麻木， 疼痛，发烧，囊肿”等；专有名词，如“头晕”，晕只能发生在头部，“胸闷”，闷只能发生在胸部，所以不进行拆分，保留完整的专有名词。涉及泛化的症状不做标注，如“无其他不适”，句子中的“不适”不需要标注，只针对具体的进行标注。注意：有较小比例的主体词会映射到ICD标准术语，所使用的ICD的版本为“国际疾病分类 ICD-10北京临床版v601.xIsx”(见下载文件)。
    描述词：对主体词的发生时序特征、轻重程度、形态颜色等多个维度的刻画，也包括疾病的起病缓急、突发。
    解剖部位：指主体词发生在患者的身体部位，也包括组织，细胞，系统等，也包括部位的方向和数量。
    发生状态：“不确定”或“否定”，肯定的情况不标注发生状态。
针对医疗文本，你将在分析其内容后以json格式标注临床发现事件：
    text: 表示病历或者医学影像报告
    event: 列表结构，由一个或者多个事件四元组组成
        core_name: 主体词，字符串
        tendency: 发生状态，字符串，如果没有发生状态，默认为""
        character: 描述词，列表结构，如果没有描述词，默认为[]
        anatomy_list: 解剖部位，列表结构，如果没有解剖部位，默认为[]
不要提供任何解释，也不要返回任何无关文字，只返回回答变量。
"""

# 假设文件名为 'data.txt'，并且文件中的内容是有效的JSON格式
filename = './data/CHIP-CDEE/CHIP-CDEE_train.json'
outputfile = './data/chatglm3trainning.json'

# 检查文件是否存在
if os.path.exists(outputfile):
    # 删除文件
    os.remove(outputfile)
    print(f"文件 '{outputfile}' 已删除。")
else:
    print(f"文件 '{outputfile}' 不存在。")

# 打开文件并读取内容
with open(filename, 'r', encoding='utf-8') as file:
    content = file.read()

# 将内容转换为JSON
try:
    data = json.loads(content)
except json.JSONDecodeError as e:
    print("转换失败，文件内容可能不是有效的JSON格式。错误信息：", e)

trainingData = []

#In ShareGPT format
""" for i in range(len(data)):
    text = data[i].get('text')
    event = data[i].get('event')
    human = {}
    human["from"] = "human"
    human["value"] = text
    gpt={}
    gpt["from"] = "gpt"
    gpt["value"] = json.dumps(event, ensure_ascii=False, indent=4)
    conversationItem=[]
    conversationItem.append(human)
    conversationItem.append(gpt)
    trainingData.append({"conversations": conversationItem,"system":promptText}) """
    
#In alpaca format
for i in range(len(data)):
    text = data[i].get('text')
    event = data[i].get('event')
    conversation = {}
    conversation["instruction"]="请将下列医疗文本转换为以json格式标注的临床发现事件："
    conversation["input"]=text
    conversation["output"]=json.dumps(event, ensure_ascii=False, indent=4)
    conversation["system"]=promptText
    trainingData.append(conversation)


# 将Python字典转换为JSON格式的字符串
json_data = json.dumps(trainingData, ensure_ascii=False, indent=4)

# 将JSON字符串写入到文件中
with open(outputfile, 'w', encoding='utf-8') as file:
    file.write(json_data)

print("JSON数据已保存到文件 " + outputfile)



