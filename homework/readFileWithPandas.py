import pandas as pd

csvFile = pd.read_csv("/Users/linlab2024/Desktop/temp_coding/aiPrinciple/homework/DTST20230500090.csv")
# print(csvFile.to_string())

with open("/Users/linlab2024/Desktop/temp_coding/aiPrinciple/homework/DTST20230500090.xml", "r", encoding="utf-8-sig") as f:
    xmlFile = pd.read_xml(f)

#print(xmlFile.to_string())


# 加入 encoding='utf-8-sig' 處理 UTF-8 with BOM 的情況
jsonFile = pd.read_json(
    "/Users/linlab2024/Desktop/temp_coding/aiPrinciple/homework/DTST20230500090.json",
    encoding='utf-8-sig'
)
#print(jsonFile.to_string())
