import pandas as pd

# 第一张表的数据
data1 = [
    ["学院", "得分", "备注"],
    ["光电科学与工程学院", 89, ""],
    ["格拉斯哥学院", 88, ""],
    ["外国语学院", 85, ""],
    ["电子科学与工程学院", 94, ""],
    ["资源与环境学院", 80, ""],
    ["经济与管理学院", 80, ""],
    ["航空航天学院", 82, ""],
    ["公共管理学院", 86, ""],
    ["医学院", 91, ""],
    ["信息与软件工程学院（示范性软件学院）", 90, ""],
    ["物理学院", 85, ""],
    ["自动化工程学院", 83, ""],
    ["信息与通信工程学院", 88, ""],
    ["机械与电气工程学院", 80, ""],
    ["计算机科学与工程学院（网络空间安全学院）", 92, ""],
    ["微电子科学与工程学院（示范性微电子学院）", 90, ""],
    ["材料与能源学院", 75, ""],
    ["英才实验学院", 83, ""]
]

# 第二张表的数据
data2 = [
    ["学院", "得分", "备注"],
    ["光电科学与工程学院", 82, ""],
    ["格拉斯哥学院", 80, ""],
    ["外国语学院", 84, ""],
    ["电子科学与工程学院", 70, ""],
    ["资源与环境学院", 78, ""],
    ["经济与管理学院", 83, ""],
    ["航空航天学院", 86, ""],
    ["公共管理学院", 87.5, ""],
    ["医学院", 88, ""],
    ["信息与软件工程学院（示范性软件学院）", 82, ""],
    ["物理学院", 81, ""],
    ["自动化工程学院", 89, ""],
    ["信息与通信工程学院", 85, ""],
    ["机械与电气工程学院", 92, ""],
    ["计算机科学与工程学院（网络空间安全学院）", 91, ""],
    ["微电子科学与工程学院（示范性微电子学院）", 75, ""],
    ["材料与能源学院", 83, ""],
    ["英才实验学院", 83, ""]
]

# 第三张表的数据（从新图片中提取）
data3 = [
    ["学院", "得分", "备注"],
    ["光电科学与工程学院", 70, ""],
    ["格拉斯哥学院", 89, ""],
    ["外国语学院", 88, ""],
    ["电子科学与工程学院", 95, ""],
    ["资源与环境学院", 85, ""],
    ["经济与管理学院", 85, ""],
    ["航空航天学院", 85, ""],
    ["公共管理学院", 90, ""],
    ["医学院", 93, ""],
    ["信息与软件工程学院（示范性软件学院）", 93, ""],
    ["物理学院", 87, ""],
    ["自动化工程学院", 89, ""],
    ["信息与通信工程学院", 92, ""],
    ["机械与电气工程学院", 94, ""],
    ["计算机科学与工程学院（网络空间安全学院）", 95, ""],
    ["微电子科学与工程学院（示范性微电子学院）", 75, ""],
    ["材料与能源学院", 75, ""],
    ["英才实验学院", 86, ""]
]

# 创建第一个工作表的DataFrame
df1 = pd.DataFrame(data1[1:], columns=data1[0])
scores1 = df1["得分"].tolist()
average1 = sum(scores1) / len(scores1)
average_row1 = pd.DataFrame([["平均分", round(average1, 2), ""]], columns=data1[0])
df1 = pd.concat([df1, average_row1], ignore_index=True)

# 创建第二个工作表的DataFrame
df2 = pd.DataFrame(data2[1:], columns=data2[0])
scores2 = df2["得分"].tolist()
average2 = sum(scores2) / len(scores2)
average_row2 = pd.DataFrame([["平均分", round(average2, 2), ""]], columns=data2[0])
df2 = pd.concat([df2, average_row2], ignore_index=True)

# 创建第三个工作表的DataFrame
df3 = pd.DataFrame(data3[1:], columns=data3[0])
scores3 = df3["得分"].tolist()
average3 = sum(scores3) / len(scores3)
average_row3 = pd.DataFrame([["平均分", round(average3, 2), ""]], columns=data3[0])
df3 = pd.concat([df3, average_row3], ignore_index=True)

# 将三个工作表写入同一个Excel文件
output_file = r"D:\桌面\学院得分统计表.xlsx"
with pd.ExcelWriter(output_file) as writer:
    df1.to_excel(writer, sheet_name="第一次评分", index=False)
    df2.to_excel(writer, sheet_name="第二次评分", index=False)
    df3.to_excel(writer, sheet_name="第三次评分", index=False)

print(f"Excel文件已成功生成：{output_file}")
print(f"\n=== 第一次评分统计 ===")
print(f"学院数量：{len(scores1)}")
print(f"最高分：{max(scores1)}")
print(f"最低分：{min(scores1)}")
print(f"平均分：{round(average1, 2)}")
print(f"\n=== 第二次评分统计 ===")
print(f"学院数量：{len(scores2)}")
print(f"最高分：{max(scores2)}")
print(f"最低分：{min(scores2)}")
print(f"平均分：{round(average2, 2)}")
print(f"\n=== 第三次评分统计 ===")
print(f"学院数量：{len(scores3)}")
print(f"最高分：{max(scores3)}")
print(f"最低分：{min(scores3)}")
print(f"平均分：{round(average3, 2)}")
