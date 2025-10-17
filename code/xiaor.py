import matplotlib.pyplot as plt

# 数据准备
x_labels = ["[1/6,1/3]", "[1/5,1/2]", "[1/4,2/3]", "[1/3,3/4]", "[1/2,4/5]", "[2/3,5/6]"]
x = range(len(x_labels))  # 数值型横坐标

# 各数据集的值
sandiego = [0.99104, 0.99729, 0.99622, 0.99836, 0.99845, 0.99816]
airport4 = [0.99438, 0.99531, 0.99192, 0.99659, 0.99744, 0.99731]
beach1 = [0.99678, 0.99567, 0.99651, 0.99447, 0.99684, 0.99544]
urban2 = [0.99209, 0.99775, 0.99304, 0.99913, 0.9991, 0.99896]

# 绘图
# 更新图表，增加纵坐标范围，放大字体
# 更新横坐标为水平显示
plt.figure(figsize=(10, 6))

# 绘制数据
plt.plot(x, sandiego, marker='o', label='Sandiego')
plt.plot(x, airport4, marker='s', label='Airport4')
plt.plot(x, beach1, marker='^', label='Beach1')
plt.plot(x, urban2, marker='d', label='Urban2')

# 设置图表
plt.xticks(x, x_labels, fontsize=18)  # 横坐标水平显示，调整字体大小
plt.yticks(fontsize=18)  # 纵坐标字体
plt.xlabel("k", fontsize=20)  # 放大横坐标标签字体
plt.ylabel("Auccarcy", fontsize=20)  # 放大纵坐标标签字体
#plt.title("Comparison of Different Datasets", fontsize=16)  # 放大标题字体
plt.legend(fontsize=12)  # 放大图例字体
plt.grid(alpha=0.5)
plt.tight_layout()

# 显示更新后的图表
plt.show()