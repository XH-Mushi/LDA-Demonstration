import matplotlib.pyplot as plt
from visualization import LDAVisualizer
import matplotlib
# matplotlib.use('QtAgg') # 或者尝试 'Qt6Agg', 'Qt5Agg'
matplotlib.use('TkAgg')  # 尝试TkAgg后端，看是否能改善卡顿

try:
    # 尝试设置常用中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except Exception as e:
    print(f"无法设置中文字体，某些字符可能无法正确显示: {e}")

if __name__ == "__main__":
    app = LDAVisualizer()
    app.show()  # This calls plt.show() internally
