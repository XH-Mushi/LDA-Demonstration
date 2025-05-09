# LDA 可视化计算器

## 项目概述

这是一个线性判别分析 (Linear Discriminant Analysis, LDA) 的可视化工具，旨在通过交互式界面帮助用户理解和探索 LDA 的核心概念。该工具允许用户通过滑块动态调整参数（如样本数量和均值），实时查看 LDA 在二维数据上的投影结果以及相关散度矩阵。

<img src="screenshot.png" alt="LDA可视化界面" width="800"/>

## 功能特点

- **交互式参数调整**：通过滑块实时调整两个类别的样本数量和均值
- **二维数据可视化**：直观展示原始二维数据分布和 LDA 投影方向
- **一维投影结果**：展示数据沿 LDA 方向投影后的一维分布
- **散度矩阵显示**：实时计算并显示类内散度矩阵 (S_w) 和类间散度矩阵 (S_b)
- **性能优化**：仅在滑块拖动完成后更新图像，减少计算负担
- **完整中文支持**：界面元素支持中文显示

## 安装步骤

### 方法一：直接使用可执行文件（Windows 用户）

1. 下载[LDA_Visualizer.zip](https://github.com/XH-Mushi/LDA-Demonstration/releases)
2. 解压下载的 ZIP 文件
3. 双击 `LDA可视化计算器.exe` 运行程序

### 方法二：从源代码运行

#### 前提条件

- Python 3.6 或更高版本
- pip 包管理器

#### 安装依赖

1. 克隆本仓库：

   ```bash
   git clone https://github.com/XH-Mushi/LDA-Demonstration.git
   cd LDA-Demonstration
   ```

2. 安装必要的 Python 包：
   ```bash
   pip install numpy matplotlib
   ```

## 使用方法

运行主程序文件：

```bash
python main.py
```

程序启动后，您可以：

1. 通过滑块调整各参数：
   - `N_1` 和 `N_2`：调整类别 1 和类别 2 的样本数量
   - `μ_1x` 和 `μ_1y`：调整类别 1 的平均值 (x 和 y 坐标)
   - `μ_2x` 和 `μ_2y`：调整类别 2 的平均值 (x 和 y 坐标)
2. 图表会在滑块释放后自动更新，显示：
   - 二维空间中两类数据的分布情况
   - LDA 投影方向（绿色线）
   - 数据沿投影方向的一维分布
   - 右侧面板中的类内散度矩阵 (S_w) 和类间散度矩阵 (S_b)

## 项目结构

项目采用模块化设计，包含以下文件：

- `main.py`：程序入口点，负责初始化
- `config.py`：存储默认参数和配置常量
- `data_processing.py`：包含 LDA 核心计算逻辑
- `visualization.py`：负责交互式界面和可视化

## 技术细节

LDA 是一种常用的降维和分类方法，它寻找最能区分不同类别数据的投影方向。本项目实现了 LDA 的两个关键步骤：

1. 计算类内散度矩阵 S_w 和类间散度矩阵 S_b
2. 求解 S_w^(-1)S_b 的特征向量，找到最优投影方向

项目使用 Matplotlib 库构建交互式用户界面，可以直观地展示这一过程。

## 许可证

[MIT](LICENSE)

## 联系方式

如有任何问题或建议，请通过 GitHub Issues 或 Pull Requests 提交。

---

项目基于 Python 和 Matplotlib 实现，旨在教育目的。欢迎贡献和改进！
