import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import config
import data_processing


class LDAVisualizer:
    def __init__(self):
        # 使用 GridSpec 调整布局以在右侧留出空间给文本
        self.fig = plt.figure(figsize=(11, 9))  # 调整图形大小 (宽度, 高度)
        gs = self.fig.add_gridspec(
            2, 2, width_ratios=[3, 1.5], height_ratios=[1, 1], wspace=0.35, hspace=0.3)

        self.ax_2d = self.fig.add_subplot(gs[0, 0])
        self.ax_1d = self.fig.add_subplot(gs[1, 0])
        self.ax_text = self.fig.add_subplot(gs[:, 1])  # 文本区域占据右侧整列
        # self.ax_legend = self.fig.add_subplot(gs[1,1]) # Potential dedicated legend space

        # 为滑块调整整体布局 (left, bottom, right, top)
        self.fig.subplots_adjust(left=0.08, bottom=0.32, right=0.95, top=0.92)

        # 初始绘图元素
        self.scatter1, = self.ax_2d.plot(
            [], [], 'o', label='类别 1', alpha=0.7, color='blue')
        self.scatter2, = self.ax_2d.plot(
            [], [], 's', label='类别 2', alpha=0.7, color='red')
        self.mean1_plot, = self.ax_2d.plot(
            [], [], 'X', color='blue', markersize=10, label='均值 1')
        self.mean2_plot, = self.ax_2d.plot(
            [], [], 'X', color='red', markersize=10, label='均值 2')
        self.lda_line, = self.ax_2d.plot(
            [], [], '-', color='green', linewidth=2, label='LDA 投影方向')

        self.projected1_scatter, = self.ax_1d.plot(
            [], [], 'o', color='blue', alpha=0.7)
        self.projected2_scatter, = self.ax_1d.plot(
            [], [], 's', color='red', alpha=0.7)

        # 设置文本区域
        self.ax_text.axis('off')  # 关闭文本区域的坐标轴
        self.sw_text_title = self.ax_text.text(
            0.05, 0.95, "$S_w$ (类内散度矩阵):", transform=self.ax_text.transAxes, fontsize=11, va='top')
        self.sw_text_matrix = self.ax_text.text(
            0.05, 0.85, "", transform=self.ax_text.transAxes, fontsize=10, va='top', fontfamily='monospace')
        self.sb_text_title = self.ax_text.text(
            0.05, 0.65, "$S_b$ (类间散度矩阵):", transform=self.ax_text.transAxes, fontsize=11, va='top')
        self.sb_text_matrix = self.ax_text.text(
            0.05, 0.55, "", transform=self.ax_text.transAxes, fontsize=10, va='top', fontfamily='monospace')

        self._setup_axes()
        self._create_sliders()

        # 在ax_text中创建图例
        handles, labels = self.ax_2d.get_legend_handles_labels()
        # 调整图例位置和标题，可以根据实际显示效果微调 bbox_to_anchor 的y值 (e.g. 0.35, 0.3, 0.25)
        self.legend_in_text_area = self.ax_text.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.30),
                                                       title="图例", fontsize=9, title_fontsize=10)

        self._active_slider_axis = None
        self.fig.canvas.mpl_connect(
            'button_press_event', self._on_figure_button_press)
        self.fig.canvas.mpl_connect(
            'button_release_event', self._on_figure_button_release)

        self.update(None)  # 初始绘图

    def _setup_axes(self):
        self.ax_2d.set_xlabel('$x_1$')
        self.ax_2d.set_ylabel('$x_2$')
        self.ax_2d.set_title('二维原始数据和LDA投影')
        # self.ax_2d.legend() # 不在这里创建图例
        self.ax_2d.grid(True)
        self.ax_2d.set_xlim(-5, 20)
        self.ax_2d.set_ylim(-5, 20)
        self.ax_2d.set_aspect('equal', adjustable='box')

        self.ax_1d.set_xlabel('投影值')
        self.ax_1d.set_yticks([])  # 1D图不需要y轴刻度
        self.ax_1d.set_title('一维投影数据')
        self.ax_1d.grid(True)
        self.ax_1d.set_ylim(-0.5, 0.5)  # 为1D投影点设置固定的Y轴范围

    def _create_sliders(self):
        axcolor = 'lightgoldenrodyellow'
        # 重新调整滑块位置 (left, bottom, width, height) - figure coordinates
        # 底部从0.02开始，每个滑块高0.03，间隙0.01，共6个滑块
        # 总高度: 6*0.03 + 5*0.01 = 0.18 + 0.05 = 0.23.  可以放在 bottom=0.28 区域内
        slider_base_left = 0.20
        slider_width = 0.65  # 对应 gs[:,0] 的宽度

        slider_ax_mu2y = plt.axes(
            [slider_base_left, 0.03, slider_width, 0.03], facecolor=axcolor)
        self.slider_mu2y = Slider(
            slider_ax_mu2y, '$\mu_{2y}$', -5.0, 15.0, valinit=config.DEFAULT_MEAN2[1])

        slider_ax_mu2x = plt.axes(
            [slider_base_left, 0.07, slider_width, 0.03], facecolor=axcolor)
        self.slider_mu2x = Slider(
            slider_ax_mu2x, '$\mu_{2x}$', -5.0, 15.0, valinit=config.DEFAULT_MEAN2[0])

        slider_ax_mu1y = plt.axes(
            [slider_base_left, 0.11, slider_width, 0.03], facecolor=axcolor)
        self.slider_mu1y = Slider(
            slider_ax_mu1y, '$\mu_{1y}$', -5.0, 15.0, valinit=config.DEFAULT_MEAN1[1])

        slider_ax_mu1x = plt.axes(
            [slider_base_left, 0.15, slider_width, 0.03], facecolor=axcolor)
        self.slider_mu1x = Slider(
            slider_ax_mu1x, '$\mu_{1x}$', -5.0, 15.0, valinit=config.DEFAULT_MEAN1[0])

        slider_ax_n2 = plt.axes(
            [slider_base_left, 0.19, slider_width, 0.03], facecolor=axcolor)
        self.slider_n2 = Slider(slider_ax_n2, '$N_2$',
                                0, 200, valinit=config.DEFAULT_N2, valstep=1)

        slider_ax_n1 = plt.axes(
            [slider_base_left, 0.23, slider_width, 0.03], facecolor=axcolor)
        self.slider_n1 = Slider(slider_ax_n1, '$N_1$',
                                0, 200, valinit=config.DEFAULT_N1, valstep=1)

        # 将所有滑块的 on_changed 连接到新的空操作处理函数
        self.slider_n1.on_changed(self._on_slider_value_changed_noop)
        self.slider_n2.on_changed(self._on_slider_value_changed_noop)
        self.slider_mu1x.on_changed(self._on_slider_value_changed_noop)
        self.slider_mu1y.on_changed(self._on_slider_value_changed_noop)
        self.slider_mu2x.on_changed(self._on_slider_value_changed_noop)
        self.slider_mu2y.on_changed(self._on_slider_value_changed_noop)

        self.all_sliders = [
            self.slider_n1, self.slider_n2,
            self.slider_mu1x, self.slider_mu1y,
            self.slider_mu2x, self.slider_mu2y
        ]

    def _on_slider_value_changed_noop(self, value):
        """滑块值连续变化时的回调，不执行任何操作以延迟主更新。"""
        pass

    def _on_figure_button_press(self, event):
        """处理图形上的鼠标按下事件，以检测滑块交互的开始。"""
        if event.inaxes:
            for slider in self.all_sliders:
                if event.inaxes == slider.ax:
                    self._active_slider_axis = slider.ax
                    break

    def _on_figure_button_release(self, event):
        """处理图形上的鼠标释放事件，如果之前在滑块上按下了鼠标，则触发更新。"""
        if self._active_slider_axis is not None:
            # 检查释放是否在同一个滑块轴上，或者只是一个通用释放
            # 为了简单起见，只要 _active_slider_axis 被设置了，就在释放时更新
            self.update(None)  # 使用 None 作为参数，因为 update 会从滑块直接读取值
            self._active_slider_axis = None

    def _format_matrix(self, matrix):
        if matrix is None or not isinstance(matrix, np.ndarray) or matrix.shape != (2, 2):
            return "  [[ N/A, N/A ],\n   [ N/A, N/A ]]"
        return (f"  [[{matrix[0,0]:7.2f}, {matrix[0,1]:7.2f}],\n   [{matrix[1,0]:7.2f}, {matrix[1,1]:7.2f}]]")

    def update(self, val):
        n1 = int(self.slider_n1.val)
        n2 = int(self.slider_n2.val)
        mu1 = np.array([self.slider_mu1x.val, self.slider_mu1y.val])
        mu2 = np.array([self.slider_mu2x.val, self.slider_mu2y.val])

        cov1 = config.DEFAULT_COV1
        cov2 = config.DEFAULT_COV2

        X1, X2 = data_processing.generate_data(n1, n2, mu1, mu2, cov1, cov2)

        if not (X1.ndim == 2 and X1.shape[0] > 0) and not (X2.ndim == 2 and X2.shape[0] > 0):
            self.scatter1.set_data([], [])
            self.scatter2.set_data([], [])
            self.mean1_plot.set_data([], [])
            self.mean2_plot.set_data([], [])
            self.lda_line.set_data([], [])
            self.projected1_scatter.set_data([], [])
            self.projected2_scatter.set_data([], [])
            self.ax_1d.set_xlim(-1, 1)  # Reset xlim for 1D plot when no data
            # 更新矩阵文本为空或默认值
            self.sw_text_matrix.set_text(self._format_matrix(np.zeros((2, 2))))
            self.sb_text_matrix.set_text(self._format_matrix(np.zeros((2, 2))))
            self.fig.canvas.draw_idle()
            return

        w, mu1_hat, mu2_hat, Sw, Sb = data_processing.compute_lda(X1, X2)

        # 更新二维散点图和均值
        if X1.ndim == 2 and X1.shape[0] > 0:
            self.scatter1.set_data(X1[:, 0], X1[:, 1])
            self.mean1_plot.set_data([mu1_hat[0]], [mu1_hat[1]])
        else:
            self.scatter1.set_data([], [])
            self.mean1_plot.set_data([], [])

        if X2.ndim == 2 and X2.shape[0] > 0:
            self.scatter2.set_data(X2[:, 0], X2[:, 1])
            self.mean2_plot.set_data([mu2_hat[0]], [mu2_hat[1]])
        else:
            self.scatter2.set_data([], [])
            self.mean2_plot.set_data([], [])

        # 绘制LDA线
        all_X_list = []
        if X1.ndim == 2 and X1.shape[0] > 0:
            all_X_list.append(X1)
        if X2.ndim == 2 and X2.shape[0] > 0:
            all_X_list.append(X2)

        if not all_X_list:  # 理论上被上面的早期return覆盖，但保留以防万一
            self.lda_line.set_data([], [])
        else:
            all_X_concat = np.vstack(all_X_list)
            overall_mean = np.mean(all_X_concat, axis=0)
            line_len = 15
            # 确保w是有效向量以进行乘法
            if w is not None and w.ndim == 1 and w.shape[0] == 2 and np.linalg.norm(w) > 1e-9:
                p1 = overall_mean - w * line_len / 2
                p2 = overall_mean + w * line_len / 2
                self.lda_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            else:  # 如果w无效（例如，只有一个类，或者lda计算结果为零向量）
                self.lda_line.set_data([], [])

        # 投影数据到1D
        y_offset = 0.1
        all_projected_vals = []

        if X1.ndim == 2 and X1.shape[0] > 0 and w is not None and w.ndim == 1 and w.shape[0] == X1.shape[1]:
            y1_projected = X1 @ w
            self.projected1_scatter.set_data(
                y1_projected, np.full_like(y1_projected, -y_offset))
            all_projected_vals.extend(y1_projected)
        else:
            self.projected1_scatter.set_data([], [])

        if X2.ndim == 2 and X2.shape[0] > 0 and w is not None and w.ndim == 1 and w.shape[0] == X2.shape[1]:
            y2_projected = X2 @ w
            self.projected2_scatter.set_data(
                y2_projected, np.full_like(y2_projected, y_offset))
            all_projected_vals.extend(y2_projected)
        else:
            self.projected2_scatter.set_data([], [])

        if all_projected_vals:
            min_proj, max_proj = np.min(
                all_projected_vals), np.max(all_projected_vals)
            range_proj = max_proj - min_proj
            if range_proj < 1e-6:
                range_proj = 1.0
            self.ax_1d.set_xlim(min_proj - 0.1 * range_proj,
                                max_proj + 0.1 * range_proj)
        else:
            self.ax_1d.set_xlim(-1, 1)

        # 更新 S_w 和 S_b 文本
        self.sw_text_matrix.set_text(self._format_matrix(Sw))
        self.sb_text_matrix.set_text(self._format_matrix(Sb))

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()
