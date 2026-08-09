# Stage 4 定稿图表

全部图片由仓库根目录的 `finalize_stage4.py` 从本地 V17 主副本生成，采用统一模型颜色、
字体层级、坐标格式和 220 DPI 输出。图背后的机器可读数据位于 `docs/tables/`。

| 文件 | 用途 | 主要数据源 |
|---|---|---|
| `four_model_rankic_by_window.png` | 四模型延迟口径 RankIC；神经网络误差条为五 seed 标准差 | V17 基线预测、六信号预测 |
| `four_model_decile_returns.png` | 四模型完整十档曲线 | V17 预测与延迟收益 |
| `gru_integrated_gradients_heatmap.png` | 48×10 GRU 平均绝对 IG | `gru_attribution_all_windows.csv` |
| `four_model_cost_curves.png` | 30分钟与日频方案的 0—20bp 成本曲线 | `cost_summary_full_period.csv` |
| `gru_increment_market_state.png` | 八窗口 GRU 增量与市场状态描述性对照 | `window_market_context.csv` |

复现命令：

```bash
MPLCONFIGDIR=/tmp/mpl-stage4-final python finalize_stage4.py
```

