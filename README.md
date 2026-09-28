# M3D-CoT-DS

M3D-CoT-DS（Multimodal 3D Chain-of-Thought Dual-System Navigation）是一个面向视觉语言导航的多模态三维理解、结构化推理与快慢双系统算法原型。

算法接收 RGB、点云、位姿和自然语言指令，生成结构化三维场景与语义导航信号；大模型不直接输出电机控制量，快速系统负责局部规划、碰撞检查和安全停止。

## 核心流程

```text
RGB + 点云 + 位姿 + 指令
          |
          v
特征编码 -> 三维对齐 -> 跨模态融合 -> 空间记忆
          |
          v
SceneDescription3D -> 结构化 CoT -> NavigationSignal
                                          |
                                          v
                         快系统局部规划与安全校验
```

主要实现包括：

- RGB、点云、位姿和语言输入的数据契约与校验；
- Point Transformer-Small、位姿编码器与 RGB 特征适配接口；
- 三维几何对齐、跨模态融合、场景查询和空间记忆；
- `SceneDescription3D` 生成与结构化导航规划；
- 事件驱动慢系统、无大模型依赖的快系统及异步调度；
- 多任务训练损失以及 Adapter-only、LoRA 参数策略。

完整算法定义、数据结构和校验规则见 [算法设计文档](docs/MULTIMODAL_COT_DUAL_SYSTEM_ALGORITHM.md)。

## 仓库结构

```text
power_vln/          核心算法与数据结构
configs/power_vln/  算法、模型和系统示例配置
scripts/power_vln/  mock Demo、环境检查与 Qwen 冒烟脚本
tests/power_vln/    单元测试、回归测试和固定 Schema 样例
docs/               核心算法设计文档
```


## 环境与安装

建议使用 Python 3.10 或更高版本。安装依赖：

```powershell
python -m pip install -r requirements.txt
```

默认依赖包含 PyTorch、Transformers 和 Qwen-VL 运行组件。模型权重需要单独准备。

## 快速运行

运行不依赖真实模型权重的确定性 Demo：

```powershell
python -m scripts.power_vln.demo_algorithm --backend mock
```

如需保存结果，可写入已忽略的运行产物目录：

```powershell
python -m scripts.power_vln.demo_algorithm `
  --backend mock `
  --output artifacts/power_vln_demo.json
```

运行测试：

```powershell
python -m unittest discover -s tests/power_vln -v
```

`tests/` 是源码的一部分：它覆盖 Schema、编码器、融合、场景生成、CoT 规划、快慢系统、调度器和训练接口，用于验证行为并防止后续修改造成回归。

## 真实模型检查

在 `configs/power_vln/algorithm.yaml` 中设置本地模型路径，或通过命令行传入：

```powershell
python -m scripts.power_vln.dry_run --check-model-files

python -m scripts.power_vln.demo_algorithm `
  --backend qwen `
  --model-path C:\path\to\Qwen2.5-VL-3B-Instruct

python -m scripts.power_vln.smoke_qwen_cot `
  --model-path C:\path\to\Qwen2.5-VL-3B-Instruct
```

真实 Qwen 模式会加载模型并依次生成结构化场景描述与导航信号。模型路径、检查点和 API 密钥不得提交到仓库。

## 训练策略与当前边界

默认训练策略为 `adapter_only`：冻结 Qwen 主体与 RGB 主干，训练点云编码器、投影器、融合层和场景查询。也可在配置中选择冻结底座或 LoRA；LoRA 模式需要额外安装 `peft`。

当前仓库提供算法模块、训练损失和离线验证链路，但不包含生产数据读取器、完整训练循环、数据集、模型权重、仿真验证或实机实验。默认 mock Demo 只验证数据流与失败封闭逻辑，不代表模型已具备真实场景中的识别、三维重建或导航能力。

遇到输入过期、坐标系不一致、Schema 错误、碰撞风险、目标占用或净空不足时，系统应拒绝发布导航信号或输出零速度安全动作。

## Open-Nav 适配

本实现源自 Open-Nav 扩展代码，原始上游项目见 [YanyuanQiao/Open-Nav](https://github.com/YanyuanQiao/Open-Nav)，相关论文见 [Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environment with Open-Source LLMs](https://arxiv.org/abs/2409.18794)。

`power_vln/adapters/opennav_rgb.py` 和 `power_vln/adapters/opennav_waypoint.py` 对 Open-Nav 的 RGB 编码器与航点预测器进行惰性适配。默认 mock Demo 和核心单元测试不依赖 Open-Nav；启用适配器时需将 Open-Nav 加入 `PYTHONPATH`，并自行准备其运行依赖与检查点。

## 许可证

本项目采用 [MIT License](LICENSE)。外部模型、数据集和上游组件分别受其自身许可证约束。
