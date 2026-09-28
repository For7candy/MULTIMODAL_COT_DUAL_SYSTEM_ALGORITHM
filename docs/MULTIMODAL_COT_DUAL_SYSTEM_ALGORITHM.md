# 多模态三维 CoT 快慢双系统导航算法设计

## 1 范围与目标

本阶段只设计算法及其软件接口，不包含仿真验证、ROS 2 集成和实机实验。算法暂命名为
`M3D-CoT-DS`（Multimodal 3D Chain-of-Thought Dual-System Navigation）。

系统只完成三项核心能力：

1. 模型接收 RGB、点云、位姿和语言，完成特征提取、空间对齐与融合；
2. 模型先生成结构化三维场景描述，再结合任务指令进行 CoT 推理，输出结构化导航信号；
3. 模型外部使用快慢双系统：慢系统负责语义规划，快系统负责几何执行与即时避障。

算法不让大模型直接输出电机或速度控制量。模型输出的是带有效期和置信度的语义子目标，
快系统将其转换为局部路径和控制动作。

## 2 总体架构

```text
 RGB I_t ──> RGB Encoder ──> Image Tokens ───────┐
                                                  │
 Point Cloud P_t -> Point Encoder -> Point Tokens ├─> 3D Alignment
                                                  │   + Token Fusion
 Pose T_t ──> Pose Encoder ──> Pose Tokens ───────┘
                                                         │
                                                         v
                                              Multimodal LLM Backbone
                                                         │
                                              Scene Decoder / Constrained JSON
                                                         │
                                                         v
                                           Structured 3D Scene Description S_t
                                                         │
 Task Instruction L ─> Text Encoder / Same Backbone ─────┤
                                                         v
                                            Structured CoT Planner
                                                         │
                                                         v
                                         Navigation Signal N_t (slow)
                                                         │
                          Point Cloud + Pose ──────────────┤
                                                         v
                                     Fast Geometric Planner / Controller
                                                         │
                                                         v
                                         local path or action a_t
```

模型由“场景理解阶段”和“任务推理阶段”组成，两阶段共享同一个大模型底座及多模态适配器，
但使用不同输出约束：第一阶段输出三维场景结构，第二阶段输出导航计划结构。

## 3 输入与输出定义

### 3.1 输入

在时间 `t`，输入定义为：

- RGB：`I_t ∈ R^(H×W×3)`，可扩展为最近 `K` 帧；
- 点云：`P_t = {(x_i,y_i,z_i,c_i)}_(i=1..N)`，`c_i` 可包含颜色、强度和时间；
- 位姿：`T_t ∈ SE(3)`，并携带坐标系、时间戳和可选协方差；
- 指令：自然语言任务 `L`；
- 可选记忆：此前场景描述、已完成子任务和导航历史 `M_(t-1)`。

所有传感器输入必须具有统一或可转换的时间戳和坐标系。位姿不是简单附加数字，而是用于把
图像、点云和历史观测投影到统一世界坐标系。

输入字段含义如下：

| 字段 | 类型/形状 | 含义 |
|---|---|---|
| `I_t` | `H×W×3` 或 `K×H×W×3` | 时刻 `t` 的单帧 RGB 图像或最近 `K` 帧图像序列。`H`、`W` 分别表示图像高度和宽度，3 表示 RGB 三个颜色通道。 |
| `P_t` | `N×(3+C)` | 时刻 `t` 的点云，共 `N` 个点；前三维是传感器坐标系下的 `x,y,z`，其余 `C` 维是可选附加属性。 |
| `c_i` | 标量或向量 | 第 `i` 个点的附加属性，可包含 RGB 颜色、激光反射强度、语义标签、相对采集时间或点级置信度。 |
| `T_t` | `4×4` 齐次矩阵 | 机器人或主传感器在参考坐标系中的六自由度位姿，用于把各模态转换到统一三维坐标系。 |
| `frame_id` | 字符串 | `T_t` 所依赖的参考坐标系名称，例如 `world`、`map` 或 `odom`。 |
| `timestamp_ms` | 整数 | 输入对应的采集时间，单位为毫秒，用于跨模态同步和输入新鲜度判断。 |
| `pose_covariance` | `6×6` 可选矩阵 | 位姿平移和旋转的不确定性；数值较大时降低空间融合及目标坐标的置信度。 |
| `camera_intrinsics` | `3×3` 矩阵 | 相机焦距和主点参数，用于将 RGB 像素射线反投影到三维空间。 |
| `sensor_extrinsics` | `4×4` 矩阵集合 | 相机、点云传感器与机器人坐标系之间的刚体变换，用于对齐 RGB、点云和位姿。 |
| `L` | UTF-8 字符串 | 用户给出的自然语言导航任务，包含动作、目标实体、顺序和可选约束。 |
| `M_(t-1)` | 结构化对象，可选 | 前一时刻的三维场景记忆、已经完成的子任务、此前失败原因及仍有效的实体轨迹。 |
| `K` | 正整数 | 模型一次使用的历史帧数量；`K=1` 表示仅处理当前帧。 |

`camera_intrinsics` 和 `sensor_extrinsics` 是 RGB 与点云完成确定性几何对齐的必要输入，虽然它们
不是独立语义模态，但必须与传感器数据一同传入预处理与融合模块。

### 3.2 中间输出：结构化三维描述

第一阶段输出 `SceneDescription3D`：

```json
{
  "scene_id": "scene-001",
  "timestamp_ms": 0,
  "frame_id": "world",
  "observer_pose": {
    "position": [0.0, 0.0, 0.0],
    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]
  },
  "entities": [
    {
      "id": "cabinet-07",
      "category": "switch_cabinet",
      "center_xyz": [4.2, 1.1, 0.9],
      "size_xyz": [0.8, 0.6, 1.8],
      "yaw": 1.57,
      "attributes": ["closed"],
      "confidence": 0.94,
      "source_observations": ["rgb:0", "point:0"]
    }
  ],
  "relations": [
    {
      "subject_id": "cabinet-07",
      "predicate": "left_of",
      "object_id": "door-02",
      "confidence": 0.88
    }
  ],
  "free_space": [],
  "obstacles": [],
  "candidate_approaches": [],
  "uncertainties": []
}
```

它是模型对当前场景的机器可读表达，不是自然语言描述。实体必须绑定三维位置、尺度、方向、
置信度和证据来源，避免第二阶段仅依据语言幻觉生成目标。

顶层及位姿字段含义如下：

| 字段 | 含义 |
|---|---|
| `scene_id` | 当前三维场景快照的唯一标识。场景内容发生语义级变化时生成新版本，供导航信号绑定。 |
| `timestamp_ms` | 该场景快照生成时的时间戳，用于判断描述是否过期。 |
| `frame_id` | 所有三维坐标采用的公共参考坐标系。实体、区域和导航目标必须使用或可转换到该坐标系。 |
| `observer_pose` | 生成该场景描述时机器人的观察位姿。 |
| `observer_pose.position` | 观察者在 `frame_id` 中的三维位置 `[x,y,z]`，单位为米。 |
| `observer_pose.orientation_xyzw` | 观察者朝向四元数 `[qx,qy,qz,qw]`，用于恢复完整三维旋转。 |
| `entities` | 场景中已完成跨模态融合和实例去重的实体数组。 |
| `relations` | 实体之间的结构化空间或语义关系数组。 |
| `free_space` | 已观测可通行空间区域数组；推荐每项包含 `region_id`、三维多边形/体素引用、净空高度和置信度。 |
| `obstacles` | 静态或动态障碍数组；推荐每项包含 `obstacle_id`、边界、速度、类别、时间戳和置信度。 |
| `candidate_approaches` | 针对实体生成的候选接近位姿数组；每项包含候选 ID、目标实体 ID、位置、朝向、净空和置信度。 |
| `uncertainties` | 无法可靠确定的信息数组，例如遮挡、模态冲突、位姿漂移或实体类别歧义；不得用默认值隐藏不确定性。 |

`entities` 中每个字段的含义如下：

| 字段 | 含义 |
|---|---|
| `id` | 场景内唯一且跨帧尽量稳定的实例 ID，是语言 grounding 和导航目标引用的主键。 |
| `category` | 规范化实体类别，例如 `switch_cabinet`、`door` 或 `corridor`。 |
| `center_xyz` | 实体三维包围体中心在 `frame_id` 中的 `[x,y,z]`，单位为米。 |
| `size_xyz` | 实体沿局部三轴的长度、宽度和高度 `[sx,sy,sz]`，单位为米。 |
| `yaw` | 实体绕公共坐标系竖直轴的朝向角，单位为弧度；完整三维姿态可在扩展字段中用四元数表示。 |
| `attributes` | 与任务相关的状态或属性列表，例如 `closed`、`red`、`number_07`。 |
| `confidence` | 模型对实体类别、实例对应和几何估计的综合置信度，范围 `[0,1]`。 |
| `source_observations` | 支撑该实体结论的原始观测引用，如 RGB 帧、点云帧或视角 ID，用于追溯和重新核验。 |

`relations` 中每个字段的含义如下：

| 字段 | 含义 |
|---|---|
| `subject_id` | 关系主体实体 ID，必须引用 `entities` 中已有实例。 |
| `predicate` | 规范化关系类型，如 `left_of`、`inside`、`connected_to`、`in_front_of`。 |
| `object_id` | 关系客体实体 ID，也必须引用现有实例。 |
| `confidence` | 当前关系判断的置信度，范围 `[0,1]`。 |

`candidate_approaches` 的推荐单项字段为：`candidate_id` 表示候选位姿唯一 ID；
`target_entity_id` 表示准备接近的实体；`position` 和 `yaw` 表示世界坐标目标位姿；
`clearance_m` 表示候选点到最近障碍的估计净空；`visibility_score` 表示从该点观察目标的质量；
`confidence` 表示该候选整体可信度。第二阶段 CoT 应优先从这些候选中选择，而不是重新生成
一个未经过三维检查的坐标。

### 3.3 最终输出：结构化导航信号

第二阶段输出 `NavigationSignal`：

```json
{
  "mission_id": "inspection-001",
  "scene_id": "scene-001",
  "subtask_id": "step-02",
  "intent": "approach",
  "target_entity_id": "cabinet-07",
  "candidate_id": "approach-a",
  "goal": {
    "frame_id": "world",
    "position": [3.3, 1.1, 0.0],
    "yaw": 1.57,
    "tolerance_m": 0.20
  },
  "constraints": {
    "minimum_clearance_m": 0.50,
    "maximum_speed_mps": 0.30,
    "forbidden_region_ids": []
  },
  "confidence": 0.87,
  "generated_at_ms": 0,
  "valid_for_ms": 5000,
  "replan_triggers": ["target_missing", "path_blocked", "scene_changed"],
  "status": "ready"
}
```

输出只表达当前应执行的一个子目标及约束。完整任务通过多个子目标迭代完成，而不是一次生成
不可修正的整条动作序列。

导航信号字段含义如下：

| 字段 | 含义 |
|---|---|
| `mission_id` | 完整任务的唯一 ID，用于关联指令、场景快照、导航信号和执行反馈。 |
| `scene_id` | 生成该导航信号所使用的 `SceneDescription3D` 版本；场景版本不一致时信号需要重新校验。 |
| `subtask_id` | 当前语义子任务的唯一 ID；同一任务可以依次产生多个子任务。 |
| `intent` | 当前导航意图的枚举值，例如 `approach`、`observe`、`pass_through`、`stop`。 |
| `target_entity_id` | 当前目标在 `SceneDescription3D.entities` 中的实例 ID，不允许使用无法绑定的自由文本名称。 |
| `candidate_id` | 从 `candidate_approaches` 中选中的接近位姿 ID，用于证明目标坐标来自已校验候选。 |
| `goal` | 快系统需要执行的三维子目标。 |
| `goal.frame_id` | 目标位姿所在坐标系，必须与当前位姿坐标系相同或存在确定性变换。 |
| `goal.position` | 目标位置 `[x,y,z]`，单位为米。移动平台不控制高度时，`z` 仍用于几何一致性检查。 |
| `goal.yaw` | 到达目标时的期望航向角，单位为弧度。 |
| `goal.tolerance_m` | 判断子目标到达时允许的位置误差半径，单位为米。 |
| `constraints` | 快系统在规划和控制过程中必须遵守的约束集合。 |
| `constraints.minimum_clearance_m` | 路径和目标与已知障碍之间要求保持的最小净空，单位为米。 |
| `constraints.maximum_speed_mps` | 执行当前子目标时允许的最高线速度，单位为米每秒。 |
| `constraints.forbidden_region_ids` | 本次导航明确禁止进入的三维区域 ID 列表。 |
| `confidence` | 慢系统对目标 grounding、候选选择和约束一致性的综合置信度，范围 `[0,1]`。 |
| `generated_at_ms` | 导航信号生成完成的时间戳，是计算有效期的起点。 |
| `valid_for_ms` | 信号自生成时间起允许使用的最长时间，单位为毫秒；超时后快系统必须停止使用。 |
| `replan_triggers` | 触发慢系统重新生成导航信号的条件枚举列表。 |
| `status` | 当前信号状态；只有 `ready` 可以交给快系统，其他状态表示需要观察、消歧、重规划或拒绝。 |

常用 `replan_triggers` 含义：`target_missing` 表示目标实体已无法在新场景中匹配；
`path_blocked` 表示快系统找不到安全局部路径；`scene_changed` 表示结构化三维描述发生影响当前
计划的变化。

## 4 多模态特征提取

### 4.1 RGB 分支

RGB 编码器使用视觉 Transformer，将图像转换为 patch tokens：

`F_img = E_img(I_t) ∈ R^(N_img×d_img)`。

建议保留两类信息：

- 高层语义特征，用于物体类别、属性和语言对齐；
- 中间层稠密特征，用于与点云进行像素级或体素级对应。

使用轻量投影器将视觉维度映射到大模型隐空间：

`Z_img = Projector_img(F_img) ∈ R^(M_img×d_llm)`。

其中 `Projector_img` 可使用 Q-Former、Perceiver Resampler 或两层 MLP 加 token pooling。

### 4.2 点云分支

点云先做范围裁剪、离群点过滤和体素化，再使用 Point Transformer 或稀疏体素 Transformer：

`F_pc = E_pc(Voxelize(P_t)) ∈ R^(N_pc×d_pc)`。

每个点云 token 保留其三维中心坐标和覆盖范围。经过点云投影器得到：

`Z_pc = Projector_pc(F_pc) ∈ R^(M_pc×d_llm)`。

点云分支负责提供深度、尺度、遮挡和可通行性信息；图像分支负责提供纹理、类别、文字标识
和细粒度语义。两者不能简单拼接后交给语言模型自行学习空间对应。

### 4.3 位姿分支

位姿包含平移、旋转、时间和不确定性。先将旋转表示转换为连续六维旋转编码，再结合平移的
Fourier positional encoding：

`Z_pose = MLP([PE(x,y,z), Rot6D(R), time, covariance])`。

位姿 token 同时承担三个作用：

1. 标记当前观察者在世界坐标中的位置；
2. 将当前观测和历史观测对齐；
3. 为导航目标提供统一参考坐标系。

### 4.4 语言分支

任务指令由大模型原生 tokenizer 和 embedding 层编码。第一阶段可以不输入任务，生成任务无关
的场景描述；第二阶段输入场景描述和任务指令，进行目标相关推理。

## 5 三维对齐与特征融合

### 5.1 几何对齐

给定相机内参 `K`、相机到机器人变换 `T_robot_camera` 和机器人位姿 `T_world_robot`，将带深度
的图像特征反投影到世界坐标；点云也通过外参和位姿变换到相同坐标系：

`p_world = T_world_robot · T_robot_sensor · p_sensor`。

随后在统一三维体素或局部邻域中建立图像 token 与点云 token 的对应关系。对于没有可靠深度
的像素，只允许通过可学习注意力建立软对应，并在输出中降低几何置信度。

### 5.2 分层融合

融合分三层完成：

1. **局部几何融合**：在相同三维邻域内对图像与点云特征进行双向 cross-attention；
2. **全局场景融合**：使用少量可学习 scene queries 汇聚实体、区域和关系 tokens；
3. **大模型语义融合**：将 scene tokens、pose token 和模态标记送入大模型底座。

局部融合可以写为：

`H_pc = Z_pc + CrossAttn(Q=Z_pc, K=Z_img, V=Z_img, mask=A_3d)`

`H_img = Z_img + CrossAttn(Q=Z_img, K=Z_pc, V=Z_pc, mask=A_3d)`

其中 `A_3d` 是根据三维邻近关系构造的稀疏注意力掩码。之后通过门控机制融合：

`g = sigmoid(W_g [H_img; H_pc; Z_pose])`

`H_fused = g ⊙ H_img + (1-g) ⊙ H_pc`。

门控权重允许模型在光照差时增加点云权重，在点云稀疏时增加图像权重。缺失某一模态时使用
显式 `MISSING_MODALITY` token，而不是补零后假装传感器正常。

### 5.3 时序与记忆

最近 `K` 帧的融合 tokens 通过位姿变换写入局部三维记忆。相同空间位置的特征按照置信度和
时间衰减更新：

`m_t = α_t f_t + (1-α_t) m_(t-1)`。

动态实体不直接写入静态记忆，而是维护独立轨迹。场景变化超过阈值时产生 `scene_changed`
事件，触发慢系统重新生成场景描述或导航信号。

## 6 第一阶段：结构化三维场景生成

融合后的 tokens 输入多模态大模型底座。模型通过带 Schema 约束的解码器生成
`SceneDescription3D`。该阶段执行以下子任务：

1. 识别实体及其三维边界；
2. 合并跨模态、跨帧的同一实体；
3. 识别实体属性和空间关系；
4. 提取障碍、自由空间和候选接近区域；
5. 标注不可确定、遮挡或模态冲突的内容。

解码过程必须使用 JSON Schema、有限状态解码或 grammar-constrained decoding，保证字段和类型
合法。生成后的几何字段还需经过确定性校验器检查：

- 引用的实体 ID 必须存在；
- 三维坐标必须处于当前地图范围；
- 尺寸和置信度必须在允许区间；
- 关系图不能引用自身或产生明显矛盾；
- 候选接近点不能落在已知障碍内部。

校验器只修正格式和拒绝错误结果，不负责凭空补充模型未观测到的信息。

## 7 第二阶段：结构化 CoT 导航推理

### 7.1 输入组织

将以下内容按固定模板输入同一大模型底座：

```text
<SCENE_3D> SceneDescription3D </SCENE_3D>
<TASK> natural-language instruction </TASK>
<HISTORY> completed subtasks and previous failures </HISTORY>
<OUTPUT_SCHEMA> NavigationReasoning + NavigationSignal </OUTPUT_SCHEMA>
```

第二阶段不再直接读取原始高分辨率图像和全部点云，而是主要使用结构化三维描述。必要时可以
通过实体 ID 对第一阶段 tokens 做检索，以核对目标外观或局部几何。

### 7.2 CoT 推理步骤

CoT 被约束为六个有类型的步骤：

1. `task_parse`：解析动作、目标、顺序、终止条件和约束；
2. `grounding`：将语言实体绑定到场景中的实体 ID；
3. `state_check`：判断当前位姿、已完成步骤和目标状态；
4. `candidate_generation`：生成多个三维候选子目标；
5. `candidate_evaluation`：按可达性、距离、可见性和约束逐项比较；
6. `decision`：选择一个子目标或返回需要补充观察/无法执行。

建议将训练时的完整推理监督与部署时的对外输出分开：模型内部保留推理 tokens，对外只输出
可审计的步骤摘要和最终结构化信号，避免自由文本被下游误当作控制命令。

CoT 的机器可读摘要形式为：

```json
{
  "task_parse": {"action": "approach", "target": "cabinet-07"},
  "grounding": {"entity_id": "cabinet-07", "confidence": 0.95},
  "candidate_ids": ["approach-a", "approach-b"],
  "evaluations": [
    {"candidate_id": "approach-a", "reachable": true, "score": 0.87},
    {"candidate_id": "approach-b", "reachable": false, "score": 0.31}
  ],
  "decision": {"selected": "approach-a", "reason_codes": ["reachable", "shorter"]}
}
```

### 7.3 拒绝与补充观察

模型不能可靠生成目标时，`NavigationSignal.status` 必须是下列状态之一：

- `need_observation`：目标可能存在，但当前观测不足，并输出建议观察方向；
- `ambiguous_target`：多个实体同样符合指令，需要消歧；
- `unreachable`：存在目标但没有合法候选；
- `invalid_instruction`：任务自相矛盾或缺少必要目标；
- `ready`：可以向快系统提交子目标。

不允许用随机目标或随机动作作为解析失败后的降级策略。

## 8 训练目标

模型采用分阶段训练，避免一开始联合优化导致三维几何能力被语言损失淹没。

### 8.1 阶段一：跨模态三维对齐

- 图像区域与点云簇对比损失 `L_align`；
- 三维中心、尺寸和朝向回归损失 `L_box`；
- 实体分类和属性损失 `L_entity`；
- 空间关系分类损失 `L_relation`。

### 8.2 阶段二：场景结构生成

- Schema token 生成损失 `L_scene_lm`；
- 实体 ID 一致性损失 `L_identity`；
- 几何字段一致性损失 `L_geometry`；
- 模态缺失与冲突条件下的置信度校准损失 `L_calibration`。

### 8.3 阶段三：CoT 与导航信号生成

- 任务解析与 grounding 损失 `L_ground`；
- CoT 步骤监督损失 `L_cot`；
- 候选排序损失 `L_rank`；
- 最终结构化导航信号损失 `L_nav`；
- 非法输出和随机降级惩罚 `L_invalid`。

总损失可表示为：

`L = λ1 L_align + λ2 L_box + λ3 L_scene_lm + λ4 L_cot + λ5 L_rank + λ6 L_nav + λ7 L_calibration`。

训练初期冻结大模型主体，先训练 RGB/点云投影器和融合层；随后逐步解冻高层或采用 LoRA，
最后联合微调结构生成和导航推理模块。

## 9 模型外部快慢双系统

### 9.1 慢系统

慢系统包含多模态模型的两个阶段以及输出校验器。它不按固定高频循环运行，只在以下事件触发：

- 收到新任务；
- 当前子目标完成；
- 场景结构发生语义级变化；
- 快系统报告路径受阻或无进展；
- 当前导航信号超时；
- 模型要求补充观察。

慢系统输出 `NavigationSignal`，典型频率为事件驱动或 `0.2–1 Hz`，允许数百毫秒到数秒的
推理延迟，但任何尚未校验或已经过期的输出都不能进入快系统。

### 9.2 快系统

快系统不使用大模型。它直接读取最新点云、位姿和已批准的子目标，维护局部占据表示并运行：

1. 局部障碍更新；
2. 到子目标的几何路径搜索；
3. 动态窗口、轨迹优化或 MPC；
4. 碰撞检查和动作限幅；
5. 到达判断与执行状态反馈。

快系统典型更新频率为 `10–50 Hz`。其输出可以是离散导航动作，也可以是线速度和角速度：

`a_t = FastPlanner(P_t, T_t, goal, constraints)`。

即使慢系统正在推理，快系统仍执行最后一个有效的导航信号；一旦信号过期、点云失效或局部
碰撞风险超过阈值，快系统停止并请求重规划。

### 9.3 调度与仲裁

系统状态为：

- `IDLE`：无任务；
- `SLOW_REASONING`：生成场景描述或导航信号；
- `FAST_EXECUTING`：执行有效子目标；
- `REPLAN_REQUIRED`：场景、目标或路径发生变化；
- `STOPPED`：输入失效、信号过期或无安全动作；
- `COMPLETE`：任务终止条件满足。

控制优先级为：

`停止条件 > 快系统碰撞检查 > 导航信号约束 > 慢系统目标偏好`。

慢系统只能替换语义子目标和约束，不能越过快系统的碰撞检查。

## 10 完整推理算法

```text
Algorithm M3D-CoT-DS
Input: RGB I_t, point cloud P_t, pose T_t, instruction L
State: scene memory M, current signal N, mission history H

1. Synchronize and validate I_t, P_t, T_t
2. F_img  <- RGBEncoder(I_t)
3. F_pc   <- PointEncoder(P_t)
4. F_pose <- PoseEncoder(T_t)
5. F_3d   <- GeometricAlignAndFuse(F_img, F_pc, F_pose)
6. M      <- UpdateSpatialMemory(M, F_3d, T_t)

7. if NewTask or SemanticSceneChanged or ReplanRequested then
8.     S_t <- ConstrainedSceneDecoder(LLM(F_3d, M))
9.     if not ValidateScene(S_t) then
10.        N <- NavigationSignal(status="need_observation")
11.    else
12.        R_t, N <- StructuredCoTPlanner(LLM(S_t, L, H))
13.        if not ValidateNavigationSignal(N, S_t) then
14.            N <- NavigationSignal(status="invalid")
15.        end if
16.    end if
17. end if

18. if N.status == "ready" and N.not_expired then
19.     local_map <- UpdateLocalMap(P_t, T_t)
20.     a_t <- FastPlanner(local_map, T_t, N.goal, N.constraints)
21.     if CollisionRisk(a_t) or InputsStale then
22.         a_t <- STOP
23.         RequestReplan()
24.     end if
25. else
26.     a_t <- STOP
27. end if

28. UpdateHistory(H, execution_feedback)
29. return S_t, R_t, N, a_t
```

其中第 `2–17` 行属于慢系统，第 `18–29` 行属于快系统。实际运行时两部分应为异步任务，
通过带版本号的不可变 `NavigationSignal` 交换信息。

## 11 关键校验规则

### 11.1 场景描述校验

- Schema 完整且可解析；
- 坐标系和时间戳存在；
- 实体 ID 唯一；
- 几何范围合法；
- 所有关系引用有效实体；
- 模态冲突和遮挡被显式标记。

### 11.2 导航信号校验

- 目标实体必须存在于对应版本的三维描述中；
- 目标坐标系必须可转换到当前位姿坐标系；
- 目标点不位于已知障碍中；
- 置信度达到阈值；
- 有效期大于零且当前未过期；
- 输出状态为 `ready`；
- 约束字段数值合法。

任何一项失败均返回慢系统重新推理或补充观察，不能随机补齐目标。

## 12 建议代码模块

```text
power_vln/
├── encoders/
│   ├── rgb_encoder.py
│   ├── point_encoder.py
│   └── pose_encoder.py
├── fusion/
│   ├── geometric_alignment.py
│   ├── cross_modal_fusion.py
│   └── spatial_memory.py
├── model/
│   ├── multimodal_backbone.py
│   ├── scene_decoder.py
│   └── cot_planner.py
├── schemas/
│   ├── scene_description.py
│   └── navigation_signal.py
├── slow_system/
│   ├── semantic_reasoner.py
│   └── output_validator.py
├── fast_system/
│   ├── local_map.py
│   ├── geometric_planner.py
│   └── collision_checker.py
└── orchestrator/
    ├── scheduler.py
    └── state_machine.py
```

模型实现与快系统实现保持依赖隔离。模型侧只依赖张量和 Schema；快系统只读取批准后的目标、
约束和本地几何信息，因此未来可以替换任一编码器、大模型底座或局部规划器。

## 13 最小实现顺序

本阶段无需进行仿真或实机实验，按以下顺序完成算法代码即可：

1. 定义 `SceneDescription3D` 与 `NavigationSignal` Schema；
2. 实现 RGB、点云、位姿三个编码器及维度统一投影；
3. 实现基于坐标的局部 cross-attention 融合；
4. 接入大模型底座，完成受约束的三维场景结构生成；
5. 实现“场景结构 + 任务指令”的结构化 CoT 提示与解码；
6. 实现输出校验器，拒绝无效实体引用和几何目标；
7. 实现模型外异步调度器和快系统接口；
8. 使用构造张量和固定样例做模块单元测试，不开展仿真性能评测。

## 14 本阶段完成判据

算法设计或代码原型满足以下条件，即视为达到当前阶段目标：

- 四类输入均有明确编码、对齐和融合路径；
- 模型能够输出符合 Schema 的结构化三维描述；
- 场景描述和指令能够进入第二阶段 CoT；
- CoT 输出结构化导航信号，而不是直接输出底盘动作；
- 慢系统与快系统通过版本化信号异步解耦；
- 快系统在信号无效、过期或存在碰撞风险时失败封闭；
- 所有数据结构和模块接口可通过不依赖仿真的单元测试验证。

## 15 Open-Nav 代码复用方案

### 15.1 复用原则

本算法不是重新实现整个 Open-Nav，而是保留其已经验证的候选路点生成、视觉特征、方向编码、
LLM 调用和导航编排能力，再替换其场景表达与决策接口。复用分为三类：

- **直接复用**：接口和语义与新算法一致，仅做封装；
- **改造复用**：保留权重、网络层或流程，但替换输入输出和错误处理；
- **当前不复用**：只服务于 Habitat 仿真、模仿学习或与新架构冲突的代码。

### 15.2 直接复用的代码

| Open-Nav 路径 | 复用内容 | 在新算法中的用途 |
|---|---|---|
| `waypoint_prediction/TRM_net.py` | `BinaryDistPredictor_TRM` 的 RGB/深度候选路点热力图网络及已有权重加载方式 | 作为快系统的候选方向和距离生成器，也可为慢系统提供 `candidate_approaches` 初始候选。 |
| `waypoint_prediction/transformer/waypoint_bert.py` | `WaypointBert`、视觉位置编码和注意力层 | 复用已有候选路点 Transformer 表达，不需要重新实现候选方向建模。 |
| `waypoint_prediction/utils.py` | 路点热力图后处理和 NMS | 从密集预测中提取离散候选路点，去除相邻重复候选。 |
| `vlnce_baselines/models/utils.py` | `angle_feature`、`dir_angle_feature`、`length2mask` 等函数 | 编码候选朝向、历史动作方向并生成变长候选掩码。 |
| `vlnce_baselines/models/encoders/resnet_encoders.py` | RGB 编码器及其预处理接口 | 作为第一版 `RGBEncoder`，输出图像特征后接新建的 `Projector_img`。 |
| `vlnce_baselines/config/default.py` | 配置合并、默认参数和命令行覆盖机制 | 增加点云编码器、融合器、大模型和快慢系统配置项。 |
| `run.py` | 配置读取、随机种子、设备选择和注册表启动方式 | 作为新算法离线入口的基础，新增 `algorithm` 子命令或独立配置。 |

上述“直接复用”是指保留核心实现；为了消除 Habitat 类型依赖，仍可在外层增加薄适配器。

### 15.3 需要改造后复用的代码

| Open-Nav 路径 | 保留内容 | 必须修改的内容 |
|---|---|---|
| `vlnce_baselines/models/Policy_ViewSelection.py` | `CMANet` 中 RGB/深度特征提取、候选构造、方向特征以及 `mode='waypoint'` 分支 | 将候选输出转换为带三维位置的 `candidate_approaches`；新增点云特征输入。原 `mode='navigation'` 不再作为最终慢系统决策器。 |
| `vlnce_baselines/common/navigator/api.py` | `llmClient` 的模型调用抽象、重试机制以及本地/远程模型切换思路 | 将自由文本 completion 改为 Schema 约束生成；用新的 RGB+点云+位姿多模态底座替换 `spatialClient` 当前的 RAM+SpatialBot RGB/深度描述流程。 |
| `vlnce_baselines/common/navigator/spatialNavigator.py` | 指令分解、环境观察、历史维护、完成度估计、多候选推理和决策复核的流程结构 | 重写为 `SceneDescription3D -> StructuredCoT -> NavigationSignal`；删除字符串切分解析、方向编号作为唯一输出以及异常时 `random.choice` 的随机降级。 |
| `vlnce_baselines/common/navigator/prompts.py` | 动作分解、地标提取、进度判断、候选比较和思想融合等任务定义 | 改写为结构化 CoT 模板；每个阶段绑定 JSON Schema 和枚举字段，不再依赖 `Thought:`、`Prediction:` 文本标记。 |
| `vlnce_baselines/common/base_il_trainer_llm.py` | `generate_input` 的多视角观测整理、`construct_image_dicts` 的候选方向映射、缓存、日志以及“观察—推理—执行—记录”循环 | 抽取为与 Habitat 无关的 `orchestrator`；输入增加原始点云、位姿、内外参和时间戳；输出由环境动作改为版本化 `NavigationSignal`。 |
| `vlnce_baselines/models/encoders/instruction_encoder.py` | 指令序列编码接口及批处理方式 | 仅用于兼容已有策略特征；新 CoT 主路径应使用大模型底座自身 tokenizer 和语言 embedding，不重复编码后再转回文本。 |
| `vlnce_baselines/models/policy.py` | 动作分布封装和策略接口形式 | 只在选择保留离散快系统策略时使用；慢系统不通过 `ILPolicy.act()` 直接产生动作。 |

### 15.4 当前阶段不复用的代码

| Open-Nav 路径 | 当前不复用的原因 |
|---|---|
| `habitat_extensions/` | 主要提供 Habitat 任务、传感器和评测扩展；当前阶段不进行仿真。 |
| `vlnce_baselines/common/env_utils.py` | 负责创建和管理 Habitat 环境，不属于纯算法核心。 |
| `vlnce_baselines/common/environments.py` | 定义仿真环境包装器，当前无运行需求。 |
| `vlnce_baselines/ss_trainer.py` | 面向原始监督/模仿学习训练流程，与新的两阶段结构化生成训练目标不同。 |
| `vlnce_baselines/common/aux_losses.py` | 原策略网络辅助损失不能直接覆盖三维对齐、场景结构生成和 CoT 损失；仅在保留原策略训练时使用。 |
| `habitat_extensions/measures.py` 等评测代码 | 当前不做仿真指标评估，后续需要验证时再接回。 |

“当前不复用”不表示删除这些文件。它们继续保留在上游目录中，确保原 Open-Nav 基线仍可运行，
但不应成为 `power_vln` 新算法包的强制运行依赖。

### 15.5 新旧模块映射

| 新模块 | 主要复用来源 | 新增实现 |
|---|---|---|
| `encoders/rgb_encoder.py` | `resnet_encoders.py` | 大模型维度投影器和多层特征导出 |
| `encoders/point_encoder.py` | 无 | Point Transformer 或稀疏体素 Transformer |
| `encoders/pose_encoder.py` | `models/utils.py` 的方向编码思想 | SE(3)、时间和协方差编码 |
| `fusion/geometric_alignment.py` | 无 | 相机内外参反投影和世界坐标对齐 |
| `fusion/cross_modal_fusion.py` | `Policy_ViewSelection.py` 的注意力结构可作参考 | 三维邻域掩码、双向 cross-attention 和门控融合 |
| `model/scene_decoder.py` | `api.py` 的模型调用封装 | `SceneDescription3D` 受约束解码与几何校验 |
| `model/cot_planner.py` | `spatialNavigator.py`、`prompts.py` 的任务流程 | 六阶段结构化 CoT 与 `NavigationSignal` 解码 |
| `fast_system/geometric_planner.py` | `waypoint_prediction/`、`Policy_ViewSelection.py` 的 waypoint 分支 | 点云局部地图、目标坐标转换和碰撞检查 |
| `orchestrator/scheduler.py` | `base_il_trainer_llm.py` 的主循环 | 异步快慢调度、版本检查、超时和事件触发 |

### 15.6 推荐迁移顺序

1. 先封装 `waypoint_prediction` 和 RGB 编码器，确保不修改上游文件也能调用；
2. 新建点云、位姿编码器以及三维对齐模块；
3. 将 `api.py` 的调用方式迁移为带 Schema 的多模态模型接口；
4. 用结构化场景生成器和 CoT 规划器替换 `spatialNavigator.py` 的自由文本决策；
5. 从 `base_il_trainer_llm.py` 抽取与环境无关的调度逻辑；
6. 最后把 waypoint 候选生成器接入快系统，并保留原 Open-Nav 入口用于回归对照。

所有新增代码放入 `power_vln/`，通过适配器引用 Open-Nav 模块；不直接改写上游文件。这样可以
持续对照原始基线，并在后续更新 Open-Nav 时降低合并成本。

## 16 计划执行步骤

### 16.1 执行范围与规则

本计划用于把上述算法设计实现为可导入、可配置、可做单元测试的代码原型。当前范围不包含
Habitat 仿真、ROS 2、机器人驱动和实机数据采集。执行过程中遵守以下规则：

1. 原 Open-Nav 目录保持不变，新增实现统一放在 `power_vln/`；
2. 先冻结数据契约，再实现网络，避免模型输出与快慢系统接口反复变化；
3. 每一步单独提交，只有通过本步验收条件后才进入依赖它的步骤；
4. 重模型和第三方权重通过适配器延迟加载，使 Schema、调度器和单元测试可在 CPU 上运行；
5. 所有模型输出先解析、再校验，校验失败不得进入下一模块；
6. 本阶段仅使用固定样例、构造张量和离线文件验证代码正确性，不报告导航性能指标。

### 16.2 总体顺序

| 步骤 | 工作包 | 前置依赖 | 主要产物 |
|---|---|---|---|
| 0 | 参数和接口决策冻结 | 无 | 配置决策记录 |
| 1 | Schema 与校验器 | 步骤 0 | 输入包、三维描述、CoT、导航信号协议 |
| 2 | Open-Nav 适配层 | 步骤 1 | RGB、waypoint、LLM 调用适配器 |
| 3 | 多模态输入管线 | 步骤 1 | 同步、坐标系和预处理模块 |
| 4 | RGB、点云、位姿编码器 | 步骤 2、3 | 统一维度的多模态 tokens |
| 5 | 三维对齐与跨模态融合 | 步骤 4 | 融合场景 tokens 与对应关系 |
| 6 | 空间记忆 | 步骤 5 | 跨帧实体和特征记忆 |
| 7 | 第一阶段场景结构生成 | 步骤 5、6 | `SceneDescription3D` |
| 8 | 第二阶段结构化 CoT | 步骤 7 | `NavigationReasoning`、`NavigationSignal` |
| 9 | 慢系统编排 | 步骤 7、8 | 事件驱动语义推理器 |
| 10 | 快系统算法 | 步骤 1、2 | 局部地图、候选路点和安全动作 |
| 11 | 快慢系统集成 | 步骤 9、10 | 异步调度器和完整状态机 |
| 12 | 训练接口与纯代码验收 | 步骤 4–11 | 损失接口、测试集和交付清单 |

### 16.3 分步执行说明

#### 步骤 0：冻结关键参数和接口决策

执行任务：

- 确定大模型底座的本地加载接口、token 维度 `d_llm` 和最大上下文长度；
- 确定 RGB 编码器第一版使用 Open-Nav ResNet，点云编码器使用 Point Transformer 或稀疏体素
  Transformer；
- 确定 `M_img`、`M_pc`、scene query 数量、历史窗口 `K` 和统一坐标系；
- 明确结构化输出采用 JSON Schema，并固定所有枚举值；
- 将参数写入 `configs/power_vln/algorithm.yaml`，禁止散落在源码中。

产物：参数决策记录和算法配置文件。

完成条件：所有下游模块都能从配置中获得维度、模型名称、阈值和路径，不再依赖未定义常量。

#### 步骤 1：实现数据 Schema 与校验器

执行任务：

- 建立 `SensorPacket`，包含 RGB、点云、位姿、时间戳、坐标系、内外参和任务指令；
- 建立 `SceneDescription3D`、`Entity3D`、`Relation3D`、`CandidateApproach`；
- 建立 `NavigationReasoning`、`NavigationSignal`、`NavigationConstraints`；
- 实现 JSON 序列化、反序列化、版本字段和枚举；
- 实现场景引用、数值范围、实体唯一性、目标存在性和有效期校验。

建议代码位置：

```text
power_vln/schemas/sensor_packet.py
power_vln/schemas/scene_description.py
power_vln/schemas/navigation_signal.py
power_vln/validators/scene_validator.py
power_vln/validators/navigation_validator.py
```

产物：与深度学习框架解耦的数据契约和失败封闭校验器。

完成条件：合法样例能完整 JSON 往返；缺字段、重复实体、错误引用、越界置信度和过期信号均被
确定性拒绝。

#### 步骤 2：建立 Open-Nav 适配层

执行任务：

- 封装 `resnet_encoders.py` 为统一 `RGBEncoderAdapter`；
- 封装 `BinaryDistPredictor_TRM` 和 NMS 为 `WaypointAdapter`；
- 封装 `llmClient` 为 `BackboneClient`，统一本地和远程底座接口；
- 为每个适配器定义轻量 Protocol，使单元测试可以使用 mock；
- Open-Nav 依赖和权重只在首次调用对应适配器时加载。

建议代码位置：

```text
power_vln/adapters/opennav_rgb.py
power_vln/adapters/opennav_waypoint.py
power_vln/adapters/backbone_client.py
```

产物：新算法调用 Open-Nav 能力的稳定边界。

完成条件：适配器可在 mock 模型下返回约定形状；关闭适配器时，Schema 和调度模块仍可独立导入。

#### 步骤 3：实现多模态输入管线

执行任务：

- 校验 RGB 和点云形状、数据类型、时间戳及坐标系；
- 根据时间阈值完成 RGB、点云和位姿匹配；
- 使用外参把点云转换到机器人或世界坐标系；
- 保存图像内参、像素索引和点云索引，供后续几何对应使用；
- 当复用 Open-Nav waypoint 网络时，从已标定点云生成对应视角的深度投影；
- 显式生成模态缺失和输入过期标志。

建议代码位置：

```text
power_vln/data/synchronizer.py
power_vln/data/calibration.py
power_vln/data/preprocessor.py
power_vln/data/pointcloud_projection.py
```

产物：规范化的 `AlignedInputBundle`。

完成条件：给定固定 RGB、点云、位姿和标定样例，重复运行得到完全相同的对齐结果；错误坐标系
或超时输入被拒绝。

#### 步骤 4：实现三类编码器

执行任务：

- RGB 分支复用 Open-Nav 编码器，导出稠密视觉特征并投影到 `d_llm`；
- 点云分支完成体素化/采样、三维位置编码和点云 Transformer 特征提取；
- 位姿分支实现平移 Fourier 编码、Rot6D、时间和协方差编码；
- 每个分支输出 token、位置、有效掩码和模态置信度；
- 实现模态缺失 token，禁止用全零输入冒充有效观测。

建议代码位置：

```text
power_vln/encoders/rgb_encoder.py
power_vln/encoders/point_encoder.py
power_vln/encoders/pose_encoder.py
power_vln/encoders/projectors.py
```

产物：`ImageTokenBatch`、`PointTokenBatch` 和 `PoseTokenBatch`。

完成条件：不同 batch 和点数输入均输出固定隐维度；掩码、梯度和设备迁移行为正确；缺失模态
不会产生 NaN。

#### 步骤 5：实现三维对齐与跨模态融合

执行任务：

- 使用内外参建立像素、点和世界坐标之间的对应；
- 构造局部三维邻域稀疏注意力掩码 `A_3d`；
- 实现图像到点云、点云到图像的双向 cross-attention；
- 实现位姿条件门控和 scene queries 汇聚；
- 输出融合 token、三维锚点、模态贡献权重和不确定性。

建议代码位置：

```text
power_vln/fusion/geometric_alignment.py
power_vln/fusion/cross_modal_fusion.py
power_vln/fusion/scene_queries.py
```

产物：`FusedSceneTokens`。

完成条件：三维邻域外的 token 不发生直接注意；关闭任一传感器分支时仍能输出有效结果；门控权重
在 `[0,1]` 内且可追溯到输入模态。

#### 步骤 6：实现空间记忆

执行任务：

- 依据世界坐标把当前 `FusedSceneTokens` 写入局部三维记忆；
- 使用空间距离、类别相似度和外观相似度合并跨帧实体；
- 为静态特征实现置信度加权和时间衰减；
- 为动态实体维护独立轨迹，不写入永久静态地图；
- 计算场景差异，生成 `scene_changed` 事件。

建议代码位置：

```text
power_vln/fusion/spatial_memory.py
power_vln/fusion/entity_tracker.py
power_vln/fusion/scene_change_detector.py
```

产物：带版本号的 `SpatialMemorySnapshot`。

完成条件：重复观测不会生成重复实体；过期观测按配置衰减；相同输入序列产生确定的场景版本。

#### 步骤 7：实现第一阶段结构化三维生成

执行任务：

- 将融合 tokens、空间记忆和位姿 token 组织为大模型底座输入；
- 为场景生成设置独立提示模板和 `SceneDescription3D` JSON Schema；
- 实现 grammar-constrained decoding 或等价的受约束解码；
- 解码实体、空间关系、自由空间、障碍、候选接近位姿和不确定性；
- 将模型输出交给 `SceneValidator`，失败时生成明确错误码而非自动补值。

建议代码位置：

```text
power_vln/model/multimodal_backbone.py
power_vln/model/scene_decoder.py
power_vln/prompts/scene_description.py
```

产物：经过校验并带 `scene_id` 的 `SceneDescription3D`。

完成条件：固定融合 tokens 能稳定得到可解析结构；所有实体和关系引用有效；非法几何输出被拒绝。

#### 步骤 8：实现第二阶段结构化 CoT

执行任务：

- 将 `SceneDescription3D`、任务指令和任务历史组装为第二阶段输入；
- 分别实现任务解析、实体 grounding、状态检查、候选生成、候选评价和最终决策；
- 输出结构化 `NavigationReasoning`，并由最终决策生成 `NavigationSignal`；
- 实现 `need_observation`、`ambiguous_target`、`unreachable`、`invalid_instruction` 和 `ready`；
- 检查目标实体、候选 ID、场景版本、置信度、有效期和重规划条件。

建议代码位置：

```text
power_vln/model/cot_planner.py
power_vln/model/candidate_ranker.py
power_vln/prompts/navigation_cot.py
```

产物：`NavigationReasoning` 和经过校验的 `NavigationSignal`。

完成条件：相同输入在确定性配置下得到相同结构；目标歧义时不擅自选择；所有 `ready` 信号都能
追溯到同版本场景中的实体与候选接近点。

#### 步骤 9：实现慢系统编排

执行任务：

- 串联“输入更新—特征融合—场景生成—CoT—信号校验”；
- 实现新任务、子目标完成、场景变化、路径受阻和信号过期事件；
- 对同一场景版本缓存结构化三维描述，避免每次 CoT 都重复生成；
- 慢系统运行时保留上一条仍有效信号，但不自动延长有效期；
- 统一记录输入版本、模型版本、提示模板版本和输出错误码。

建议代码位置：

```text
power_vln/slow_system/semantic_reasoner.py
power_vln/slow_system/event_handler.py
power_vln/slow_system/plan_cache.py
```

产物：事件驱动的 `SlowSystem` 接口。

完成条件：每种触发事件只产生一次预期重规划；缓存不会跨场景版本误用；异常不会生成可执行信号。

#### 步骤 10：实现快系统算法

执行任务：

- 用最新点云和位姿维护轻量局部占据表示；
- 将 `NavigationSignal.goal` 转换到当前局部坐标系；
- 优先使用点云几何生成可行局部候选；有标定深度投影时，可调用 Open-Nav
  `WaypointAdapter` 补充候选方向和距离；
- 对候选执行碰撞、净空、速度和禁区约束检查；
- 选择最优候选并输出 `FastAction`、局部路径或 `STOP`；
- 生成 `ACTIVE`、`REACHED`、`BLOCKED`、`INPUT_STALE` 和 `SAFETY_STOP` 反馈。

建议代码位置：

```text
power_vln/fast_system/local_map.py
power_vln/fast_system/geometric_planner.py
power_vln/fast_system/collision_checker.py
power_vln/fast_system/action_selector.py
```

产物：不依赖大模型的 `FastSystem` 接口。

完成条件：无有效信号、点云过期、目标落入障碍或所有候选不安全时只返回 `STOP`；快系统单次调用
不得触发大模型推理。

#### 步骤 11：集成快慢系统

执行任务：

- 使用不可变消息或线程安全队列交换 `NavigationSignal` 和执行反馈；
- 扩展现有 `DualSystemStateMachine`，覆盖 `IDLE`、`SLOW_REASONING`、`FAST_EXECUTING`、
  `REPLAN_REQUIRED`、`STOPPED` 和 `COMPLETE`；
- 对信号执行原子替换，快系统不得读取一半更新的数据；
- 实现场景版本、信号时间戳和任务 ID 的三重一致性检查；
- 实现慢系统超时、取消和重复事件合并；
- 提供一个离线 `step()` API，输入固定传感器包并返回中间结果及动作。

建议代码位置：

```text
power_vln/orchestrator/scheduler.py
power_vln/orchestrator/signal_store.py
power_vln/orchestrator/pipeline.py
power_vln/state_machine.py
```

产物：可离线调用的 `M3DCoTDualSystem` 完整算法对象。

完成条件：慢系统阻塞不影响快系统处理有效信号；过期信号不会执行；任何 STOP 条件都能抢占普通
导航动作；相同事件序列得到相同状态迁移。

#### 步骤 12：建立训练接口和纯代码验收

执行任务：

- 为三维对齐、实体几何、场景生成、CoT、候选排序和导航信号定义损失接口；
- 提供冻结底座、仅训练适配器，以及 LoRA 微调两种配置；
- 建立构造张量 fixture、固定 JSON golden samples 和 mock backbone；
- 编写 Schema、几何变换、融合形状、受约束解码、状态机和端到端离线管线测试；
- 提供仅检查依赖和配置的 `dry-run` 命令；
- 更新模型输入输出、配置、权重加载和错误码说明。

建议代码位置：

```text
power_vln/training/losses.py
power_vln/training/module.py
tests/power_vln/fixtures/
tests/power_vln/test_*.py
```

产物：可训练接口、测试套件、配置样例和使用说明。

完成条件：CPU 环境下 Schema、校验器、调度器和 mock 端到端测试全部通过；有 GPU 和权重时能够
完成一次真实模型前向并输出两个合法 Schema；不要求运行仿真或连接机器人。

### 16.4 阶段门与交付物

| 阶段门 | 包含步骤 | 必须交付 | 通过条件 |
|---|---|---|---|
| M1：接口可用 | 0–3 | 配置、完整 Schema、校验器、Open-Nav 适配层、输入管线 | 固定样例可从原始输入转换为合法 `AlignedInputBundle`，所有异常输入被拒绝。 |
| M2：多模态表征可用 | 4–6 | 三类编码器、三维对齐、融合器、空间记忆 | 构造张量可完成端到端前向，形状、掩码、坐标和版本一致。 |
| M3：模型两阶段输出可用 | 7–8 | 三维场景解码器、结构化 CoT、导航信号 | 同一任务可得到合法 `SceneDescription3D` 和 `NavigationSignal`，错误引用无法通过校验。 |
| M4：快慢系统闭环可用 | 9–11 | 慢系统、快系统、调度器、状态机 | 离线输入序列能够产生确定的计划、动作和反馈；过期与不安全状态只输出 STOP。 |
| M5：代码原型可交付 | 12 | 训练接口、测试、配置和文档 | CPU mock 测试全部通过，可选真实模型单次前向通过。 |

### 16.5 当前代码与计划的衔接

当前仓库已经具备以下基础，不需要重新实现：

- `power_vln/models.py`：已有最小 `SemanticPlan` 和候选目标协议；
- `power_vln/plan_gate.py`：已有置信度、规则、时效、坐标和可达性网关；
- `power_vln/state_machine.py`：已有快慢系统状态机骨架；
- `tests/power_vln/test_contracts.py`：已有合同、计划网关和状态迁移测试。

正式执行计划时，从步骤 0 开始冻结配置；步骤 1 在现有协议之上扩展三维场景、传感器输入、
结构化 CoT 和完整导航信号，不删除已经通过的兼容字段。步骤 11 再将现有计划网关和状态机接入
完整调度器。

### 16.6 建议提交边界

为便于回退和审查，建议形成以下独立提交：

1. `feat: add multimodal schemas and validators`
2. `feat: add Open-Nav adapters and input pipeline`
3. `feat: add RGB point and pose encoders`
4. `feat: add geometric alignment and spatial fusion`
5. `feat: add structured 3D scene decoder`
6. `feat: add structured CoT navigation planner`
7. `feat: add slow and fast system implementations`
8. `feat: integrate dual-system scheduler`
9. `test: add offline algorithm acceptance suite`

每个提交只进入一个主要能力，附带对应测试；不得把模型接口变化、调度变化和大规模格式化混在
同一提交中。
