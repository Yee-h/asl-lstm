# ASL-LSTM 项目完整修复执行计划（强约束版）

> 本计划用于指导后续 AI Coding Agent 执行项目修复。
> 本计划为**唯一执行规范**，所有步骤必须按顺序执行，不允许跳步、不允许并行、不允许口头化完成。

---

## 1. 执行总原则（必须遵守）

1. 全程使用中文输出、中文注释、中文提交信息。
2. 全程使用 `uv` 管理 Python 环境与依赖，禁止 `pip/conda`。
3. 每次仅执行一个任务，形成闭环：`理解 -> 测试 -> 编码 -> 验证 -> 提交 -> 文档`。
4. 所有改动必须满足分层依赖方向：`接口层 -> 应用层 -> 领域层 -> 基础设施层`，禁止反向依赖。
5. 每个任务完成后必须执行统一验证命令并记录结果。
6. 严禁使用“可选、建议、尽量、后续再看”等模糊指令。
7. 所有任务必须给出明确输入文件、输出文件、修改文件、验收标准。

---

## 2. 目标架构（分层、解耦、高内聚、低耦合）

### 2.1 目标目录结构（修复完成后必须达到）

```text
asl-lstm/
├── docs/
│   ├── Product-Spec.md
│   ├── Product-Spec-CHANGELOG.md
│   ├── feature_list.json
│   ├── progress.md
│   ├── API.md
│   └── Architecture.md
├── src/
│   ├── core/                      # 基础设施层（路径、配置、日志、IO）
│   │   ├── paths.py
│   │   ├── labels.py
│   │   └── hdf5_schema.py
│   ├── data_process/              # 数据处理应用层
│   ├── model/                     # 训练/评估/推理应用层
│   └── test/                      # 自动化测试
├── fix.md                         # 本修复计划（当前文件）
└── README.md
```

### 2.2 分层依赖规则（必须满足）

1. `src/core/*` 不能依赖 `src/model/*` 与 `src/data_process/*`。
2. `src/model/*` 与 `src/data_process/*` 可以依赖 `src/core/*`，不能相互复制公共逻辑。
3. 标签映射解析、路径构建、HDF5 结构校验必须集中在 `src/core/*`，禁止在多个脚本重复实现。

---

## 3. 统一质量门禁（每个任务结束后必须执行）

按顺序执行以下命令，全部通过才允许进入下一任务：

```bash
uv run python -m compileall src
uv run python -m unittest discover -s src/test -p "test_*.py"
```

从任务 T06 开始，新增并强制执行：

```bash
uv run ruff check src
```

门禁失败处理规则：
1. 同一任务内连续失败 3 次，立即停止编码。
2. 在 `docs/progress.md` 记录失败原因与定位结论。
3. 修复后重新执行完整门禁，全部通过后继续。

---

## 4. 任务总览（固定顺序，不允许调整）

| 任务ID | 任务名称 | 目标 | 依赖 |
|---|---|---|---|
| T00 | 建立文档基线 | 恢复 docs 单一事实来源 | 无 |
| T01 | 修复阻断脚本 | 修复 `transfer_hdf5_data.py` 运行错误 | T00 |
| T02 | 清除硬编码路径 | 移除绝对路径与旧项目名残留 | T01 |
| T03 | 标签映射契约统一 | 统一 `id_to_label/label_to_id` 语义 | T02 |
| T04 | 公共模块解耦 | 抽离路径/标签/HDF5 公共逻辑到 `src/core` | T03 |
| T05 | 文档与代码对齐 | README/API/架构文档与代码一致 | T04 |
| T06 | 引入静态检查 | 接入 ruff 并修复违规 | T05 |
| T07 | 补齐集成测试 | 预处理/训练/推理最小链路可回归 | T06 |
| T08 | 完成验收与封版 | 统一验收、更新进度、输出总结 | T07 |

---

## 5. 详细执行计划（逐任务明确实施）

## T00 建立文档基线

### 输入
- `README.md`
- `fix.md`

### 输出
- `docs/Product-Spec.md`
- `docs/Product-Spec-CHANGELOG.md`
- `docs/feature_list.json`
- `docs/progress.md`
- `docs/API.md`
- `docs/Architecture.md`

### 修改文件
- 创建 `docs/` 目录及上述 6 个文件。

### 执行步骤
1. 创建 `docs/` 目录。
2. 生成 `docs/Product-Spec.md`，内容必须包含：项目背景、用户角色、核心流程、功能性需求、非功能性需求。
3. 生成 `docs/Product-Spec-CHANGELOG.md`，写入首条记录：`初始化修复基线`。
4. 生成 `docs/feature_list.json`，写入 T00~T08 全部任务，状态初始化为 `pending`，T00 标记为 `in_progress`。
5. 生成 `docs/progress.md`，写入本次会话启动日志。
6. 生成 `docs/API.md` 与 `docs/Architecture.md` 初版目录结构。
7. 将 `docs/feature_list.json` 中 T00 更新为 `completed`，T01 更新为 `in_progress`。

### 验收标准
1. `docs/` 下 6 个文件全部存在且非空。
2. `docs/feature_list.json` 为合法 JSON，且包含 T00~T08。
3. `docs/progress.md` 至少包含 1 条时间戳日志。

---

## T01 修复阻断脚本（transfer_hdf5_data）

### 输入
- `src/data_process/transfer_hdf5_data.py`
- `src/config.py`

### 输出
- 可正常调用 `get_data_dir()`，不再抛 `AttributeError`。

### 修改文件
- `src/data_process/transfer_hdf5_data.py`

### 执行步骤
1. 将 `import config` 替换为 `import src.config as cfg`。
2. 将 `Path(config.PROJECT_ROOT)` 替换为 `Path(cfg.PATHS.project_root)`。
3. 全文检查并替换所有旧配置引用，统一使用 `cfg.PATHS`。
4. 执行命令：
   ```bash
   uv run python -c "import src.data_process.transfer_hdf5_data as m; print(m.get_data_dir('100'))"
   ```
5. 命令必须输出路径并退出码为 0。
6. 在 `docs/progress.md` 记录修复结果。

### 验收标准
1. 上述命令无异常。
2. 文件内不再出现 `config.PROJECT_ROOT`。
3. 通过统一质量门禁。

---

## T02 清除硬编码路径

### 输入
- `src/test/check_json.py`
- `src/test/check_coords.py`
- `src/test/processed_data_test.py`
- `src/data_process/count_dataset_samples.py`

### 输出
- 所有路径由配置或命令行参数驱动，不含机器绑定绝对路径。

### 修改文件
- `src/test/check_json.py`
- `src/test/check_coords.py`
- `src/test/processed_data_test.py`
- `src/data_process/count_dataset_samples.py`

### 执行步骤
1. 删除上述文件中的绝对路径字符串（`d:/Document/...`、`csl-lstm`）。
2. 统一增加参数解析：`--dataset-scale`、`--split`、`--data-root`。
3. 默认路径全部从 `cfg.PATHS.project_root` 推导。
4. 统一输出当前解析后的实际文件路径，便于排查。
5. 执行命令验证：
   ```bash
   uv run python src/data_process/count_dataset_samples.py --dataset-scale 100
   uv run python src/test/processed_data_test.py --dataset-scale 100 --sample-limit 1
   ```
6. 在 `docs/progress.md` 记录“硬编码路径已清零”。

### 验收标准
1. 代码中不再出现 `csl-lstm` 字样。
2. 代码中不再出现绝对路径盘符前缀（`d:/`、`D:\\`）。
3. 通过统一质量门禁。

---

## T03 标签映射契约统一

### 输入
- `src/data_process/preprocess_wlasl.py`
- `src/model/dataloader.py`
- `src/model/evaluate_lstm.py`
- `src/model/realtime_inference.py`

### 输出
- 标签映射 JSON 固定契约：
  - `id_to_label`: `{ "0": "book" }`
  - `label_to_id`: `{ "book": 0 }`

### 修改文件
- `src/data_process/preprocess_wlasl.py`
- `src/model/dataloader.py`
- `src/model/evaluate_lstm.py`
- `src/model/realtime_inference.py`
- 新增 `src/test/test_label_map_contract.py`

### 执行步骤
1. 在 `preprocess_wlasl.py` 中修正 `create_maplabels_json()` 生成逻辑，严格按契约写入。
2. 在 `dataloader.py` 中实现单一解析函数，读取契约并返回 `label_to_id`。
3. 在 `evaluate_lstm.py` 与 `realtime_inference.py` 复用同一解析逻辑，禁止重复实现。
4. 编写 `test_label_map_contract.py`，覆盖：
   - 正常契约读取。
   - 非法契约抛出明确异常。
   - 旧格式输入被拒绝并给出错误提示。
5. 执行新增测试并通过。
6. 在 `docs/API.md` 补充“标签映射文件格式”章节。

### 验收标准
1. 项目内仅存在一种标签契约。
2. 推理与评估共享同一解析实现。
3. 新增测试通过，且统一质量门禁通过。

---

## T04 公共模块解耦（高内聚低耦合）

### 输入
- `src/model/*`
- `src/data_process/*`

### 输出
- 新增 `src/core`，集中承载路径、标签、HDF5 结构公共逻辑。

### 修改文件
- 新增 `src/core/paths.py`
- 新增 `src/core/labels.py`
- 新增 `src/core/hdf5_schema.py`
- 修改 `src/model/evaluate_lstm.py`
- 修改 `src/model/realtime_inference.py`
- 修改 `src/data_process/read_data_struct.py`
- 修改 `src/data_process/verify_consistency.py`

### 执行步骤
1. 在 `src/core/paths.py` 实现路径构建函数，替代脚本内路径拼接重复代码。
2. 在 `src/core/labels.py` 实现标签映射加载/校验函数，替代重复 `load_label_map_inverse()`。
3. 在 `src/core/hdf5_schema.py` 定义 HDF5 必填字段常量与校验函数。
4. 所有调用方改为 import `src.core.*`。
5. 删除调用方中的重复实现函数。
6. 新增测试 `src/test/test_core_modules.py` 覆盖核心公共模块。

### 验收标准
1. `evaluate_lstm.py` 与 `realtime_inference.py` 不再各自维护标签解析函数。
2. HDF5 字段校验逻辑在项目内仅有一处实现。
3. 通过统一质量门禁。

---

## T05 文档与代码对齐

### 输入
- `README.md`
- `docs/*`
- 当前代码结构

### 输出
- 文档与实际代码一致，无失效说明。

### 修改文件
- `README.md`
- `docs/API.md`
- `docs/Architecture.md`
- `docs/Product-Spec-CHANGELOG.md`

### 执行步骤
1. 删除 README 中不存在的文件描述或恢复对应文件。
2. 修正 README 中“置信度过滤”描述，确保与实时推理实现一致。
3. 在 `docs/Architecture.md` 明确四层依赖图与禁止依赖规则。
4. 在 `docs/API.md` 增加：HDF5 结构契约、maplabels 契约、脚本参数说明。
5. 在变更日志中记录本次对齐条目。

### 验收标准
1. README 中每个路径均在仓库存在。
2. README 不得出现与实现不一致的能力宣称。
3. 通过统一质量门禁。

---

## T06 引入静态检查并收敛规范

### 输入
- `pyproject.toml`
- 全部 `src/*.py`

### 输出
- `ruff` 纳入项目工具链并全量通过。

### 修改文件
- `pyproject.toml`
- 代码文件（按 lint 结果修改）

### 执行步骤
1. 在 `pyproject.toml` 中加入 `ruff` 依赖与配置。
2. 执行：
   ```bash
   uv sync
   uv run ruff check src
   ```
3. 逐条修复 `ruff` 报告直到 0 告警。
4. 将 `ruff check` 加入统一质量门禁。

### 验收标准
1. `uv run ruff check src` 返回码为 0。
2. 统一质量门禁全部通过。

---

## T07 补齐集成测试（防回归）

### 输入
- 训练、预处理、推理入口脚本

### 输出
- 最小链路自动化测试集，覆盖关键路径。

### 修改文件
- 新增 `src/test/test_preprocess_smoke.py`
- 新增 `src/test/test_train_smoke.py`
- 新增 `src/test/test_inference_smoke.py`

### 执行步骤
1. 为预处理入口构建最小样本数据，验证输出结构合法。
2. 为训练入口构建最小 epoch smoke test（不追求精度，仅验证流程可跑通）。
3. 为推理入口构建无摄像头模式 smoke test（使用离线帧序列）。
4. 将三类测试纳入 `test_*.py` 发现范围。
5. 执行统一质量门禁。

### 验收标准
1. 三个 smoke test 全部通过。
2. 任一核心入口回归均能被自动测试捕获。

---

## T08 完成验收与封版

### 输入
- 所有前置任务产物

### 输出
- 完整修复交付包与最终状态更新。

### 修改文件
- `docs/feature_list.json`
- `docs/progress.md`
- `docs/Product-Spec-CHANGELOG.md`
- `README.md`（仅在必要时）

### 执行步骤
1. 执行最终全量验证：
   ```bash
   uv run python -m compileall src
   uv run python -m unittest discover -s src/test -p "test_*.py"
   uv run ruff check src
   ```
2. 将 `docs/feature_list.json` 的 T00~T08 全部标记为 `completed`。
3. 在 `docs/progress.md` 写入最终验收日志（包含命令与结果）。
4. 在 `docs/Product-Spec-CHANGELOG.md` 写入“修复版本发布记录”。
5. 输出最终修复报告（问题清单 -> 修复映射 -> 验证证据）。

### 验收标准
1. 全量验证命令全部通过。
2. 文档状态与代码状态一致。
3. 修复报告可独立复核。

---

## 6. 强制提交规范（每个任务完成后执行）

提交信息格式固定：

```text
<type>(<scope>): <中文动作描述> [Txx]
```

示例：

```text
fix(data_process): 修复 transfer_hdf5_data 配置引用错误 [T01]
refactor(core): 抽离标签映射与路径公共模块 [T04]
test(integration): 增加训练与推理最小链路回归测试 [T07]
```

禁止事项：
1. 禁止一次提交跨越多个任务 ID。
2. 禁止跳过测试直接提交。
3. 禁止提交未更新 `docs/progress.md` 的代码。

---

## 7. 执行检查清单（每次会话开始与结束必须勾选）

### 开始前检查
- [ ] 已读取 `fix.md`
- [ ] 已读取 `docs/feature_list.json`
- [ ] 当前仅有一个 `in_progress` 任务
- [ ] 已明确本次任务输入/输出文件

### 结束前检查
- [ ] 代码改动与任务范围一致
- [ ] 统一质量门禁通过
- [ ] `docs/progress.md` 已更新
- [ ] `docs/feature_list.json` 状态已推进
- [ ] 提交信息符合规范

---

## 8. 最终说明

本计划已经将“问题 -> 任务 -> 文件 -> 命令 -> 验收”完全固化。
后续 AI Agent 必须逐条执行，不得改变任务顺序，不得新增分支流程，不得输出含糊结论。
