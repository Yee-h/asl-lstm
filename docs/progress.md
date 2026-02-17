# 项目修复进度日志

## [2026-02-17 15:54:30] 会话启动
- **目标**: 按 `fix.md` 固定顺序执行 T00~T08 修复任务。
- **状态**: 已读取修复计划并完成现状扫描。
- **动作**: 开始执行 T00（建立文档基线）。

## [2026-02-17 15:58:00] T00 文档基线建立
- **结果**: 已创建 `docs/` 及 6 个核心文档文件。
- **状态推进**: `T00 -> completed`，`T01 -> in_progress`。
- **备注**: 待执行统一质量门禁并记录结果。

## [2026-02-17 16:01:00] T00 统一质量门禁
- **命令**: `uv run python -m compileall src`
- **结果**: 通过。
- **命令**: `uv run python -m unittest discover -s src/test -p "test_*.py"`
- **结果**: 通过（16 项测试）。

## [2026-02-17 16:05:00] T01 修复完成
- **变更**: `src/data_process/transfer_hdf5_data.py` 改为 `import src.config as cfg`，并统一使用 `cfg.PATHS.project_root` 推导数据路径。
- **验证命令**: `uv run python -c "import src.data_process.transfer_hdf5_data as m; print(m.get_data_dir('100'))"`
- **验证结果**: 正常输出路径并返回码 0。
- **质量门禁**:
  - `uv run python -m compileall src` 通过。
  - `uv run python -m unittest discover -s src/test -p "test_*.py"` 通过（16 项测试）。
- **状态推进**: `T01 -> completed`，`T02 -> in_progress`。

## [2026-02-17 16:12:00] T02 修复完成
- **变更文件**:
  - `src/data_process/count_dataset_samples.py`
  - `src/test/processed_data_test.py`
  - `src/test/check_json.py`
  - `src/test/check_coords.py`
- **修复结果**: 硬编码路径已清零，统一提供 `--dataset-scale`、`--split`、`--data-root` 参数，并输出解析后的实际路径。
- **验证命令**:
  - `uv run python src/data_process/count_dataset_samples.py --dataset-scale 100`
  - `uv run python src/test/processed_data_test.py --dataset-scale 100 --sample-limit 1`
- **质量门禁**:
  - `uv run python -m compileall src` 通过。
  - `uv run python -m unittest discover -s src/test -p "test_*.py"` 通过（16 项测试）。
- **状态推进**: `T02 -> completed`，`T03 -> in_progress`。

## [2026-02-17 16:22:00] T03 修复完成
- **核心改动**:
  - `src/data_process/preprocess_wlasl.py`: 修正 `create_maplabels_json()` 输出契约为 `id_to_label={"0":"book"}` 与 `label_to_id={"book":0}`。
  - `src/model/dataloader.py`: 新增统一解析函数 `load_label_to_id_map` / `load_id_to_label_map`，严格校验契约并拒绝旧格式。
  - `src/model/evaluate_lstm.py`、`src/model/realtime_inference.py`: 改为复用 dataloader 统一解析逻辑。
  - 新增测试 `src/test/test_label_map_contract.py`。
- **新增测试**: `uv run python -m unittest src.test.test_label_map_contract -v` 通过（3 项测试）。
- **文档更新**: `docs/API.md` 已补充“标签映射文件格式”正式契约。
- **质量门禁**:
  - `uv run python -m compileall src` 通过。
  - `uv run python -m unittest discover -s src/test -p "test_*.py"` 通过（19 项测试）。
- **状态推进**: `T03 -> completed`，`T04 -> in_progress`。

## [2026-02-17 16:33:00] T04 修复完成
- **新增核心模块**:
  - `src/core/paths.py`
  - `src/core/labels.py`
  - `src/core/hdf5_schema.py`
- **调用方改造**:
  - `src/model/evaluate_lstm.py` 与 `src/model/realtime_inference.py` 改为复用 `src.core.labels`。
  - `src/data_process/read_data_struct.py` 改为复用 `src.core.paths` 和 `src.core.hdf5_schema`。
  - `src/data_process/verify_consistency.py` 改为复用 `src.core.paths`、`src.core.labels`、`src.core.hdf5_schema`。
  - `src/data_process/transfer_hdf5_data.py` 的字段校验改为复用 `src.core.hdf5_schema`。
- **新增测试**: `src/test/test_core_modules.py`。
- **质量门禁**:
  - `uv run python -m compileall src` 通过。
  - `uv run python -m unittest discover -s src/test -p "test_*.py"` 通过（23 项测试）。
- **状态推进**: `T04 -> completed`，`T05 -> in_progress`。

## [2026-02-17 16:40:00] T05 修复完成
- **文档对齐**:
  - `README.md`：修正项目结构路径、任务追踪路径和推理能力描述。
  - `docs/API.md`：补充 HDF5 契约、标签契约、脚本参数说明。
  - `docs/Architecture.md`：补充四层依赖图与禁止依赖规则。
  - `docs/Product-Spec-CHANGELOG.md`：记录文档对齐变更。
- **质量门禁**:
  - `uv run python -m compileall src` 通过。
  - `uv run python -m unittest discover -s src/test -p "test_*.py"` 通过（23 项测试）。
- **状态推进**: `T05 -> completed`，`T06 -> in_progress`。

## [2026-02-17 16:48:00] T06 修复完成
- **工具链变更**: `pyproject.toml` 新增 `ruff` 依赖与配置。
- **执行命令**:
  - `uv sync`
  - `uv run ruff check src`
- **结果**: `ruff` 已接入并实现 0 告警。
- **质量门禁**:
  - `uv run python -m compileall src` 通过。
  - `uv run python -m unittest discover -s src/test -p "test_*.py"` 通过（23 项测试）。
  - `uv run ruff check src` 通过。
- **状态推进**: `T06 -> completed`，`T07 -> in_progress`。

## [2026-02-17 17:02:00] T07 修复完成
- **新增集成测试**:
  - `src/test/test_preprocess_smoke.py`
  - `src/test/test_train_smoke.py`
  - `src/test/test_inference_smoke.py`
- **覆盖能力**:
  - 预处理最小样本输出结构校验。
  - 训练入口最小 1 epoch 流程校验。
  - 推理入口离线帧序列前向校验（无摄像头）。
- **质量门禁**:
  - `uv run python -m compileall src` 通过。
  - `uv run python -m unittest discover -s src/test -p "test_*.py"` 通过（26 项测试）。
  - `uv run ruff check src` 通过。
- **状态推进**: `T07 -> completed`，`T08 -> in_progress`。

## [2026-02-17 17:08:00] T08 最终验收完成
- **最终全量验证命令**:
  - `uv run python -m compileall src`
  - `uv run python -m unittest discover -s src/test -p "test_*.py"`
  - `uv run ruff check src`
- **结果**:
  - 编译检查通过。
  - 单元测试通过（26 项）。
  - Ruff 静态检查通过（0 告警）。
- **状态推进**: `docs/feature_list.json` 中 `T00~T08` 全部更新为 `completed`。

## [2026-02-17 16:33:48] 运行兼容性修复（后验）
- **触发问题**: 离线推理启动时读取历史旧格式 `maplabels` 文件，触发 `ValueError`。
- **修复动作**:
  - `src/core/labels.py` 新增旧格式迁移能力：`migrate_legacy_label_map_file`。
  - 新增兼容加载函数：`load_label_to_id_map_compat`、`load_id_to_label_map_compat`。
  - `src/model/dataloader.py`、`src/model/evaluate_lstm.py`、`src/model/realtime_inference.py` 改为运行时兼容加载。
- **验证证据**:
  - `.venv` 解释器执行 `src/model/realtime_inference.py --camera -1`，不再出现标签映射契约异常。
  - `.venv` 解释器执行 `src/data_process/preprocess_wlasl.py --limit 1 --output-dir dataset/processed_smoke3` 成功。
  - `.venv` 解释器执行 `python -m unittest src.test.test_train_smoke -v` 通过。
  - `.venv` 解释器执行 `python -m unittest src.test.test_inference_smoke -v` 通过。
  - `uv run python -m compileall src && uv run python -m unittest discover -s src/test -p "test_*.py" && uv run ruff check src` 通过（27 项测试，Ruff 0 告警）。
