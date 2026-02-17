# ASL-LSTM 架构文档

## 1. 架构目标

通过分层解耦实现高内聚、低耦合，减少重复实现，提升可测试性与可维护性。

## 2. 目标目录结构

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
│   ├── core/          # 基础设施层（路径、标签、HDF5 契约）
│   ├── data_process/  # 数据处理应用层
│   ├── model/         # 训练/评估/推理应用层
│   └── test/          # 自动化测试
```

## 3. 四层依赖图

```text
接口层（CLI 脚本入口）
  -> 应用层（src/model, src/data_process）
    -> 领域层（任务逻辑、训练流程、预处理流程）
      -> 基础设施层（src/core：路径、标签契约、HDF5 契约）
```

## 4. 禁止依赖规则

1. `src/core/*` 不能依赖 `src/model/*` 与 `src/data_process/*`。
2. `src/model/*` 与 `src/data_process/*` 可以依赖 `src/core/*`，但不允许互相复制公共逻辑。
3. 标签映射解析、路径构建、HDF5 字段校验必须只在 `src/core/*` 维护。
4. 新增脚本不得在应用层重复定义标签或 HDF5 契约校验函数。

## 5. 当前核心模块

- `src/core/paths.py`: 统一路径构建函数。
- `src/core/labels.py`: 标签映射加载与契约校验。
- `src/core/hdf5_schema.py`: HDF5 必填字段常量与校验函数。
- `src/model/*`: 训练、评估、推理流程。
- `src/data_process/*`: 预处理、迁移、结构检查与一致性验证。
- `src/test/*`: 单元测试与 smoke 测试。
