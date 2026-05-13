"""从 summary.json 导出 Markdown 对比表格（核心指标 + 架构参数）。"""
import json
import os

summary_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "summary.json",
)

with open(summary_path, "r", encoding="utf-8") as f:
    data = json.load(f)

groups = [
    ("基线模型（不同随机种子）", ["seed42", "seed123", "seed456", "seed789", "seed2024", "seed4096"]),
    ("时间掩码增强（E05）", ["seed42_temporal_mask", "seed123_temporal_mask", "seed456_temporal_mask", "seed789_temporal_mask"]),
    ("架构变体实验", ["seed456_layernorm", "seed456_multihead", "seed456_bilstm192", "seed456_accel6ch"]),
    ("训练策略实验", ["seed456_warmup"]),
    ("模型集成", ["ensemble_temporal_mask"]),
]

labels = {
    "seed42": "Seed=42（基线）",
    "seed123": "Seed=123",
    "seed456": "Seed=456",
    "seed789": "Seed=789",
    "seed2024": "Seed=2024",
    "seed4096": "Seed=4096",
    "seed42_temporal_mask": "Seed=42 + TM",
    "seed123_temporal_mask": "Seed=123 + TM",
    "seed456_temporal_mask": "Seed=456 + TM",
    "seed789_temporal_mask": "Seed=789 + TM",
    "seed456_layernorm": "+ LayerNorm（E04）",
    "seed456_multihead": "+ MultiHead k=4（E06）",
    "seed456_bilstm192": "Hidden=192（E07a）",
    "seed456_accel6ch": "+ Accel 6ch（E08）",
    "seed456_warmup": "+ LR Warmup（E03）",
    "ensemble_temporal_mask": "4-Model Ensemble（Softmax Avg）",
}

results_by_name = {r["model_name"]: r for r in data["results"]}

lines = []
lines.append("## 模型在 WLASL100 测试集上的性能对比")
lines.append("")

# ── 表1：核心性能指标 ──
lines.append("### 核心性能指标")
lines.append("")
lines.append("| 实验分组 | 模型 | Top-1 (%) | Top-5 (%) | Macro-P (%) | Macro-R (%) | Macro-F1 (%) |")
lines.append("|---------|------|-----------|-----------|-------------|-------------|-------------|")

for group_name, model_names in groups:
    first = True
    for name in model_names:
        if name not in results_by_name:
            continue
        r = results_by_name[name]
        group_cell = f"**{group_name}**" if first else ""
        label = labels.get(name, name)

        cr = r.get("classification_report", {})
        macro_p = cr.get("macro avg", {}).get("precision", 0) * 100
        macro_r = cr.get("macro avg", {}).get("recall", 0) * 100
        macro_f1 = cr.get("macro avg", {}).get("f1-score", 0) * 100

        top1 = r["test_accuracy"]
        top5 = r.get("test_top5_accuracy", 0)

        lines.append(
            f"| {group_cell} | {label} | {top1:.2f} | {top5:.2f} | "
            f"{macro_p:.2f} | {macro_r:.2f} | {macro_f1:.2f} |"
        )
        first = False

# ── 表2：模型架构参数 ──
lines.append("")
lines.append("### 模型架构参数")
lines.append("")
lines.append("| 模型 | 参数量 | Input | Hidden | Heads | LayerNorm |")
lines.append("|------|--------|-------|--------|-------|-----------|")

for group_name, model_names in groups:
    for name in model_names:
        if name not in results_by_name:
            continue
        r = results_by_name[name]
        label = labels.get(name, name)
        params = r.get("params", {})
        param_count = r.get("param_count", 0)
        input_size = params.get("input_size", "-")
        hidden_size = params.get("hidden_size", "-")
        num_heads = params.get("num_heads", "-")
        use_ln = "Y" if params.get("use_layer_norm") else "N"

        if name == "ensemble_temporal_mask":
            param_str = f"{param_count:,}（4×）"
            num_heads = "-"
        else:
            param_str = f"{param_count:,}"

        lines.append(
            f"| {label} | {param_str} | {input_size} | {hidden_size} | "
            f"{num_heads} | {use_ln} |"
        )

# ── 统计汇总 ──
lines.append("")
lines.append("### 统计汇总")
baseline_accs = [results_by_name[n]["test_accuracy"] for n in groups[0][1] if n in results_by_name]
tm_accs = [results_by_name[n]["test_accuracy"] for n in groups[1][1] if n in results_by_name]
lines.append(f"- 基线模型平均 Top-1：{sum(baseline_accs)/len(baseline_accs):.2f}%")
lines.append(f"- TM 增强平均 Top-1：{sum(tm_accs)/len(tm_accs):.2f}%（提升 +{sum(tm_accs)/len(tm_accs)-sum(baseline_accs)/len(baseline_accs):.2f}%）")
lines.append(f"- 集成模型 Top-1 / Top-5：{results_by_name['ensemble_temporal_mask']['test_accuracy']:.2f}% / {results_by_name['ensemble_temporal_mask'].get('test_top5_accuracy', 0):.2f}%")
lines.append(f"- 所有模型在 100 类手语词汇、258 个测试样本上评估")

output_path = os.path.join(os.path.dirname(summary_path), "comparison_table.md")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"表格已保存至: {output_path}")
print()
print("\n".join(lines))
