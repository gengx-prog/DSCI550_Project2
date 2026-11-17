"""Build the Chinese optimization handoff from completed, audited artifacts.

Refuses incomplete experiments or live handoffs instead of publishing an empty
report. All displayed scores and checks come from the selected run's artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["O0_small_200k", "O1_medium_600k_ensemble",
          "O2_medium_1200k_ensemble", "O3_regularized_1200k_ensemble"]
LABELS = {
    "O0_small_200k": "O0：small 20万单模型",
    "O1_medium_600k_ensemble": "O1：medium 60万三种子",
    "O2_medium_1200k_ensemble": "O2：medium 120万三种子",
    "O3_regularized_1200k_ensemble": "O3：medium 120万正则化集成",
    "B0_fixed_uploaded": "固定 B0（本地参照）",
    "B0_fixed": "固定 B0（本地参照）",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value, decimals: int = 6) -> str:
    numeric = float(value)
    require(np.isfinite(numeric), "Cannot report a missing or nonfinite numeric result")
    return f"{numeric:.{decimals}f}"


def percent(value) -> str:
    return f"{100 * float(value):.1f}%"


def date(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def label(model: str, selected: str | None = None) -> str:
    result = LABELS[model]
    return result + "（CV入选）" if model == selected else result


def link(path: Path, base: Path, title: str) -> str:
    target = os.path.relpath(path, base).replace("\\", "/")
    return f"[{title}](<{target}>)"


def table(headers: list[str], rows: list[list]) -> str:
    require(bool(rows), "Refusing to create an empty result table")
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def metric_row(metrics: pd.DataFrame, phase: str, model: str, metric: str) -> pd.Series:
    row = metrics.loc[metrics.phase.eq(phase) & metrics.model.eq(model) & metrics.metric.eq(metric)]
    require(len(row) == 1, f"Missing or duplicated metric: {phase}/{model}/{metric}")
    return row.iloc[0]


def phase_table(metrics: pd.DataFrame, phase: str, models: list[str], selected: str) -> str:
    rows = []
    for model in models:
        for metric in ("CORR", "NCORR"):
            record = metric_row(metrics, phase, model, metric)
            rows.append([label(model, selected), metric, number(record["mean"]),
                         number(record.std_ddof0), number(record.sharpe, 3),
                         number(record.max_drawdown), percent(record.positive_fraction)])
    return table(["方案", "指标", "逐日期均值", "总体标准差", "mean/std", "累计分数最大回撤", "正分日期率"], rows)


def build(output: Path, live_output: Path) -> Path:
    output, live_output = output.resolve(), live_output.resolve()
    run = read_json(output / "run.json")
    validation = read_json(output / "validation.json")
    live = read_json(live_output / "live_generation.json")
    live_validation = read_json(live_output / "validation.json")
    require(run.get("status") == "complete", "Experiment has not completed")
    require(validation.get("status") == "passed", "Historical validation has not passed")
    require(live.get("status") == "complete" and live_validation.get("status") == "passed",
            "Live generation and its validation must both complete before reporting")
    for name, expected_hash in run.get("output_sha256", {}).items():
        require(sha256(output / name) == expected_hash, f"Completed result has changed: {name}")
    plan = read_json(output / "plan.json")
    selection = read_json(output / "selection.json")
    final = read_json(output / "final_training.json")
    comparisons = read_json(output / "paired_comparison.json")
    bundle = read_json(Path(run["final_bundle"]))
    selected = selection["selected_model"]
    require(selected in MODELS and selected == run["selected_model"] == final["selected_model"]
            == bundle["model_id"] == live["model_name"], "Winner identity differs across artifacts")
    require(selection.get("development_used_for_selection") is False,
            "Report requires a CV-only frozen selection")
    metrics = pd.read_csv(output / "metrics.csv")
    scores = pd.read_csv(output / "per_date_metrics.csv", parse_dates=["date"])
    new_dates = pd.read_csv(output / "newly_matured_dates.csv", parse_dates=["date"])
    require(not metrics.empty and not scores.empty and not new_dates.empty, "Incomplete metric exports")
    reference = plan["reference_model"]["id"]
    require(reference in LABELS and set(scores.model) == set(MODELS + [reference]), "Scoring model coverage differs")
    require(not scores.duplicated(["phase", "fold", "date", "model"]).any(), "Duplicate date/model score")
    require(set(scores.phase) == {"cv", "reused_development"}, "Unexpected evaluation phases")
    numeric = metrics[["mean", "std_ddof0", "sharpe", "max_drawdown", "positive_fraction"]].to_numpy(float)
    require(np.isfinite(numeric).all(), "Metrics contain nonfinite values")
    csv_path = live_output / live["csv_file"]
    require(sha256(csv_path) == live["csv_sha256"] == live_validation["csv_sha256"], "Live CSV hash differs")
    require(len(run["completed_folds"]) == len(plan["folds"]) == len(validation["folds"]), "Fold completion/audit counts differ")
    models = [reference] + MODELS
    cv = scores.loc[scores.phase.eq("cv")]
    dev = scores.loc[scores.phase.eq("reused_development")]
    winner_dev = metric_row(metrics, "reused_development", selected, "NCORR")
    o1_dev = metric_row(metrics, "reused_development", MODELS[1], "NCORR")
    o2_dev = metric_row(metrics, "reused_development", MODELS[2], "NCORR")
    reference_dev = metric_row(metrics, "reused_development", reference, "NCORR")
    root = Path(run["data_root"])
    feature_metadata = read_json(root / "data/raw/v6.0/features.json")
    feature_counts = {name: len([f for f in feature_metadata["feature_sets"][name] if f != "feature_country"])
                      for name in ("small", "medium")}
    candidate_rows = []
    for model in MODELS:
        spec = plan["candidates"][model]
        params = spec["params"]
        candidate_rows.append([label(model, selected), f"{int(spec['max_rows']):,}",
                               feature_counts[spec["feature_set"]], ", ".join(map(str, spec["seeds"])),
                               params["n_estimators"], params["learning_rate"], params["colsample_bytree"]])
    selection_rows = [[label(model, selected), number(selection["candidates"][model]["mean_NCORR"]),
                       number(selection["candidates"][model]["fold_mean_std_ddof0"]),
                       number(selection["candidates"][model]["selection_score"])] for model in MODELS]
    annual_2025 = dev.loc[dev.date.dt.year.eq(2025)]
    require(not annual_2025.empty, "Missing 2025 diagnostics")
    year_rows = []
    for model in models:
        rows = annual_2025.loc[annual_2025.model.eq(model)]
        require(not rows.empty, "Incomplete 2025 model coverage")
        year_rows.append([label(model, selected), rows.date.nunique(), number(rows.CORR.mean()),
                          number(rows.NCORR.mean()), percent(rows.NCORR.gt(0).mean())])
    new_rows = []
    for when, rows in new_dates.groupby("date", sort=True):
        require(set(rows.model) == set(models), "New-date score population differs")
        selected_rows = rows.loc[rows.model.eq(selected)].iloc[0]
        o1_rows = rows.loc[rows.model.eq(MODELS[1])].iloc[0]
        reference_rows = rows.loc[rows.model.eq(reference)].iloc[0]
        new_rows.append([date(when), f"{int(selected_rows.n_rows):,}", number(selected_rows.CORR),
                         number(selected_rows.NCORR), number(o1_rows.NCORR), number(reference_rows.NCORR)])
    bootstrap_rows = []
    for baseline in (reference, MODELS[0], MODELS[1]):
        comparison = comparisons[baseline]
        require(comparison["selected_model"] == selected and comparison["baseline_model"] == baseline,
                "Paired comparison identity differs")
        intervals = comparison["block_bootstrap"]
        bootstrap_rows.append([label(baseline), number(comparison["observed_mean_NCORR_delta"]),
                               f"[{number(intervals['13']['ci95_low'])}, {number(intervals['13']['ci95_high'])}]",
                               f"[{number(intervals['26']['ci95_low'])}, {number(intervals['26']['ci95_high'])}]"])
    observed_new = new_dates.loc[new_dates.model.eq(selected)]
    o1_delta = float(winner_dev["mean"] - o1_dev["mean"])
    o1_relative = o1_delta / abs(float(o1_dev["mean"]))
    o1_interval = comparisons[MODELS[1]]["block_bootstrap"]["13"]
    o1_interval_contains_zero = float(o1_interval["ci95_low"]) <= 0 <= float(o1_interval["ci95_high"])
    interval_statement = (
        "相对 O1 的主配对差值区间包含零，当前历史证据不足以确认平均分优势。"
        if o1_interval_contains_zero else
        "相对 O1 的主配对差值区间未包含零，但它仍未校正选型及之前的开发，不能作为确认性优势。"
    )
    winner_2025 = float(annual_2025.loc[annual_2025.model.eq(selected), "NCORR"].mean())
    o1_2025 = float(annual_2025.loc[annual_2025.model.eq(MODELS[1]), "NCORR"].mean())
    stability_statement = (
        "稳定性指标没有优于 O1。"
        if float(winner_dev.sharpe) <= float(o1_dev.sharpe)
        and float(winner_dev.max_drawdown) >= float(o1_dev.max_drawdown) else
        "这些指标各自反映不同的历史风险，平均分差值不能代表全面胜出。"
    )
    final_cutoff = pd.Timestamp(final["train_cutoff_exclusive"])
    live_date = pd.Timestamp(live["live_date"])
    training_max = pd.Timestamp(bundle["full_training_date_max"])
    require(final_cutoff == live_date - pd.offsets.BDay(65) and training_max < final_cutoff,
            "Final live fit does not satisfy the reported embargo")
    max_corr_error = max(float(fold["all_dates_CORR_max_abs_error"]) for fold in validation["folds"])
    max_ncorr_error = max(float(fold["independent_NCORR_sample"]["NCORR_max_abs_error"])
                          for fold in validation["folds"])
    validation_dates = sum(int(fold["prediction_dates"]) for fold in validation["folds"])
    validation_rows = sum(int(fold["prediction_rows"]) for fold in validation["folds"])
    summary_count = int(validation["summary_records_recomputed"])
    elapsed = float(run["elapsed_seconds"])
    peak_gib = float(run["memory"]["peak_rss_gib"])
    tests_path = output / "tests.log"
    test_log = tests_path.read_text(encoding="utf-8")
    test_matches = re.findall(r"Ran (\d+) tests? in", test_log)
    require(bool(test_matches) and re.search(r"(?m)^OK\s*$", test_log) is not None,
            "A passed test-suite log is required before reporting")
    test_count = int(test_matches[-1])
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot_arg = os.path.relpath(root, ROOT).replace("\\", "/")
    baseline_arg = os.path.relpath(Path(plan["reference_model"]["directory"]), ROOT).replace("\\", "/")
    live_data_arg = os.path.relpath(Path(plan["final_fit"]["live_data"]), ROOT).replace("\\", "/")
    output_arg = os.path.relpath(output, ROOT).replace("\\", "/")
    live_arg = os.path.relpath(live_output, ROOT).replace("\\", "/")
    lines = [
        "# Numerai v6 模型优化与提交交接", "",
        f"本轮 CV 入选 **{label(selected)}**，作为待 forward 验证的新候选。"
        f"在 {date(dev.date.min())} 至 {date(dev.date.max())} 的已观察历史开发区间，"
        f"平均 NCORR 从同协议 O1 的 {number(o1_dev['mean'])} 变为 **{number(winner_dev['mean'])}**，"
        f"差值为 {number(o1_delta)}（相对变化 {100 * o1_relative:+.2f}%）。", "",
        f"**平均分变化没有带来全面的稳定性改善。** NCORR mean/std 从 O1 的 {number(o1_dev.sharpe, 3)} "
        f"变为 {number(winner_dev.sharpe, 3)}，累计分数最大回撤从 {number(o1_dev.max_drawdown)} "
        f"变为 {number(winner_dev.max_drawdown)}；2025 年平均 NCORR 从 {number(o1_2025)} "
        f"变为 {number(winner_2025)}。{stability_statement}{interval_statement}"
        "建议保留 O1 和固定 baseline 作为对照，在后续未参与选型的轮次继续检验，"
        "当前不能把这个挑战者称为已经全面胜出的替换方案。这些均为离线历史分数，不是未来 live 收益。", "",
        "用户截图中的模型名为 `ZEXUGAO`，显示 **+3.4% Simulated**，CORR60 与 MMC60 均为空。"
        "截图没有提供足以诊断模型泛化的评分序列，模拟数值也不是本次训练或选型目标。"
        "`B0_fixed_uploaded` 是仓库中先前 baseline 的本地参照标识；未核实该文件是否对应截图账号或其模型。", "",
        "## 本次改动与比较范围", "",
        f"数据使用独立快照 `{snapshot_arg}`，版本 `{run['data_version']}`、目标 `{run['target']}`。"
        f"研究池保留 {int(plan['source_provenance']['rows']):,} 条成熟记录，"
        f"最近成熟日期为 {plan['source_provenance']['pool_date_max']}。原 baseline、历史结果和既有提交均保留。", "",
        table(["方案", "每成员最多训练行", "数值特征", "模型种子", "树数", "学习率", "列采样"], candidate_rows), "",
        "四个方案均按年度重训，每个年度只使用早于实际首个评分日期减去 65 个工作日的记录；"
        "工作日按 pandas BDay 的周一至周五定义。抽样先执行时间截断，再从同一个 seed 7 的最多 120 万行排列取前缀；"
        "O1 的 60 万行嵌套在 O2 的 120 万行中。三个成员分别在每个预测日期进行保留并列的百分位排名，再等权平均。", "",
        "O0/O1/O2 使用 300 棵树、最大深度 5、31 个叶子。O3 同时改变树数、学习率、列采样、"
        "叶节点最小样本数 1000、L2 正则化 10 及 0.8 行采样，因此它检验的是一组联合配置。", "",
        f"**可归因的样本量对照：** O1 与 O2 的特征、模型种子、参数、时间切分和集成方法相同；"
        f"开发期 NCORR 从 {number(o1_dev['mean'])} 变为 {number(o2_dev['mean'])}，"
        f"差值为 {number(o2_dev['mean'] - o1_dev['mean'])}。固定 B0 不参与候选选择，其训练历史与重训频率也不同，"
        "与它的差值仅反映总体流程变化，不能全部归因于样本增加或集成。", "",
        "## CV 选型", "",
        f"CV 覆盖 {date(cv.date.min())} 至 {date(cv.date.max())}，共 {cv.date.nunique()} 个日期、四个年度。"
        "固定选择规则为：逐日期等权平均 NCORR，减去四个年度平均 NCORR 的总体标准差的 0.5 倍；"
        "并列按候选编号排序。选择记录在本轮开发区间评分前冻结。CV 也已被项目开发使用，不能视为新的独立测试。", "",
        table(["方案", "CV 平均 NCORR", "年度均值标准差", "选择分数"], selection_rows), "",
        f"冻结记录：{link(output / 'selection.json', output, 'selection.json')}。"
        "2023 年起的结果继续标记为 `reused_development`，不会据此重新挑选方案。", "",
        "## 两个历史区间的完整指标", "",
        "mean/std 为非年化的逐日期均值除以总体标准差。最大回撤按从零开始的累计相关分数计算；"
        "它不是投资组合回撤或实际收益率。2019、2022、2026 的评分区间不是完整自然年。", "",
        f"### CV：{date(cv.date.min())}—{date(cv.date.max())}（{cv.date.nunique()} 个日期）", "",
        phase_table(metrics, "cv", models, selected), "",
        f"### 已观察开发：{date(dev.date.min())}—{date(dev.date.max())}（{dev.date.nunique()} 个日期）", "",
        phase_table(metrics, "reused_development", models, selected), "",
        f"数值来源：{link(output / 'metrics.csv', output, '汇总指标')}、"
        f"{link(output / 'per_date_metrics.csv', output, '逐日期指标')}。", "",
        "![历史平均 NCORR 对照](figures/ncorr_comparison.png)", "",
        "![已观察开发期累计 NCORR](figures/cumulative_ncorr.png)", "",
        "## 2025 年与新增成熟日期", "",
        f"2025 年实际覆盖 {date(annual_2025.date.min())} 至 {date(annual_2025.date.max())}。"
        "年度对照用于发现不同时期的表现差异；没有据此追溯调参，也不足以确定表现变化的市场原因。", "",
        table(["方案", "日期数", "平均 CORR", "平均 NCORR", "NCORR 正分日期率"], year_rows), "",
        "以下是相对于旧研究池截止日 2026-07-24 新增的成熟日期，仍属于本项目已经可见的历史开发数据：", "",
        table(["日期", "记录数", "入选 CORR", "入选 NCORR", "O1 NCORR", "固定 B0 NCORR"], new_rows), "",
        f"入选模型在这 {observed_new.date.nunique()} 个日期的平均 NCORR 为 {number(observed_new.NCORR.mean())}。"
        "观测日期很少且 60 日目标相互重叠，不能单独证明稳定优势。"
        f"全方案明细见 {link(output / 'newly_matured_dates.csv', output, '新增成熟日期')}。", "",
        "## 配对差值及其限制", "",
        "下表在完全相同日期和股票总体上比较入选模型与各参照的 NCORR。"
        "使用 2,000 次圆形连续日期块 bootstrap；13 日期为主分析，26 日期为敏感性检查。", "",
        table(["参照方案", "平均 NCORR 差值", "13 日期块 95% 区间", "26 日期块 95% 区间"], bootstrap_rows), "",
        "这些区间仅描述已复用历史上的差值，未校正候选选择及此前的自适应开发。"
        "圆形块可能跨过年度重训边界并首尾相接，未建模结构变化；即使区间不含零，"
        "也不能称为经过选型校正的确认性显著性，更不能证明未来 live 胜出。"
        f"完整结果与限定条件见 {link(output / 'paired_comparison.json', output, '配对比较记录')}。", "",
        "## 最终训练与本轮 CSV", "",
        f"CV 入选配置已经单独重训。轮次 **{live['round']}** 的真实 live 数据日期为 **{date(live_date)}**；"
        f"严格训练边界为 `{date(final_cutoff)}`，要求日期 **小于** 该边界，因此排除边界当天，"
        f"训练池最近日期为 **{date(training_max)}**。", "",
        f"完整可用训练池 {int(final['full_training_rows']):,} 行；每成员实际使用 **{int(bundle['training_rows']):,}** 行、"
        f"**{len(bundle['features'])}** 个有序数值特征，模型种子为 **{', '.join(map(str, bundle['member_seeds']))}**。"
        f"本轮 CSV 包含 **{int(live['live_rows']):,}** 个唯一 ticker，"
        f"{int(live['unique_predictions']):,} 个不同预测值；最后再对集成值进行保留并列的百分位排名。", "",
        f"下载：{link(csv_path, output, '优化模型本轮 CSV')}；"
        f"{link(Path(run['final_bundle']), output, '最终模型 bundle')}；"
        f"{link(live_output / 'live_generation.json', output, '生成记录')}；"
        f"{link(live_output / 'validation.json', output, 'live 文件复核')}。", "",
        f"CSV SHA256：`{live['csv_sha256']}`。生成前后公开 API 的当前轮次一致，"
        "live 日期与该轮 dataDatestamp 一致。当前产物是待提交文件；生成流程没有上传账号、替换既有提交或执行质押。"
        "轮次及截止状态会变化，使用 CSV 前应再次确认适用轮次。", "",
        "离线 NCORR 使用官方 neutralizer；它不是网站的 live MMC/NMMC。"
        f"{link(output / 'validation_predictions.csv', output, 'validation_predictions.csv')} 是本次评估日期的历史预测导出，"
        f"覆盖 CV 的 {cv.date.nunique()} 个日期及开发期的 {dev.date.nunique()} 个日期，固定 embargo 间隔未预测。"
        "未验证官网是否接受这一日期子集，因此不将它称为可直接上传的完整 diagnostics 文件。"
        f"本轮待提交 CSV 则完整覆盖实际 live 文件的 {int(live['live_rows']):,} 个 ticker。", "",
        "## 验证与运行资源", "",
        f"训练、历史评分、图表与最终重训共耗时 **{elapsed / 60:.2f} 分钟**（{elapsed:.2f} 秒），"
        f"Python 进程峰值工作集约 **{peak_gib:.2f} GiB**，使用 {int(run['n_jobs'])} 线程。"
        "该范围不包含最初原始数据下载、独立复核和随后 live 公共文件下载。", "",
        f"独立历史复核覆盖全部 **{validation_dates}** 个日期、**{validation_rows:,}** 条保存预测记录、"
        f"**{summary_count}** 条汇总记录，并重算候选选择。全部模型的 CORR 经独立公式逐日期重算，"
        f"最大绝对误差为 `{max_corr_error:.3g}`；NCORR 在每个年度分区抽查一个完整日期，"
        f"共 {len(validation['folds'])} 个完整日期，最大误差为 `{max_ncorr_error:.3g}`。"
        "这不是第二遍重算所有日期的 NCORR。", "",
        "live 复核从交接目录重新加载模型成员，核对文件哈希、特征顺序、ticker 覆盖与行序；"
        "CSV 数值通过精确往返检查，并用 scipy 的独立平均排名实现复核并列名次。"
        f"模型重新加载的预测最大绝对误差为 `{float(live_validation['model_reload_prediction_max_abs_error']):.3g}`。", "",
        f"项目测试集 **{test_count} 项通过**，原始运行日志见 {link(tests_path, output, 'tests.log')}。", "",
        f"历史独立复核范围以 {link(output / 'validation.json', output, 'validation.json')} 为准；"
        f"冻结方案、输入文件及源代码哈希见 {link(output / 'plan.json', output, 'plan.json')} 和 "
        f"{link(output / 'run.json', output, 'run.json')}。这些检查验证计算与产物一致性，不证明新的 live 表现。", "",
        "## 复现", "",
        "从 `Project2` 目录执行。训练和 live 生成要求新的空输出目录；下面用 `_reproduction` 后缀保留本次材料。"
        "live 生成只接受运行时仍为当前轮次的 `--round`；轮次变化后应下载新 live 数据并按新日期重训。", "",
        "```powershell",
        f'.\\.venv\\Scripts\\python.exe scripts/run_model_optimization.py --data-root "{snapshot_arg}" --baseline "{baseline_arg}" --live-data "{live_data_arg}" --output "{output_arg}_reproduction" --n-jobs {run["n_jobs"]}',
        f'.\\.venv\\Scripts\\python.exe scripts/validate_model_optimization.py --root "{snapshot_arg}" --output "{output_arg}_reproduction" --n-jobs {run["n_jobs"]}',
        f'.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v 2>&1 | Out-File -Encoding utf8 "{output_arg}_reproduction/tests.log"',
        f'.\\.venv\\Scripts\\python.exe scripts/generate_optimized_live.py --bundle "{output_arg}_reproduction/final_model/{selected}/bundle.json" --output "{live_arg}_reproduction" --round {live["round"]} --n-jobs {run["n_jobs"]}',
        f'.\\.venv\\Scripts\\python.exe scripts/build_model_optimization_report.py --output "{output_arg}_reproduction" --live-output "{live_arg}_reproduction"',
        "```", "",
        "后续每周只需对已经冻结的入选方案重训，无需重新搜索候选或反复回看开发分数。"
        "先取得该轮完整 live.parquet 与包含成熟标签的最新历史快照，再执行以下模板命令；"
        "将尖括号内文字替换为实际路径或日期，输出目录须为新的空目录。", "",
        "```powershell",
        f'.\\.venv\\Scripts\\python.exe scripts/refit_optimized_model.py --data-root "<最新快照>" --selection "{output_arg}/selection.json" --live-data "<当轮完整live.parquet>" --output "results/optimized_refit_<新日期>" --n-jobs {run["n_jobs"]}',
        "```", "",
        "该入口沿用冻结的 winner、样本量、特征和种子，并按新 live 日期重新执行严格的 65 工作日成熟隔离。"
        "重训后再用生成脚本读取新 bundle 产生对应轮次的 CSV。", "",
        "官方口径：[Numerai v6 数据](https://docs.numer.ai/numerai-tournament/data)、"
        "[评分与诊断](https://docs.numer.ai/numerai-tournament/scoring)、"
        "[提交规则](https://docs.numer.ai/numerai-tournament/submissions)。", "",
        f"报告构建时间（UTC）：{timestamp}。", "",
    ]
    destination = output / "OPTIMIZATION_REPORT_ZH.md"
    temporary = destination.with_suffix(".md.part")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"status": "complete", "report": str(destination),
                      "selected_model": selected, "sha256": sha256(destination)}, ensure_ascii=False))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Completed and independently audited results directory")
    parser.add_argument("--live-output", type=Path, required=True, help="Completed live handoff directory")
    args = parser.parse_args()
    build(args.output, args.live_output)


if __name__ == "__main__":
    main()
