# 第 1399 轮优化候选交接

模型为 `O3_regularized_1200k_ensemble`：350 个特征，每成员 120 万训练样本，三个 LightGBM 成员的种子为 7、19、43；每成员 800 棵树，学习率 0.03、L2 10。所有参数、特征顺序及成员哈希保存在 [模型 bundle](model/bundle.json)。

[Optimized_live_round1399.csv](Optimized_live_round1399.csv) 覆盖该轮全部 7,106 个 ticker，列为 `numerai_ticker,prediction`。实际 live 日期是 2026-11-13；训练日期严格早于 2026-08-14，训练池最近日期为 2026-08-07。该文件仅适用于第 1399 轮，生成不代表已上传或准时提交。

四年度 CV 选出了该方案。已观察历史开发期 NCORR 均值为 0.027082，本地固定 B0 为 0.010132，同协议 60 万行集成为 0.026297。相比后者，均值约提高 3%，但稳定性指标及 2025 年表现回退，配对差值区间包含零。因此交付为待后续新轮次验证的候选，尚未证明是原强化集成的稳定替代。

90 项项目测试通过。全部历史日期的 CORR、各年度抽样日期的 NCORR、模型哈希和抽样记录通过独立复核；本目录 CSV 经过模型重新加载、scipy 独立排名、精确 CSV 往返及 ticker 覆盖核对，误差为零。见 [live 复核](validation.json)、[生成记录](live_generation.json) 和 [完整优化报告](../../results/model_optimization_20261116_round1399/OPTIMIZATION_REPORT_ZH.md)。

交接 zip 保留 `Project2` 目录结构，包含最终模型、public live 数据、CSV、历史预测导出、评分及复核记录、脚本和测试。多 GB 历史原始数据及各年度大体积模型仍保存在原工作区；重新运行完整历史复核前，应按报告下载数据并重跑对照。部分运行记录保留原机器的绝对路径，换机器时通过 CLI 显式提供新路径。

从 `Project2` 目录运行时，依赖见 `requirements.txt`。后续模型更新使用 `scripts/refit_optimized_model.py` 的冻结选择入口，然后用 `scripts/generate_optimized_live.py` 下载并生成对应当前轮次 CSV。完整参数和新目录要求见优化报告。
