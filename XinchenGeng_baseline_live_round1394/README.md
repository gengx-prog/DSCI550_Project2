# Xinchen Geng baseline live predictions — round 1394

## 交接内容

直接交给负责 Numerai 提交的队友：`XinchenGeng_baseline_live_round1394.csv`。

- 当前生成轮次：**1394**；数据版本：**v6.0**；共 **7,121** 条预测。
- CSV 两列为 `numerai_ticker,prediction`，覆盖当轮全部 live 标识，无重复或缺失；预测值在 `[0,1]`。
- 复用原有 `B0_small_main` 已训练模型，没有重新训练或改参数。该模型与 10 月 30 日、11 月 6 日归档中的模型哈希一致。
- `B0_small_main.txt` 是 LightGBM 原生模型；两份 Python 脚本分别用于生成和独立复核。
- `validation.json` 记录全量独立推理与平均秩校验，以及校验时公开 API 的当前轮次。
- 本包仅准备交接材料，**没有向 Numerai 账号提交，也没有提交回执**。

此 CSV 对应第 1394 轮。提交前核对官网当前轮次；轮次改变后用脚本重新生成。准确的生成时间、当轮数据日期及 API 返回的各类截止时间保存在 `live_generation.json`。`closeTime` 和 `closeStakingTime` 是不同字段。

## Files

| File | Purpose |
| --- | --- |
| `XinchenGeng_baseline_live_round1394.csv` | The current-round prediction file to pass to the submission owner |
| `B0_small_main.txt` | Original saved 100-feature, 300-tree LightGBM baseline |
| `generate_xinchen_live.py` | Public round-pinned download and live prediction generation |
| `validate_xinchen_live.py` | Independent complete model/rank/identity validation |
| `baseline_run.json`, `baseline_protocol.json` | Original historical run, ordered features and model identity |
| `features.json`, `live_data_manifest.json`, `live_generation.json` | Official metadata, full download hashes and generation provenance |
| `audit_data/live.parquet` | Exact original public live data, included for offline reproduction of this audit |
| `validation.json` | Full audit and current-round check performed at handoff preparation |
| `HANDOFF_MANIFEST.json` | SHA256 and byte counts for all delivered files |

The historical baseline source run completed November 5 in America/Los_Angeles / November 6 UTC. The live CSV was generated November 7 in America/Los_Angeles. These are different stages, and their original timestamps remain in the records. `dataDatestamp=20261106` is the API's data version stamp for round 1394, not the date of this prediction run.

## Install

The recorded environment uses Python 3.13.7. From the extracted handoff directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS/Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Independently check this CSV

```powershell
.\.venv\Scripts\python.exe validate_xinchen_live.py --bundle . --live audit_data/live.parquet --report validation_recheck.json
```

For an additional public API current-round check, add `--check-current-round`. This will fail when round 1394 is no longer current. Choose a new report name on each run; the original audit is preserved. The audit recomputes every saved-model prediction using the exact feature order, then independently computes average ranks using SciPy and compares the complete ticker coverage and CSV values. Numeric feature NaNs follow the model's native missing-value handling.

## Generate the next current round without retraining

```powershell
.\.venv\Scripts\python.exe generate_xinchen_live.py --baseline . --output ../baseline_live_next --data-dir ../baseline_live_next_data
```

Use fresh directories on each run. An optional `--round NUMBER` asserts the expected current round and fails if it has changed. Generation pins both `features.json` and `live.parquet` to that same round, checks the model's feature names/order against v6 small numeric metadata, and checks the current round again before finalizing the CSV. The output CSV uses official average percentile ranks, with maximum prediction allowed to equal 1. No account credentials, model ID, upload call, or staking action is used by either script.

Official format: https://docs.numer.ai/numerai-tournament/submissions
