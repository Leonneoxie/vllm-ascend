# W4A4 基线复测与新算法接入

本目录通过 `params/` 归档参数复现基线，并支撑新算法接入。当前已归档 **FlatQuant MLP W4A4**，
以下流程以其为例；其他场景参数归档后按同样方式接入（见第五节）。

## 验收标准

复现结果与下表对比，**分数落在 ±2pt 内即通过**（采样解码，不要求逐位一致），
且各 report 的样本数必须与下表一致：

| 指标 | 样本数 | 基线分数 |
|---|---:|---:|
| MATH-500（0-shot） | 500 | 0.934 |
| MMLU-Pro（5-shot） | 12032 | 0.7843 |
| LiveCodeBench（0-shot） | 1055 | 0.5877 |

本表是归档参数的复测基线；[baseline-results.md](baseline-results.md) 是历史矩阵（另一批次），两者不混用。
PPL（8.2619）属 AMCT 侧流程，本指南不涉及。

## 环境

- Linux/Ascend + 匹配的 CANN / torch_npu / vLLM 0.23.0 系，EvalScope **1.10.0**
- 本仓 `fake-mx-lite` 分支，custom ops 已构建（`vllm_ascend/_cann_ops_custom`）
- LCB 需要 Docker 与 `python:3.11-slim` 镜像
- 服务与生成口径（启动/评测脚本已内置，无需手动传参）：TP1 / BF16 / eager /
  util 0.90 / max-num-seqs 32 / max-model-len 16384；temp=1.0 / top_k=20 /
  top_p=0.95 / do_sample / seed=42 / max_tokens=8192 / thinking=false

## 1. 取参数

```bash
git lfs install
git lfs pull --include='examples/quantization/fake_mx/w4a4_eval/params/**/*.safetensors'
```

## 2. 写配置（一次性，保存为绝对路径的 retest.env，改前六项）

```bash
export MODEL_DIR=/absolute/path/to/Qwen3.5-9B
export REPO=/absolute/path/to/vllm-ascend
export OUT=/absolute/path/to/new-retest-output   # 全新目录
export CARD=0
export PORT=8000
export ASCEND_CUSTOM_OPP_PATH=/absolute/path/to/custom_transformer
export E="$REPO/examples/quantization/fake_mx/w4a4_eval"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
# export MODELSCOPE_CACHE=/absolute/path/to/dataset-cache   # 离线时指定数据缓存
```

## 3. 一键复现（推荐）

```bash
source /absolute/path/to/retest.env
python3 "$E/scripts/eval/reproduce_flatquant_mlp.py" \
  --model "$MODEL_DIR" --out "$OUT" --card "$CARD" --port "$PORT"
```

入口依次完成：参数 SHA 校验 → 端口检查 → 独立模型视图 → serve（独立进程组）→
就绪与 288 参数检查 → smoke → MATH → MMLU → LCB → 清理本次服务，证据全程落 `$OUT`。
长任务建议放 tmux，中途退出用 Ctrl+C。参考耗时：MMLU 约 11.5h、LCB 约 9h；
只跑 MATH 可加 `--datasets math500`。

## 4. 查结果并对照验收

```bash
source /absolute/path/to/retest.env
python3 - "$OUT" <<'PY'
import json, pathlib, sys
state = json.loads((pathlib.Path(sys.argv[1]) / "status.json").read_text())
print("status:", state["status"])
for name, item in state["datasets"].items():
    print(name, "score:", item["score"], "num:", item["num"], "within_2pt:", item["within_2pt"])
PY
```

对照验收标准表；`*-error-candidates.json` 非空时先排查请求/沙箱失败再判定。

LCB 数据快照核对（digest 应等于基线值，不同则数据集版本已变化，需重新核对基准）：

```bash
python3 - "$OUT/livecodebench" <<'PY'
import json, glob, hashlib, sys
rows = []
for f in sorted(glob.glob(f"{sys.argv[1]}/*/reviews/qwen3.5/*.jsonl")):
    for line in open(f, encoding="utf-8"):
        d = json.loads(line)
        msgs = d.get("messages") or []
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        q = msgs[0].get("content", "") if msgs else ""
        rows.append(f"{d.get('index','')}\n{q}")
print("digest=" + hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()[:16])
PY
```

基线摘要：`86c94a144b50a5c6`

## 分步执行（调试或单独补测；勿与一键流程共用同一 OUT / 端口）

```bash
# 步骤 1：构建模型视图（source retest.env 后执行；OUT 须不存在）
source /absolute/path/to/retest.env
python3 - <<'PY'
import hashlib, os, pathlib, shutil
e = pathlib.Path(os.environ['E'])
model = pathlib.Path(os.environ['MODEL_DIR']).resolve()
out = pathlib.Path(os.environ['OUT'])
param = e / 'params/flatquant/qwen3_5_9b_flatquant_mlp-only_w4a4.safetensors'
assert hashlib.sha256(param.read_bytes()).hexdigest() == '6d47223d8431f525d0bbcbf9e4588ee39a8b482bc91607aef27ca845abd8bb6e'
out.mkdir(parents=True, exist_ok=False)
view = out / 'model_view'
view.mkdir()
for p in model.iterdir():
    if p.name not in {'quant_model_description.json', 'flatquant_params.safetensors'}:
        (view / p.name).symlink_to(p, target_is_directory=p.is_dir())
shutil.copy2(param, view / 'flatquant_params.safetensors')
shutil.copy2(e / 'configs/qwen3_5_9b_flatquant_mlp-only_w4a4.json', view / 'quant_model_description.json')
PY

# 步骤 2：终端 A 前台 serve（Ctrl+C 停止；传视图目录，脚本会向其写入量化配置）
source /absolute/path/to/retest.env
PYTHON="$(command -v python3)" bash "$E/scripts/serve/vllm_serve.sh" \
  "$OUT/model_view" "$CARD" "$PORT" \
  "$E/configs/qwen3_5_9b_flatquant_mlp-only_w4a4.json" eager > "$OUT/serve.log" 2>&1

# 步骤 3：终端 B 核验（须看到 vllm_ascend from: <本仓路径> 与 Loaded 288 fake-MX transform params）
source /absolute/path/to/retest.env
curl -fsS --max-time 3 "http://localhost:$PORT/v1/models"
grep -E 'fake-MX commit:|vllm_ascend from:|Loaded 288' "$OUT/serve.log"
# 可选快检：python3 "$E/scripts/eval/run_level3_verify.py" "$PORT" "$OUT/level3"（105 题，参考 0.9905）

# 步骤 4：串行三数据集
python3 "$E/scripts/eval/run_math500.py" "$PORT" "$OUT/math500" &&
python3 "$E/scripts/eval/run_mmlu_pro.py" "$PORT" "$OUT/mmlu_pro" &&
python3 "$E/scripts/eval/run_livecodebench.py" "$PORT" "$OUT/livecodebench"
```

## 5. 新算法接入

1. **配置**：`configs/` 新增 `qwen3_5_9b_{algo}_{scope}_{precision}.json`（字段见 [README.md](README.md)，sidecar 路径用视图内文件名）
2. **训练**（学习型算法需要；RTN/RHT 无需）：
   `AMCT_ROOT=/path/to/amct scripts/ptq/run_amct_ptq.sh <algo> <target> <model> <data_dir> <output_dir>`
   （统一口径 epochs 15 / lr 1e-3 / bsz 4 / nsamples 128 / k_size 128 / adamw / cosine；AMCT 环境见其仓说明）
3. **转换**：`scripts/convert/convert_ptq_to_vllm.py --algo <algo> --target <target> --ptq_dir <PTQ 产物> --model_dir <模型> --output <sidecar>`
4. **测评**：按上文流程 serve + 三数据集
5. **归档**：sidecar 入 `params/<algo>/`，更新 [params/README.md](params/README.md)（SHA 与结构），在本指南验收标准表登记该场景基线分数

重训参数是随机收敛解，与既有基线不可比：新算法/新批次须先建立自己的基线分数再归档。
