# Qwen3.5-9B 部署与测评指南

本文介绍 Qwen3.5-9B 的部署与数据集测评，以 BF16 对照和 FlatQuant MLP W4A4 为例。
两种场景共用 MATH-500、MMLU-Pro、LiveCodeBench 测评步骤；其他算法见文末指导链接。

fake-MX 使用浮点计算模拟量化：权重在加载期变换并量化/反量化，激活在推理期处理。
本流程验证数值精度，不代表原生低比特推理性能。

## 环境准备（一次性）

执行拓扑：**vllm serve 在 vllm-ascend 容器内运行，三数据集评测在宿主机运行**，
通过 host 网络端口通信。

**1. 下载权重**（`--local_dir` 即后续配置中 `MODEL_DIR` 的值）：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3.5-9B --local_dir /your/model/dir/Qwen3.5-9B
```

**2. 拉取官方镜像**（内置 CANN / torch_npu / vLLM-Ascend）：

```bash
docker pull quay.io/ascend/vllm-ascend:v0.23.0rc1-a3-openeuler
```

**3. 宿主机评测依赖**（评测在宿主机执行，不在容器内）。在独立Python环境中安装与归档测评脚本一致的版本：

```bash
python3 -m pip install evalscope==1.10.0 pyarrow==19.0.1 ms-enclave==0.0.8
python3 -m pip check
docker pull python:3.11-slim   # LCB 沙箱镜像，评测经宿主机 Docker 拉起沙箱
```

后续测评使用同一环境的 `python3`，当前用户需能访问 Docker。

**4. 数据集**：默认联网自 ModelScope 下载（用法详见
[EvalScope 文档](https://evalscope.readthedocs.io)）；离线或固定快照环境设置
`MODELSCOPE_CACHE`（见 [Using EvalScope](../../../../docs/source/developer_guide/evaluation/using_evalscope.md)）。

**5. 写环境配置**。先创建个人目录 `/data/USER`，将下面内容保存为该目录下的 `retest.env`，替换 USER 和实际路径。
这份配置只包含公共环境；输出目录在下面各场景中分别指定。

共享目录需允许容器内和宿主机执行用户读写；尽量使用一致的UID/GID。
若容器使用root，需由目录所有者或管理员将本次工作目录、输出目录的权限配置给宿主机执行用户，
再运行宿主机命令。不要全局放宽权限或关闭Git目录所有权检查。

```bash
export MODEL_DIR=/your/model/dir/Qwen3.5-9B
export WORKSPACE=/data/USER/vllm-workspace
export REPO="$WORKSPACE/vllm-ascend"
export CARD=0
export PORT=8000
export E="$REPO/examples/quantization/fake_mx/w4a4_eval"
# export MODELSCOPE_CACHE=/your/dataset-cache   # 离线时指定
```

| 变量 | 说明 |
|---|---|
| `MODEL_DIR` | BF16 权重目录（环境准备第 1 步的下载位置） |
| `REPO` | 本仓根目录；serve 从这里加载 fake-mx 分支代码，启动脚本会校验实际导入路径 |
| `CARD` / `PORT` | NPU 卡号 / 服务端口（host 网络，宿主机评测直连） |
| `MODELSCOPE_CACHE` | 可选：离线或固定数据集快照 |

**6. 启动容器**（宿主机执行；已有满足以下挂载和网络条件的容器可复用）。serve 在容器内运行，需要读取**本仓代码、模型权重**，
并向**输出目录**写日志——以下挂载让个人工作目录、配置及模型在容器内外路径一致，
使容器内外用同一份 `retest.env`：

```bash
source /data/USER/retest.env
docker run -itd --name vllm-ascend \
  --device /dev/davinci"$CARD" --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v "$MODEL_DIR:$MODEL_DIR" -v /data/USER:/data/USER \
  --network host \
  quay.io/ascend/vllm-ascend:v0.23.0rc1-a3-openeuler bash
```

**7. 准备代码**。执行 `docker exec -it vllm-ascend bash` 进入容器，首次准备个人工作副本（目标目录尚不存在时执行）：

```bash
cp -a /vllm-workspace /data/USER/vllm-workspace
```

随后在容器内检查工作区。若有已有修改，先保留并处理，不使用强制覆盖：

```bash
cd /data/USER/vllm-workspace/vllm-ascend
git status --short
```

确认后配置测评仓库远端并切换代码（需先安装Git LFS）：

```bash
git remote add benchmark https://github.com/Leonneoxie/vllm-ascend.git
git lfs install
git fetch benchmark fake-mx-lite
GIT_LFS_SKIP_SMUDGE=1 git switch --detach refs/remotes/benchmark/fake-mx-lite
git log -1 --format='%H %s'
```

`benchmark`只需添加一次；已有同名远端时先用 `git remote get-url benchmark` 确认地址，再跳过添加。
代码和后续LFS下载均使用此远端，不依赖镜像的 `origin`。切换时暂不下载参数，FlatQuant步骤再显式获取。
以detached HEAD固定本次提交；复现指定版本时，将分支引用换成已获取的完整提交号，并记录最后一条命令输出。

也可下载指定版本源码复制覆盖。保留配套 `vllm` 目录；已有个人副本时复用，不重复复制。
两个终端均使用个人副本中的脚本。

## BF16 部署

BF16不需要sidecar或模型视图，直接使用原始BF16模型目录。
该目录应保持原始状态，不含额外注入的 `quant_model_description.json`。

终端 A：执行 `docker exec -it vllm-ascend bash` 进入容器，确认端口空闲后启动：

```bash
source /data/USER/retest.env
export OUT=/data/USER/results/bf16-run01
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT" || exit 1
export PYTHONPATH="$REPO:$WORKSPACE/vllm:${PYTHONPATH:-}"
set -o pipefail
PYTHON="$(command -v python3)" bash "$E/scripts/serve/vllm_serve.sh" \
  "$MODEL_DIR" "$CARD" "$PORT" bf16 eager 2>&1 | tee "$OUT/serve.log"
```

第四个参数 `bf16` 让脚本跳过量化配置注入及 `--quantization ascend`。
服务启动后，按下方“数据集测评”执行，终端 B 使用相同的BF16输出目录。

## FlatQuant MLP 部署

使用原始BF16权重和归档sidecar，不需要重新训练AMCT。
先停止上一场景服务，再准备参数和独立模型视图，原模型保持不变。

**1. 下载参数并准备视图**（宿主机，需 Git LFS）：

```bash
source /data/USER/retest.env
export OUT=/data/USER/results/flatquant-mlp-run01
git -C "$REPO" lfs install
git -C "$REPO" lfs pull benchmark --include='examples/quantization/fake_mx/w4a4_eval/params/**/*.safetensors'
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT" && mkdir "$OUT/model_view" || exit 1
cp -as "$MODEL_DIR/." "$OUT/model_view/"
rm -f "$OUT/model_view/quant_model_description.json" "$OUT/model_view/flatquant_params.safetensors"
cp "$E/params/flatquant/qwen3_5_9b_flatquant_mlp-only_w4a4.safetensors" "$OUT/model_view/flatquant_params.safetensors"
echo "6d47223d8431f525d0bbcbf9e4588ee39a8b482bc91607aef27ca845abd8bb6e  $OUT/model_view/flatquant_params.safetensors" | sha256sum -c - || exit 1
```

这里的 `cp -as` 只建立软链接；`rm -f` 仅移除新视图中的两个链接，不删除原模型文件。

**2. 启动服务**（终端 A，容器内）：

```bash
source /data/USER/retest.env
export OUT=/data/USER/results/flatquant-mlp-run01
export PYTHONPATH="$REPO:$WORKSPACE/vllm:${PYTHONPATH:-}"
set -o pipefail
PYTHON="$(command -v python3)" bash "$E/scripts/serve/vllm_serve.sh" \
  "$OUT/model_view" "$CARD" "$PORT" \
  "$E/configs/qwen3_5_9b_flatquant_mlp-only_w4a4.json" eager 2>&1 | tee "$OUT/serve.log"
```

启动脚本将配置注入模型视图，并在加载期应用归档变换参数。
服务启动后，按下方测评步骤执行，终端 B 使用FlatQuant MLP输出目录。

## 数据集测评

两个场景均使用TP1、BF16计算、eager、显存利用率0.90、max-num-seqs=32、
max-model-len=16384。同一服务串行运行三个数据集，无需反复重启。

**1. 核验服务**（终端 B：宿主机）：

```bash
source /data/USER/retest.env
export OUT=/data/USER/results/bf16-run01  # 测FlatQuant时改为flatquant-mlp-run01
curl -fsS --max-time 3 "http://localhost:$PORT/v1/models"
grep -E 'vllm_ascend from:|Loaded 288' "$OUT/serve.log"
```

BF16不应出现参数加载记录；FlatQuant MLP应出现288个参数加载记录。
`vllm_serve.sh` 对导入路径不符会自动中止。全量前先执行一次短生成：

```bash
curl -fsS --max-time 120 "http://localhost:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"1+1=? Only output the answer."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}' \
  -o "$OUT/smoke.json" && python3 -m json.tool "$OUT/smoke.json"
```

确认响应包含有效的 `choices[0].message.content`、答案正常且没有报错，再开始全量测评。
需要更多检查时可选用105题快检（约11分钟）：
`python3 "$E/scripts/eval/run_level3_verify.py" "$PORT" "$OUT/level3"`

**2. 三数据集**（终端 B：宿主机）。等待服务就绪后执行；耗时参考 [基线矩阵](baseline-results.md)。

```bash
python3 "$E/scripts/eval/run_math500.py" "$PORT" "$OUT/math500" &&
python3 "$E/scripts/eval/run_mmlu_pro.py" "$PORT" "$OUT/mmlu_pro" &&
python3 "$E/scripts/eval/run_livecodebench.py" "$PORT" "$OUT/livecodebench"
```

生成参数已内置（temp 1.0 / top_k 20 / top_p 0.95 / do_sample / seed 42 /
max_tokens 8192 / thinking off / 请求超时600s）。MATH-500和LCB使用0-shot，MMLU-Pro使用5-shot。

**3. 看结果**（终端 B）：

```bash
python3 -m json.tool "$OUT"/math500/*/reports/qwen3.5/*.json | grep -E '"score"|"num"'
python3 -m json.tool "$OUT"/mmlu_pro/*/reports/qwen3.5/*.json | grep -E '"score"|"num"'
python3 -m json.tool "$OUT"/livecodebench/*/reports/qwen3.5/*.json | grep -E '"score"|"num"'
```

LCB 数据快照核对（`release_latest` 会随时间加题，题数相同不代表同一批题）：

```bash
python3 "$E/scripts/eval/lcb_input_digest.py" "$OUT/livecodebench"
```

FlatQuant MLP 归档基线摘要：`86c94a144b50a5c6`；不一致时检查输入集合与记录完整性。摘要只覆盖题目输入，不校验隐藏测试。

**收尾**：终端 A Ctrl+C 停 serve。更换场景时使用新OUT目录，复用容器并按对应部署章节启动。

## 结果对照

两种场景均对照 [baseline-results.md](baseline-results.md)，分数只在该文件维护。
超过2个百分点需复核；阈值以内也需确认样本完整、无评测异常，不能只凭分差认定通过。
保留本次retest.env副本、代码版本、serve.log、smoke.json及完整数据集输出（配置、逐题结果、报告）。
PPL由AMCT侧单独执行，不属于本流程的vLLM数据集测评。

## 其他算法

- RTN/RHT：无需训练参数，选择对应配置启动；参见 [README：使用步骤](README.md#使用步骤)。
- LHT、OmniQuant及其他FlatQuant场景：准备对应sidecar和配置，训练与转换方法见
  [README](README.md#使用步骤)。当前仓库仅归档FlatQuant MLP参数，其他参数需自行准备。
- 部署后均可复用本页的三数据集测评步骤；小样本验证见 [快速验证指导](quick-verify-guide.md)。
- 新算法代码实现、注册及测试见 [fake-MX README](../README.md)；学习型算法还需扩展训练和转换工具。
  参数登记到 [params/README.md](params/README.md)，最终分数统一登记到 [baseline-results.md](baseline-results.md)。
