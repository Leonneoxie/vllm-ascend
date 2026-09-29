#!/bin/bash
# 通用 vllm serve 启动脚本
# 用法: ./vllm_serve.sh <model_path> <card_id> <port> <config_json|bf16> [eager|decode_graph|--enforce-eager]
# 示例: ./vllm_serve.sh /path/to/view 0 8001 configs/qwen3_5_9b_rtn_attn-only_w4a4.json
#       ./vllm_serve.sh /path/to/model 0 8001 bf16          # BF16 对照（不量化，可直接用原模型目录）
set -e

MODEL_PATH=$1
CARD=$2
PORT=$3
CONFIG=$4
MODE=${5:-${EXECUTION_MODE:-eager}}
case "$MODE" in
  eager|--enforce-eager) EXECUTION=(--enforce-eager) ;;
  decode_graph) EXECUTION=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}') ;;
  *) echo "mode must be eager or decode_graph" >&2; exit 2 ;;
esac

# 按脚本自身位置定位仓库根目录，并置于 PYTHONPATH 首位（确保运行本仓代码，
# 而非环境里已安装的其他 vllm_ascend）
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$REPO_ROOT" ]; then
    REPO_ROOT=$SCRIPT_DIR
    while [ "$REPO_ROOT" != "/" ] && [ ! -d "$REPO_ROOT/vllm_ascend" ]; do
        REPO_ROOT=$(dirname "$REPO_ROOT")
    done
fi
export PYTHONPATH=$REPO_ROOT:${PYTHONPATH:-}
PYTHON=${PYTHON:-python3}
export ASCEND_RT_VISIBLE_DEVICES=$CARD

# 启动即打印运行身份（必须核验：导入路径在 $REPO_ROOT 下）
echo "fake-MX repo root:    $REPO_ROOT"
echo "fake-MX commit:       $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
"$PYTHON" - "$REPO_ROOT" <<'PY'
import pathlib
import sys
import vllm
import vllm_ascend
root = pathlib.Path(sys.argv[1]).resolve()
actual = pathlib.Path(vllm_ascend.__file__).resolve()
if not actual.is_relative_to(root / "vllm_ascend"):
    raise SystemExit(f"Wrong vllm_ascend import: {actual}, expected {root}")
print(f"vllm_ascend from:     {actual}", flush=True)
print(f"Python: {sys.executable}; vLLM: {vllm.__version__} from {vllm.__file__}", flush=True)
PY

# 量化场景：注入配置并启用 ascend 量化；bf16 对照：两者均跳过（可直接传原模型目录）
QUANT_FLAGS=(--quantization ascend)
if [ "$CONFIG" = "bf16" ]; then
    QUANT_FLAGS=()
else
    cp "$CONFIG" "$MODEL_PATH/quant_model_description.json"
fi

printf 'fake-MX execution mode: %s\n' "$MODE"
exec "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size 1 --served-model-name qwen3.5 \
    --max-num-seqs 32 --max-model-len 16384 \
    --trust-remote-code --gpu-memory-utilization 0.90 \
    --mamba-ssm-cache-dtype bfloat16 --dtype bfloat16 \
    "${QUANT_FLAGS[@]}" "${EXECUTION[@]}"
