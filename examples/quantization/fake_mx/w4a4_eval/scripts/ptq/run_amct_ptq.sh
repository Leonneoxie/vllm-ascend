#!/bin/bash
# AMCT PTQ 多卡并行训练脚本（统一训练口径）
# 用法: AMCT_ROOT=/path/to/amct ./run_amct_ptq.sh <algo> <quant_target> <model_path> <data_dir> <output_dir> [bit_config]
#   algo:         flatquant | omniquant | learnable_had
#   quant_target: attn-linear | mlp
#   bit_config:   可选，默认 $AMCT_ROOT/amct_pytorch/configs/w4a4.yaml
# 统一训练口径: epochs 15 / base_lr 1e-3 / cali_bsz 4 / nsamples 128 / k_size 128 / adamw / cosine
# AMCT 运行环境（CANN env、依赖安装）见 AMCT 仓自带说明。
# 可选环境变量: NPU_TASKS（默认 8）、NPU_CARDS（默认 "0 1 2 3 4 5 6 7"，空格分隔）
set -e

ALGO=$1
TARGET=$2
MODEL=$3
DATA_DIR=$4
OUTPUT_DIR=$5

: "${AMCT_ROOT:?需要设置 AMCT_ROOT 指向 AMCT 仓根目录}"
BIT_CONFIG=${6:-$AMCT_ROOT/amct_pytorch/configs/w4a4.yaml}
[ -f "$BIT_CONFIG" ] || { echo "ERROR: bit_config 不存在: $BIT_CONFIG" >&2; exit 2; }

export PYTHONPATH=$AMCT_ROOT:${PYTHONPATH:-}
python -c "import amct_pytorch" >/dev/null 2>&1 || { echo "ERROR: amct_pytorch 不可导入（检查 AMCT_ROOT 与 AMCT 环境说明）" >&2; exit 2; }

NUM_BLOCKS=32
NUM_TASKS=${NPU_TASKS:-8}
CARDS_STR=${NPU_CARDS:-"0 1 2 3 4 5 6 7"}
mkdir -p "$OUTPUT_DIR/logs"

avg=$((NUM_BLOCKS / NUM_TASKS))
rem=$((NUM_BLOCKS % NUM_TASKS))
start=0
read -ra CARDS <<< "$CARDS_STR"
PIDS=""

for ((i=0; i<NUM_TASKS; i++)); do
  [ $i -lt $rem ] && len=$((avg+1)) || len=$avg
  end=$((start+len))
  npu=${CARDS[$i]}
  echo "Task $i: blocks [$start,$end) npu=$npu"
  ASCEND_RT_VISIBLE_DEVICES=$npu python -m amct_pytorch.ptq \
    --model "$MODEL" --model_name qwen3_5 --data_dir "$DATA_DIR" \
    --device npu:0 --granularity block \
    --start_block_idx $start --end_block_idx $end \
    --quant_target "$TARGET" --quant_dtype mxfp --bit_config "$BIT_CONFIG" \
    --algos "$ALGO" --output_dir "$OUTPUT_DIR" \
    --epochs 15 --base_lr 1e-3 --cali_bsz 4 --nsamples 128 \
    --k_size 128 --optimizer adamw --lr_scheduler cosine \
    > "$OUTPUT_DIR/logs/ptq_${ALGO}_${TARGET}_card${npu}.log" 2>&1 &
  PIDS="$PIDS $!"
  start=$end
  [ $i -lt $((NUM_TASKS-1)) ] && sleep 15
done

FAIL=0
for p in $PIDS; do
  wait "$p" || FAIL=1
done

# 参数完整性校验：layer_*.pt 必须覆盖全部 NUM_BLOCKS 层
PARAM_DIR=$(find "$OUTPUT_DIR" -type d -name "$TARGET" -path "*ptq_params*" | head -1)
GOT=$(ls "$PARAM_DIR"/layer_*.pt 2>/dev/null | wc -l)
if [ "$FAIL" -ne 0 ] || [ -z "$PARAM_DIR" ] || [ "$GOT" -ne "$NUM_BLOCKS" ]; then
  echo "FAILED: param files ${GOT:-0}/$NUM_BLOCKS in ${PARAM_DIR:-<none>} (see $OUTPUT_DIR/logs)" >&2
  exit 1
fi
echo "PTQ DONE: $GOT/$NUM_BLOCKS param files -> $PARAM_DIR"
