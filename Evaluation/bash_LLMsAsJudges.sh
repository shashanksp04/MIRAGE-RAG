#!/usr/bin/env bash
set -eo pipefail  # Exit on error, catch pipeline failures

############################################
# Configuration
BENCH_TYPES=("standard_without_db")
NUM_PROCESSES=8

################### Judge and Subject Models ###################
# Qwen/Qwen3-32B 0.6
# microsoft/Phi-4-reasoning 0.8
# deepseek-ai/DeepSeek-R1-Distill-Llama-70B 0.6
JUDGE_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
SUBJECT_NAME="Llama-3.2-11B-Vision-Instruct"

JUDGE_CLEAN="Llama-3.2-11B-Vision-Instruct"
SUBJECT_CLEAN="Llama-3.2-11B-Vision-Instruct"

TEMPERATURE=0.6
OPENAI_API_BASE="http://127.0.0.1:11434/v1"
JUDGE_CLEAN=$(echo "$JUDGE_NAME" | sed 's|.*/||')

INPUT_DIR="../Datasets/${BENCH_TYPES}"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/benchmark_run_${JUDGE_CLEAN}.log"
> "$LOG_FILE"

# Calculate total tasks: standard => ID+MG, contextual => MG only
TOTAL_TASKS=0
for bench in "${BENCH_TYPES[@]}"; do
  if [ "$bench" == "contextual" ]; then
    TOTAL_TASKS=$((TOTAL_TASKS + 1))  # only MG
  else
    TOTAL_TASKS=$((TOTAL_TASKS + 2))  # ID and MG
  fi
 done
COMPLETED_TASKS=0

# Helper functions
log() {
  local message="$1"
  local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
  echo -e "[$timestamp] $message" | tee -a "$LOG_FILE"
}

log_header() {
  local message="$1"
  local separator=$(printf '=%.0s' {1..80})
  log "\n$separator"
  log "$message"
  log "$separator"
}

update_progress() {
  COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
  local progress=$((COMPLETED_TASKS * 100 / TOTAL_TASKS))
  local elapsed=$(( $(date +%s) - START_TIME ))
  if [ $COMPLETED_TASKS -gt 0 ]; then
    local tpt=$(( elapsed / COMPLETED_TASKS ))
    local remaining=$(( (TOTAL_TASKS - COMPLETED_TASKS) * tpt ))
    local hr=$(( remaining / 3600 ))
    local mn=$(((remaining % 3600) / 60))
    log "Progress: $progress% ($COMPLETED_TASKS/$TOTAL_TASKS) | Elapsed: $(format_time $elapsed) | Remaining: ${hr}h ${mn}m"
  else
    log "Progress: $progress% ($COMPLETED_TASKS/$TOTAL_TASKS) | Elapsed: $(format_time $elapsed) | Remaining: calculating..."
  fi
}

format_time() {
  local s="$1"
  printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

run_benchmark() {
  local bench="$1"
  local type="$2"  # ID or MG
  local input_file="$3"
  local outdir="$4"

  log "Running: $bench ($type) | Subject: $SUBJECT_NAME | Judge: $JUDGE_CLEAN"
  mkdir -p "$outdir"

  local script_name="LLMsAsJudges_${type}.py"
  local output_file="$outdir/score_${SUBJECT_NAME}.json"

  if python "$script_name" \
       --input_file "$input_file" \
       --output_file "$output_file" \
       --model_name "$JUDGE_NAME" \
       --num_processes "$NUM_PROCESSES" \
       --subject_name "$SUBJECT_NAME" \
       --temperature "$TEMPERATURE" \
       --openai_api_base "$OPENAI_API_BASE" >> "$LOG_FILE" 2>&1; then
    log "✅ Completed: $bench ($type)"
  else
    log "❌ Failed:    $bench ($type)"
  fi
  update_progress
}

############################################
# Main execution
START_TIME=$(date +%s)
log_header "Benchmark Configuration"
log "Judge Model: $JUDGE_NAME"
log "Benchmarks: ${BENCH_TYPES[*]}"
log "Subject Model: $SUBJECT_NAME"
log "Processes: $NUM_PROCESSES"
log "Temperature: $TEMPERATURE"
log "Log File: $LOG_FILE"
log "Total Tasks: $TOTAL_TASKS"

# ID-based benchmarks (standard only)
log_header "Starting ID-based scoring"
for BENCH in "${BENCH_TYPES[@]}"; do
  if [ "$BENCH" == "contextual" ]; then
    continue
  fi
  ID_INPUT_PATH="$INPUT_DIR/${BENCH}_benchmark_ID_${SUBJECT_NAME}.json"
  # ID_INPUT_PATH="$INPUT_DIR/${BENCH}_benchmark.json"
  OUTDIR="./results/${BENCH}_ID_score/${JUDGE_CLEAN}"
  run_benchmark "$BENCH" ID "$ID_INPUT_PATH" "$OUTDIR"
done

# MG-based benchmarks (all types)
log_header "Starting MG-based scoring"
for BENCH in "${BENCH_TYPES[@]}"; do
  MG_INPUT_PATH="$INPUT_DIR/${BENCH}_benchmark_MG_${SUBJECT_NAME}.json"
  # MG_INPUT_PATH="$INPUT_DIR/${BENCH}_benchmark.json"
  OUTDIR="./results/${BENCH}_MG_score/${JUDGE_CLEAN}"
  run_benchmark "$BENCH" MG "$MG_INPUT_PATH" "$OUTDIR"
done

# Completion summary
TOTAL_RUNTIME=$(( $(date +%s) - START_TIME ))
log_header "Benchmark Completed"
log "Total Runtime: $(format_time $TOTAL_RUNTIME)"
log "Results saved in ./results/"
log "Log saved to $LOG_FILE"