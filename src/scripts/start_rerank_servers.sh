#!/bin/bash
# Start N resident Qwen3-Reranker servers, ONE per GPU (data-parallel rerank pool).
# Each loads the 4B model ONCE and stays up; point eval at them with the printed
# comma-list:  --rerank_remote_url "<URLS>"  (or set rerank_remote_url in the yaml).
#
#   bash scripts/start_rerank_servers.sh "1,2,3" 8200 qwen3_reranker  # GPUs->ports, reranker type
#   bash scripts/start_rerank_servers.sh "1,2,3" 8200 monot5_3b       # any HF reranker type
#   pkill -f apcir.search.rerank_server                               # stop them all
#
# Run from src/. Uses the trec_ikat py3.12 env. Avoid GPUs others are using (nvidia-smi).
set -e
GPUS=${1:-1,2,3}
BASE=${2:-8200}
RTYPE=${3:-qwen3_reranker}
ENVBIN=/data/rech/huiyuche/envs/trec_ikat/bin
LOGDIR=/part/01/Tmp/yuchen/rerank_servers
mkdir -p "$LOGDIR"

IFS=',' read -ra G <<< "$GPUS"
urls=""
i=0
for g in "${G[@]}"; do
  port=$((BASE + i))
  CUDA_VISIBLE_DEVICES=$g HUGGINGFACE_HUB_CACHE=/data/rech/huiyuche/huggingface \
    nohup "$ENVBIN/python" -m apcir.search.rerank_server --reranker_type "$RTYPE" --gpu_id 0 --port "$port" \
    > "$LOGDIR/server_gpu${g}_port${port}.log" 2>&1 &
  echo "  GPU $g -> http://127.0.0.1:$port  (pid $!, log $LOGDIR/server_gpu${g}_port${port}.log)"
  urls="$urls,http://127.0.0.1:$port"
  i=$((i + 1))
done
urls="${urls#,}"

echo "  waiting for all servers healthy (model load ~30-60s each, parallel)..."
for u in ${urls//,/ }; do
  until curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$u/health" 2>/dev/null | grep -q 200; do
    sleep 5
  done
  echo "  ready: $u"
done
echo "RERANK_URLS=$urls"
