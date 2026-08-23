#!/usr/bin/env bash
# Build + run the NIM container for ShopFloor-Nemotron on an A100.
# Used as the fallback when Jetson is unreachable at the booth, and as
# the default for remote demos.
#
# Refs:
#   - docs.nvidia.com/nim/large-language-models/latest/getting-started.html
#   - https://build.nvidia.com  (Brev "Launch on A100" one-liner uses this)
#
# Env vars
#   NGC_API_KEY        (required) — your ngc.nvidia.com token
#   MODEL_REPO         (optional) — defaults to nvidia/Nemotron-Nano-9B-v2
#   ADAPTER_PATH       (optional) — local path to merged SFT+GRPO weights
#   HOST_PORT          (optional) — defaults to 8080
#   NIM_CACHE_PATH     (optional) — defaults to /opt/nim/.cache
#
# Brev one-liner for a fresh A100 instance:
#   curl -fsSL https://brev.dev/run.sh | bash -s -- \
#     --gpu a100 --repo shopfloor-nemotron --entry serve/nim_deploy.sh

set -euo pipefail

: "${NGC_API_KEY:?NGC_API_KEY env var is required}"

MODEL_REPO="${MODEL_REPO:-nvidia/Nemotron-Nano-9B-v2}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
HOST_PORT="${HOST_PORT:-8080}"
NIM_CACHE_PATH="${NIM_CACHE_PATH:-/opt/nim/.cache}"
CONTAINER_NAME="shopfloor-nim"
IMG_REPO="nvcr.io/nim/nvidia/nemotron-nano-9b"
IMG_TAG="latest"

echo "==> Logging into nvcr.io"
echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin

echo "==> Preparing NIM cache at ${NIM_CACHE_PATH}"
sudo mkdir -p "${NIM_CACHE_PATH}"
sudo chmod 777 "${NIM_CACHE_PATH}"

echo "==> Pulling image ${IMG_REPO}:${IMG_TAG}"
docker pull "${IMG_REPO}:${IMG_TAG}"

# Optional: mount local adapter weights for the SFT+GRPO-merged checkpoint.
ADAPTER_ARGS=()
if [[ -n "${ADAPTER_PATH}" ]]; then
    if [[ ! -d "${ADAPTER_PATH}" ]]; then
        echo "ERROR: ADAPTER_PATH ${ADAPTER_PATH} does not exist." >&2
        exit 1
    fi
    ADAPTER_ARGS=(
        -v "${ADAPTER_PATH}:/model-adapter:ro"
        -e "NIM_PEFT_SOURCE=/model-adapter"
    )
    echo "==> Mounting LoRA adapter from ${ADAPTER_PATH}"
fi

# Stop / remove previous container if any (idempotent)
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "==> Removing existing ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" || true
fi

echo "==> Starting NIM container on :${HOST_PORT}"
docker run -d --name "${CONTAINER_NAME}" \
    --gpus all \
    --shm-size=16g \
    -e NGC_API_KEY \
    -e NIM_MODEL_NAME="${MODEL_REPO}" \
    -v "${NIM_CACHE_PATH}:/opt/nim/.cache" \
    "${ADAPTER_ARGS[@]}" \
    -p "${HOST_PORT}:8000" \
    "${IMG_REPO}:${IMG_TAG}"

echo "==> Waiting for /v1/models to come up (up to 5 min)"
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${HOST_PORT}/v1/models" >/dev/null; then
        echo "    ready after $((i*5))s"
        break
    fi
    sleep 5
done

echo "==> 5-prompt smoke test"
PROMPTS=(
    "बेयरिंग जाम, P3 line down, motor गरम — give me RCA JSON"
    "Classify HSN for induction motor 5 kW three-phase"
    "Is IS 14543:2004 applicable to food-bottling compressors? Answer JSON."
    "மீட்டர் வேலை செய்யவில்லை — suggest SAP T-code in JSON"
    "Ambiguous fault: machine 'feels wrong'. Reply with confidence + next step."
)
for p in "${PROMPTS[@]}"; do
    echo "--- $p"
    curl -s "http://localhost:${HOST_PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "$(cat <<EOF
{
  "model": "${MODEL_REPO}",
  "messages": [{"role":"user","content":"${p}"}],
  "max_tokens": 256,
  "temperature": 0.0
}
EOF
)" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["choices"][0]["message"]["content"][:500])' \
        || echo "    (smoke prompt failed — check container logs)"
done

cat <<EOF

NIM is up.
  - OpenAI-compatible endpoint: http://localhost:${HOST_PORT}/v1
  - Inspect:  docker logs -f ${CONTAINER_NAME}
  - Stop:     docker rm -f ${CONTAINER_NAME}
EOF
