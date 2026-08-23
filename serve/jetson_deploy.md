# Jetson Orin Nano Deployment — ShopFloor-Nemotron

End-to-end recipe for getting the NVFP4-quantised model running on a
Jetson Orin Nano 8GB Super-Dev-Kit as the booth's primary inference path.

Latency budget (end-to-end voice-in -> SAP POST): **340 ms**

| Stage                 | Target  | Notes                              |
|-----------------------|---------|------------------------------------|
| Whisper.cpp small ASR | 100 ms  | Hinglish / Tamil code-mix          |
| Nemotron-SAP inference| 200 ms  | NVFP4 engine, 64 new tokens median |
| S/4HANA REST POST     |  40 ms  | local LAN, JSON                    |
| **Total**             | **340 ms** | Within booth WiFi margin        |

If the engine fails to load (Jetson SD card corruption, JetPack drift, etc.)
the `--ghost-mode` flag on `jetson_inference.py` switches to the
pre-recorded replay path in `serve/ghost_demo.py` so the booth demo never
goes dark.

---

## 1. Flash JetPack 6.0 on the Orin Nano

```bash
# On host workstation (Ubuntu 22.04 recommended)
sudo apt install -y nvidia-jetpack-config qemu-user-static
wget https://developer.nvidia.com/jetpack-6.0-orin-nano-sd-card-image -O jp6.zip
unzip jp6.zip
sudo dd if=JetPack_6.0_Orin_Nano.img of=/dev/sdX bs=4M status=progress
```

Boot the Jetson, finish the OEM config (English, headless or DP), then on
the device:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-pip python3-venv git ffmpeg portaudio19-dev
```

## 2. Install TensorRT-LLM (ARM aarch64)

JetPack 6 ships CUDA 12.4 + TensorRT 10. TRT-LLM has an aarch64 wheel:

```bash
python3 -m venv ~/sfn-venv && source ~/sfn-venv/bin/activate
pip install --upgrade pip
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
pip install fastapi uvicorn[standard] httpx pydantic
pip install openai-whisper  # we use whisper.cpp at runtime; this lib is for tokenizer fallback
```

Verify the wheel sees the GPU:

```bash
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
nvidia-smi
```

If `nvidia-smi` is missing, install the JetPack CUDA stack:

```bash
sudo apt install -y nvidia-jetpack
```

## 3. Whisper.cpp small (Hinglish ASR)

```bash
git clone https://github.com/ggerganov/whisper.cpp ~/whisper.cpp
cd ~/whisper.cpp
make -j$(nproc) WHISPER_CUBLAS=1
bash ./models/download-ggml-model.sh small
# warmup
./main -m models/ggml-small.bin -f samples/jfk.wav -l auto
```

We call whisper.cpp via subprocess from `jetson_inference.py` (stable
ABI, no Python binding drift across JetPack releases).

## 4. Copy the engine

```bash
# from training workstation
scp engines/shopfloor-nano-nvfp4.engine jetson@192.168.0.42:/opt/shopfloor/
scp -r engines/shopfloor-nano-nvfp4/  jetson@192.168.0.42:/opt/shopfloor/
```

On the Jetson:

```bash
sudo mkdir -p /opt/shopfloor && sudo chown $USER:$USER /opt/shopfloor
ls -lh /opt/shopfloor/
# expected: shopfloor-nano-nvfp4.engine  (~3-4 GiB after NVFP4 quant)
```

## 5. Run the inference server

```bash
export SAP_PM_URL="http://sap-pm.local:8000/notifications"
export WHISPER_BIN="$HOME/whisper.cpp/main"
export WHISPER_MODEL="$HOME/whisper.cpp/models/ggml-small.bin"
python3 -m serve.jetson_inference \
    --engine /opt/shopfloor/shopfloor-nano-nvfp4 \
    --host 0.0.0.0 --port 9000
```

Smoke test from a laptop:

```bash
curl -F "audio=@samples/bearing_jam.wav" http://jetson.local:9000/listen | jq .
```

Expected response (latency_ms < 340):

```json
{
  "transcript": "बेयरिंग जाम P3 line down motor गरम",
  "rca": "Bearing seizure due to lubrication failure ...",
  "bis": "IS 14543",
  "hsn": "84821010",
  "tcode": "IW21",
  "confidence": 0.91,
  "latency_ms": 312
}
```

## 6. Ghost-demo replay fallback

If `model.generate` errors or the engine fails to load on startup,
`jetson_inference.py` flips to ghost mode automatically. You can also
force it for booth dress rehearsal:

```bash
python3 -m serve.jetson_inference --ghost-mode --port 9000
```

The Gradio booth UI (`serve/ghost_demo.py`) reads
`serve/ghost_demo_cache.json` and replays 5 hand-curated cases in the
exact JSON shape the live model emits. Ctrl+G toggles live <-> ghost in
the booth UI; either side of the Jetson dying mid-demo still gives the
visitor a meaningful response in <1 s.

## 7. Systemd unit (booth auto-start)

`/etc/systemd/system/shopfloor.service`:

```ini
[Unit]
Description=ShopFloor-Nemotron Jetson inference
After=network-online.target

[Service]
User=jetson
WorkingDirectory=/home/jetson/shopfloor-nemotron
Environment=SAP_PM_URL=http://sap-pm.local:8000/notifications
ExecStart=/home/jetson/sfn-venv/bin/python -m serve.jetson_inference \
    --engine /opt/shopfloor/shopfloor-nano-nvfp4 --host 0.0.0.0 --port 9000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now shopfloor
journalctl -u shopfloor -f
```

## 8. Booth checklist

- [ ] Jetson powered via 65W USB-C PSU (not the 5V/4A old brick — TRT-LLM trips OOM under-power)
- [ ] WiFi reachable on the booth SSID
- [ ] `nvidia-smi` shows the engine memory pinned (~3.5 GiB)
- [ ] `curl /healthz` returns 200 within 5 s of boot
- [ ] `serve/ghost_demo.py` running on the laptop, Ctrl+G works
- [ ] Recovery USB stick with the .engine file at hand
