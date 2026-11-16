"""Unix socket TTS 客户端，协议与 shittim 完全一致。

发：{"text": "...", "out": "...", "lang": "ja"}\n
收：{"status": "ok", "out": "..."}\n
"""
from __future__ import annotations

import json
import socket


def synthesize(
    text: str,
    out_wav: str,
    sock_path: str = "/tmp/plana_tts.sock",
    lang: str = "ja",
    timeout: float = 300,
) -> str:
    req = json.dumps({"text": text, "out": out_wav, "lang": lang}) + "\n"

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
        s.sendall(req.encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
    finally:
        s.close()

    resp = json.loads(data.decode().strip())
    if resp.get("status") != "ok":
        raise RuntimeError(f"TTS 服务返回错误：{resp}")
    return out_wav
