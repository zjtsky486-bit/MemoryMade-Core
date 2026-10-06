from __future__ import annotations

import ipaddress
import json
import socket
import tempfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from .config import GENERATOR_API_KEY, GENERATOR_WEBHOOK
from .disk_guard import ensure_disk_space

MAX_RESPONSE_BYTES = 1024 * 1024
MAX_ARTIFACT_BYTES = 2 * 1024**3


def webhook_enabled() -> bool:
    return bool(GENERATOR_WEBHOOK)


def _validate_url(url: str, trusted_host: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('生成服务必须返回 HTTP 或 HTTPS 文件地址。')
    if parsed.hostname.lower() == trusted_host.lower():
        return  # An explicitly configured local generation server is supported.
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80), type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(entry[4][0]).is_global for entry in addresses):
        raise ValueError('生成服务返回了不允许的本机、内网或保留地址。')


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, trusted_host):
        self.trusted_host = trusted_host

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_url(newurl, self.trusted_host)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            for name in ('Authorization', 'X-api-key'):
                redirected.remove_header(name)
        return redirected


def _multipart(files: list[Path], fields: dict[str, str]):
    boundary = '----MEMORYMADE' + uuid.uuid4().hex
    body = tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode='w+b')
    try:
        # Large original photos go straight to the spool file, avoiding a second
        # in-memory copy when the first megabyte would otherwise roll over.
        if sum(file.stat().st_size for file in files) > 1024 * 1024:
            ensure_disk_space('准备外部生成素材')
            body.rollover()
        for key, value in fields.items():
            body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode('utf-8'))
        for index, file in enumerate(files):
            # Supported uploads have known types; Windows registry MIME discovery
            # adds startup latency and several megabytes for this fixed set.
            mime = {'.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg',
                    '.webp':'image/webp','.bmp':'image/bmp'}.get(file.suffix.lower(),'application/octet-stream')
            suffix = file.suffix if file.suffix.lower() in ('.png','.jpg','.jpeg','.webp','.bmp') else '.bin'
            body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="images"; filename="image-{index}{suffix}"\r\nContent-Type: {mime}\r\n\r\n'.encode('utf-8'))
            with file.open('rb') as reader:
                while chunk := reader.read(256 * 1024):
                    ensure_disk_space('准备外部生成素材')
                    body.write(chunk)
            body.write(b'\r\n')
        body.write(f'--{boundary}--\r\n'.encode('utf-8'))
        body.seek(0)
        return body, f'multipart/form-data; boundary={boundary}'
    except Exception:
        body.close()
        raise


def generate_via_webhook(image_paths: list[str], mode: str, prompt: str, output_dir: str | Path) -> dict | None:
    if not webhook_enabled():
        return None
    ensure_disk_space('外部三维生成')
    files = [Path(path) for path in image_paths if Path(path).is_file()]
    if not files:
        return None
    trusted_host = urllib.parse.urlsplit(GENERATOR_WEBHOOK).hostname or ''
    _validate_url(GENERATOR_WEBHOOK, trusted_host)
    opener = urllib.request.build_opener(_SafeRedirect(trusted_host))
    body, content_type = _multipart(files, {'mode': mode, 'prompt': prompt})
    with body:
        body.seek(0, 2);length=body.tell();body.seek(0)
        headers = {'Content-Type': content_type, 'Content-Length': str(length), 'Accept': 'application/json'}
        if GENERATOR_API_KEY:
            headers['Authorization'] = f'Bearer {GENERATOR_API_KEY}'
            headers['X-API-Key'] = GENERATOR_API_KEY
        request = urllib.request.Request(GENERATOR_WEBHOOK, data=body, headers=headers, method='POST')
        with opener.open(request, timeout=600) as response:
            metadata = response.read(MAX_RESPONSE_BYTES + 1)
        if len(metadata) > MAX_RESPONSE_BYTES:
            raise ValueError('生成服务响应过大，请检查服务配置。')
        payload = json.loads(metadata.decode('utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('生成服务响应格式错误。')
    output_dir = Path(output_dir) / ('cloud-' + uuid.uuid4().hex)
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {}
    for key in ('glb', 'stl', 'preview'):
        url = payload.get(f'{key}_url') or payload.get(key)
        if not url:
            continue
        if not isinstance(url, str):raise ValueError('生成服务文件地址格式错误。')
        _validate_url(url, trusted_host)
        suffix = Path(urllib.parse.urlsplit(url).path).suffix.lower() if key == 'preview' else '.' + key
        if key == 'preview' and suffix not in ('.jpg','.jpeg','.png','.webp'):suffix='.jpg'
        target = output_dir / f'cloud-{key}{suffix}'
        temporary = target.with_suffix(target.suffix + '.part')
        try:
            total = 0
            with opener.open(url, timeout=600) as remote, temporary.open('wb') as writer:
                declared=getattr(remote,'headers',{}).get('Content-Length')
                expected=int(declared) if declared is not None else None
                if expected is not None and (expected<=0 or expected>MAX_ARTIFACT_BYTES):raise ValueError('外部文件大小无效或超过 2 GiB。')
                while chunk := remote.read(256 * 1024):
                    total += len(chunk)
                    if total > MAX_ARTIFACT_BYTES:raise ValueError('外部文件超过 2 GiB，请检查生成服务。')
                    ensure_disk_space('下载外部三维文件')
                    writer.write(chunk)
            if not total:raise ValueError('生成服务返回了空文件。')
            if expected is not None and total!=expected:raise ValueError('外部文件下载不完整，旧模型已保留。')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        result[key] = str(target)
    return result or None
