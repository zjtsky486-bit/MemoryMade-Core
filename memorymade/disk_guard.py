"""Stop MEMORYMADE and its own child processes at the C: disk reserve."""
from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from .storage_policy import reserve_bytes,manual_approval_required,d_reserve_bytes
except ImportError:
    from storage_policy import reserve_bytes,manual_approval_required,d_reserve_bytes
MIN_FREE_BYTES = reserve_bytes()
D_RESERVE_BYTES = d_reserve_bytes()
CHECK_SECONDS = 2
STOP_EVENT = threading.Event()
_started = False


class DiskReserveError(RuntimeError):
    pass


def free_bytes():
    return shutil.disk_usage('C:\\' if os.name == 'nt' else '/').free


def ensure_disk_space(action='运行任务'):
    if STOP_EVENT.is_set():
        raise DiskReserveError(f'磁盘保护已触发，停止{action}。请检查剩余空间并重新启动软件。')
    available = free_bytes()
    if os.name=='nt' and Path('D:/1223456789').is_dir() and shutil.disk_usage('D:/').free<=D_RESERVE_BYTES:
        STOP_EVENT.set()
        raise DiskReserveError(f'D 盘已达到 {D_RESERVE_BYTES/1024**3:g} GiB 保护线，停止{action}，保留已有文件。')
    if manual_approval_required() and available <= MIN_FREE_BYTES:
        STOP_EVENT.set()
        raise DiskReserveError(f'C 盘剩余 {available/1024**3:.2f} GiB，达到{MIN_FREE_BYTES/1024**3:g} GiB停止线，后续操作需要用户允许。')
    target = os.getenv('MEMORYMADE_WRITE_ROOT', '')
    if target and Path(target).resolve().drive.upper() == 'D:':
        if shutil.disk_usage(target).free <= D_RESERVE_BYTES:
            STOP_EVENT.set()
            raise DiskReserveError(f'D 盘已达到 {D_RESERVE_BYTES/1024**3:g} GiB 保护线，停止{action}，保留已有文件。')
        return available
    if STOP_EVENT.is_set() or available <= MIN_FREE_BYTES:
        STOP_EVENT.set()
        if manual_approval_required():
            raise DiskReserveError(f'C 盘剩余 {available/1024**3:.2f} GiB，已达到 {MIN_FREE_BYTES/1024**3:g} GiB 保护线，停止{action}。后续步骤需要用户允许，不自动切换或恢复。')
        raise DiskReserveError(f'C 盘剩余 {available / 1024**3:.2f} GB，已达到 {MIN_FREE_BYTES/1024**3:g} GiB 保护线，停止{action}。自动切换启动器将后续任务改存到 D:\\1223456789；当前未完成任务需重新开始。')
    return available


def record_stop(message):
    try:
        logs = Path(os.getenv('MEMORYMADE_WRITE_ROOT', str(Path(__file__).resolve().parents[2]))) / 'logs'
        if os.name=='nt' and free_bytes()<=MIN_FREE_BYTES and Path('D:/').is_dir():
            root=Path('D:/1223456789');root.mkdir(parents=True,exist_ok=True)
            marker='.memorymade-manual-approval-required.json' if manual_approval_required() else '.memorymade-storage-requested.json'
            (root/marker).write_text(json.dumps({'resource_root':os.getenv('MEMORYMADE_RESOURCE_ROOT',str(Path(__file__).resolve().parents[2])),'reason':message},ensure_ascii=False),encoding='utf-8')
            logs=root/'logs'
        logs.mkdir(parents=True,exist_ok=True)
        with (logs / 'disk-protection.log').open('a', encoding='utf-8') as stream:
            stream.write(time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + message + '\n')
    except OSError:
        pass


def stop_tree(pid):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    else:
        os.kill(pid, 15)


def start_watchdog():
    global _started
    ensure_disk_space('启动软件')
    if _started:
        return
    _started = True
    def monitor():
        while True:
            signalled=STOP_EVENT.wait(CHECK_SECONDS)
            try:
                if signalled:raise DiskReserveError('磁盘警戒线已触发，本软件停止，不再启动任务。')
                ensure_disk_space('软件及其识别、生成、Blender、导出等任务')
            except (DiskReserveError, OSError) as exc:
                STOP_EVENT.set()
                record_stop(str(exc))
                # Kill only this software process and its descendants.
                stop_tree(os.getpid())
                os._exit(70)
    threading.Thread(target=monitor, name='C-drive-reserve-watchdog', daemon=True).start()


def run_guarded(command, timeout=None, **kwargs):
    ensure_disk_space('启动外部任务')
    process = subprocess.Popen(command, **kwargs)
    started = time.monotonic()
    try:
        while process.poll() is None:
            try:
                ensure_disk_space('外部任务')
            except (DiskReserveError, OSError) as exc:
                record_stop(str(exc))
                raise
            if timeout is not None and time.monotonic() - started > timeout:
                raise subprocess.TimeoutExpired(command, timeout)
            time.sleep(.5)
        ensure_disk_space('完成任务')
        return subprocess.CompletedProcess(command, process.returncode)
    finally:
        if process.poll() is None:
            stop_tree(process.pid)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == '__main__':
    command = sys.argv[1:]
    if command and command[0] == '--':
        command = command[1:]
    try:
        if command:
            raise SystemExit(run_guarded(command).returncode)
        print(f'C 盘剩余 {ensure_disk_space("检查磁盘") / 1024**3:.2f} GiB，保护线 {MIN_FREE_BYTES/1024**3:g} GiB。')
    except (DiskReserveError, OSError) as exc:
        record_stop(str(exc))
        print(str(exc), file=sys.stderr)
        raise SystemExit(70)
