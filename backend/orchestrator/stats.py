"""GPU / disk statistics, sampled in the background (one sampler for all users)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import psutil

log = logging.getLogger("stats")

try:
    import pynvml  # provided by the nvidia-ml-py package

    pynvml.nvmlInit()
    NVML = True
except Exception as e:  # noqa: BLE001
    log.warning("NVML unavailable: %s", e)
    NVML = False

DISK_PATHS = ("/data", "/models", "/")


def _gpu() -> list[dict[str, Any]]:
    if not NVML:
        return []
    gpus = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        g: dict[str, Any] = {
            "index": i,
            "name": pynvml.nvmlDeviceGetName(h),
            "mem_used_gb": round(mem.used / 1024**3, 2),
            "mem_total_gb": round(mem.total / 1024**3, 2),
            "mem_pct": round(100 * mem.used / mem.total, 1),
            "gpu_util_pct": util.gpu,
        }
        for key, fn in (
            ("temp_c", lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)),
            ("power_w", lambda: round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1)),
            ("power_limit_w", lambda: round(pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000, 1)),
            ("fan_pct", lambda: pynvml.nvmlDeviceGetFanSpeed(h)),
            ("compute_capability", lambda: "%d.%d" % pynvml.nvmlDeviceGetCudaComputeCapability(h)),
        ):
            try:
                g[key] = fn()
            except pynvml.NVMLError:
                g[key] = None
        gpus.append(g)
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        cuda = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        for g in gpus:
            g["driver"] = driver
            g["cuda_driver"] = f"{cuda // 1000}.{(cuda % 1000) // 10}"
    except pynvml.NVMLError:
        pass
    return gpus


def _disk() -> list[dict[str, Any]]:
    out = []
    for p in DISK_PATHS:
        try:
            u = psutil.disk_usage(p)
        except OSError:
            continue
        out.append(
            {
                "path": p,
                "used_gb": round(u.used / 1024**3, 1),
                "total_gb": round(u.total / 1024**3, 1),
                "pct": u.percent,
            }
        )
    return out


class StatsSampler:
    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.latest: dict[str, Any] = {"gpu": [], "disk": [], "ts": 0}
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        while True:
            try:
                gpu, disk = await asyncio.to_thread(lambda: (_gpu(), _disk()))
                self.latest = {"gpu": gpu, "disk": disk, "ts": time.time()}
            except Exception:  # noqa: BLE001
                log.exception("stats sampling failed")
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="stats")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
