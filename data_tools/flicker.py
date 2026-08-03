"""Detect mains-related exposure flicker and rolling brightness bands in video."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FlickerResult:
    video: str
    frames: int
    fps: float
    flagged: bool
    flags: list[str]
    luma_mean: float
    luma_relative_std: float
    dominant_frequency_hz: float
    dominant_power_fraction: float
    mains_alias_hz: float
    mains_alias_power_fraction: float
    row_delta_median: float
    row_delta_p95: float
    row_low_correlation_fraction: float


def _spectral_metrics(luma: np.ndarray, fps: float, mains_hz: float) -> tuple[float, float, float, float]:
    if len(luma) < max(30, int(fps * 2)):
        return 0.0, 0.0, 0.0, 0.0
    # Remove scene/exposure drift with a one-second moving mean. Edge samples are
    # excluded because convolution padding otherwise creates artificial peaks.
    window = max(3, int(round(fps)))
    trend = np.convolve(luma, np.ones(window) / window, mode="same")
    signal = luma - trend
    signal[:window] = 0.0
    signal[-window:] = 0.0
    freq = np.fft.rfftfreq(len(signal), 1.0 / fps)
    power = np.abs(np.fft.rfft(signal)) ** 2
    valid = (freq >= 0.3) & (freq <= min(15.0, fps / 2.0))
    total = float(power[valid].sum())
    if total <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    valid_indices = np.flatnonzero(valid)
    peak_index = valid_indices[int(np.argmax(power[valid]))]
    dominant_hz = float(freq[peak_index])
    dominant_fraction = float(power[peak_index] / total)
    # A sampled sinusoid aliases to |f - round(f/fs)*fs|. Illumination can pulse
    # at mains frequency or twice mains frequency; take whichever visible alias
    # lies inside the sampled band (50Hz at 30fps -> 10Hz).
    aliases = []
    for physical in (mains_hz, 2.0 * mains_hz):
        alias = abs(physical - round(physical / fps) * fps)
        alias = min(alias, abs(fps - alias))
        if 0.3 <= alias <= fps / 2.0:
            aliases.append(alias)
    alias_hz = aliases[0] if aliases else 0.0
    fractions = []
    for alias in aliases:
        band = valid & (np.abs(freq - alias) <= 0.5)
        fractions.append(float(power[band].sum() / total))
    return dominant_hz, dominant_fraction, alias_hz, max(fractions, default=0.0)


def flicker_metrics(
    luma: np.ndarray,
    row_profiles: np.ndarray,
    fps: float,
    *,
    mains_hz: float = 50.0,
) -> dict[str, float]:
    dominant_hz, dominant_fraction, alias_hz, alias_fraction = _spectral_metrics(
        luma, fps, mains_hz
    )
    if len(row_profiles) >= 2:
        row_delta = np.sqrt(np.mean(np.diff(row_profiles, axis=0) ** 2, axis=1))
        correlations = []
        for previous, current in zip(row_profiles[:-1], row_profiles[1:], strict=True):
            if previous.std() <= 1e-8 or current.std() <= 1e-8:
                correlations.append(1.0)
            else:
                correlations.append(float(np.corrcoef(previous, current)[0, 1]))
        corr = np.nan_to_num(np.asarray(correlations), nan=1.0)
    else:
        row_delta = np.zeros(1)
        corr = np.ones(1)
    return {
        "luma_mean": float(luma.mean()) if len(luma) else 0.0,
        "luma_relative_std": float(luma.std() / max(luma.mean(), 1e-6)) if len(luma) else 0.0,
        "dominant_frequency_hz": dominant_hz,
        "dominant_power_fraction": dominant_fraction,
        "mains_alias_hz": alias_hz,
        "mains_alias_power_fraction": alias_fraction,
        "row_delta_median": float(np.median(row_delta)),
        "row_delta_p95": float(np.percentile(row_delta, 95)),
        "row_low_correlation_fraction": float(np.mean(corr < 0.90)),
    }


def inspect_video(
    video: Path,
    *,
    mains_hz: float = 50.0,
    downscale_stride: int = 8,
    max_frames: int = 0,
    row_delta_threshold: float = 3.0,
    low_correlation_fraction_threshold: float = 0.02,
    alias_power_threshold: float = 0.12,
) -> FlickerResult:
    import av

    means: list[float] = []
    profiles: list[np.ndarray] = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 30.0)
        for frame in container.decode(stream):
            gray = frame.to_ndarray(format="gray")[::downscale_stride, ::downscale_stride].astype(np.float32)
            means.append(float(gray.mean()))
            profile = gray.mean(axis=1)
            profiles.append(profile - profile.mean())
            if max_frames and len(means) >= max_frames:
                break
    metrics = flicker_metrics(np.asarray(means), np.asarray(profiles), fps, mains_hz=mains_hz)
    flags: list[str] = []
    if (
        metrics["row_delta_median"] >= row_delta_threshold
        and metrics["row_low_correlation_fraction"] >= low_correlation_fraction_threshold
    ):
        flags.append("rolling_brightness_bands")
    if metrics["mains_alias_power_fraction"] >= alias_power_threshold:
        flags.append("periodic_mains_alias")
    return FlickerResult(
        video=str(video), frames=len(means), fps=fps, flagged=bool(flags), flags=flags, **metrics
    )


def discover_videos(path: Path, camera: str | None = None) -> list[Path]:
    if path.is_file():
        return [path]
    videos = sorted(path.glob("**/episode_*.mp4"))
    if camera:
        expected = f"observation.images.{camera}"
        videos = [video for video in videos if video.parent.name in {camera, expected}]
    return videos


def scan_flicker(
    path: Path,
    *,
    camera: str | None = None,
    sample: int = 0,
    output: Path | None = None,
    **kwargs,
) -> list[FlickerResult]:
    videos = discover_videos(path, camera)
    if sample and len(videos) > sample:
        indices = np.linspace(0, len(videos) - 1, sample, dtype=int)
        videos = [videos[index] for index in sorted(set(indices.tolist()))]
    results = [inspect_video(video, **kwargs) for video in videos]
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(asdict(result), ensure_ascii=False) + "\n" for result in results),
            encoding="utf-8",
        )
    return results
