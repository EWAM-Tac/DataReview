from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..types import DimensionScore, EpisodeArrays
from ._util import clip01, finite_ratio, safe_mean


def _probe_video_ffprobe(path: str) -> dict[str, Any] | None:
    import json
    import shutil
    import subprocess

    if shutil.which("ffprobe") is None:
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            return {"error": proc.stderr.strip() or "ffprobe_failed"}
        data = json.loads(proc.stdout or "{}")
        streams = data.get("streams") or []
        if not streams:
            return {"error": "no_video_stream"}
        s0 = streams[0]
        w = int(s0.get("width") or 0)
        h = int(s0.get("height") or 0)
        n = int(s0.get("nb_frames") or 0) if str(s0.get("nb_frames", "")).isdigit() else 0
        return {
            "backend": "ffprobe",
            "codec": s0.get("codec_name"),
            "width": w,
            "height": h,
            "n_frames": n,
            "ok": w > 0 and h > 0,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _probe_video(path: str, n_frames_expect: int) -> dict[str, Any]:
    p = Path(path)
    out: dict[str, Any] = {"path": path, "exists": p.exists(), "ok": False}
    if not out["exists"]:
        return out
    out["size_bytes"] = p.stat().st_size
    if out["size_bytes"] < 1024:
        out["error"] = "file_too_small"
        return out

    # Prefer OpenCV when it can actually open the codec
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            ok_read, frame = cap.read()
            cap.release()
            out.update(
                {
                    "backend": "opencv",
                    "n_frames": n,
                    "width": w,
                    "height": h,
                    "read_ok": bool(ok_read),
                }
            )
            if ok_read and frame is not None:
                out["mean_pixel"] = float(frame.mean())
            if n_frames_expect > 0 and n > 0:
                out["len_ratio"] = n / n_frames_expect
            if ok_read and w > 0 and h > 0:
                out["ok"] = True
                return out
    except Exception as e:  # noqa: BLE001
        out["opencv_error"] = str(e)

    ff = _probe_video_ffprobe(path)
    if ff is not None:
        out.update(ff)
        if n_frames_expect > 0 and out.get("n_frames"):
            out["len_ratio"] = float(out["n_frames"]) / n_frames_expect
        return out

    # Last resort: non-empty file counts as weak OK
    out.update({"backend": "filesize", "ok": True, "weak": True})
    return out


def score_integrity(
    episodes: list[EpisodeArrays],
    *,
    role_map: dict[str, list[str]],
    fps: float,
    video_probe_per_ep: int = 2,
    **_: Any,
) -> DimensionScore:
    if not episodes:
        return DimensionScore("integrity", "Integrity", 0.0, status="fail", notes=["no episodes"])

    finite_scores = []
    ts_scores = []
    const_flags = []
    bad_eps = []

    for ep in episodes:
        frs = []
        for role, arr in ep.arrays.items():
            if not isinstance(arr, np.ndarray):
                continue
            fr = finite_ratio(arr)
            frs.append(fr)
            # near-constant channel collapse
            if arr.ndim == 2 and arr.shape[0] > 5:
                std = arr.std(axis=0)
                const_flags.append(float((std < 1e-8).mean()))
        finite_scores.append(safe_mean(frs, 1.0))
        if finite_scores[-1] < 0.999:
            bad_eps.append(ep.episode_index)

        if "timestamp" in ep.arrays:
            ts = ep.arrays["timestamp"].reshape(-1)
            if ts.size >= 2:
                dts = np.diff(ts.astype(np.float64))
                mono = float((dts >= -1e-4).mean())
                med = float(np.median(dts))
                expected = 1.0 / max(fps, 1e-6)
                jitter = float(np.std(dts) / max(abs(med), 1e-6)) if med != 0 else 1.0
                rate_ok = clip01(1.0 - abs(med - expected) / max(expected, 1e-6))
                ts_scores.append(0.5 * mono + 0.3 * clip01(1.0 - jitter) + 0.2 * rate_ok)
            else:
                ts_scores.append(0.5)

    # Video probes (light)
    video_ok_rates = []
    probed = 0
    for ep in episodes[: max(1, min(len(episodes), 30))]:
        keys = list(ep.video_paths.keys())[:video_probe_per_ep]
        for k in keys:
            r = _probe_video(ep.video_paths[k], ep.n_frames)
            probed += 1
            video_ok_rates.append(1.0 if r.get("ok") else 0.0)

    finite_s = safe_mean(finite_scores, 1.0)
    ts_s = safe_mean(ts_scores, 1.0) if ts_scores else None
    const_s = clip01(1.0 - safe_mean(const_flags, 0.0))
    vid_s = safe_mean(video_ok_rates, 1.0) if video_ok_rates else None

    parts = [finite_s, const_s]
    if ts_s is not None:
        parts.append(ts_s)
    if vid_s is not None:
        parts.append(vid_s)
    score = float(np.mean(parts))

    notes = []
    status = "ok"
    if finite_s < 0.999:
        status = "fail"
        notes.append(f"non-finite values present (finite_ratio={finite_s:.6f})")
    if vid_s is not None and vid_s < 0.9:
        status = "fail" if vid_s < 0.5 else "warn"
        notes.append(f"video probe ok rate {vid_s:.2f} (n={probed})")
    if ts_s is not None and ts_s < 0.7:
        status = "warn" if status == "ok" else status
        notes.append(f"timestamp quality {ts_s:.2f}")

    # Gate: fail if score very low
    if score < 0.5:
        status = "fail"

    return DimensionScore(
        id="integrity",
        name="Integrity",
        score=score,
        confidence=1.0,
        status=status,
        evidence={
            "finite_ratio": finite_s,
            "timestamp_score": ts_s,
            "nonconstant_score": const_s,
            "video_ok_rate": vid_s,
            "n_video_probes": probed,
            "bad_episode_examples": bad_eps[:20],
        },
        notes=notes,
    )
