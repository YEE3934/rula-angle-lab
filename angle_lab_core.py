"""RULA Angle Lab — research-only 2D multi-view diagnostics.

This deliberately does NOT calculate official RULA Grand Scores, replace validated
anatomical angles, or reconstruct calibrated 3D. Thresholds are exploratory.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd

SIDES = ("左手側", "右手側")
CAMERAS = ("Left", "Front", "Right")
SIDE_INDICES = {
    "左手側": {"ear": 7, "shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23},
    "右手側": {"ear": 8, "shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24},
}
PARTS = ("前臂", "頸部")
FIELDS = ("sample_id", "time_sec", "body_side", "manual_forearm_deg", "manual_neck_deg",
          "baseline_forearm_deg", "baseline_neck_deg")


def angle2d(a: Any, b: Any, c: Any) -> float:
    """Interior projected joint angle at b; coordinates must use SAME pixel scale."""
    u = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom < 1e-8:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(np.dot(u, v)) / denom, -1.0, 1.0))))


def _pt(lm: dict[int, dict[str, float]], index: int, width: int, height: int):
    d = lm.get(index)
    if not d:
        return None
    try:
        x, y, vis = float(d["x"]), float(d["y"]), float(d.get("visibility", 0.0))
        if not all(math.isfinite(a) for a in (x, y, vis)):
            return None
        return (np.array([x * width, y * height], dtype=float), max(0.0, min(vis, 1.0)))
    except (ValueError, TypeError, KeyError):
        return None



def legacy_neck_proxy(lm: dict[int, dict[str, float]] | None, width: int, height: int, side: str) -> float:
    """Research diagnostic for 222.py's older blended eye/ear + torso-center proxy.

    Returns a 2D angle for methodological comparison, NEVER an automatic
    replacement for the adopted v11.4 same-side neck angle.
    """
    ids=SIDE_INDICES[side]
    get=lambda index:_pt(lm or {},index,width,height)
    ear=get(ids["ear"])
    eye_pts=[get(2),get(5)]
    eye=max((v for v in eye_pts if v is not None),key=lambda v:v[1],default=None)
    shoulders=[get(11),get(12)]
    hips=[get(23),get(24)]
    if ear is None or eye is None or any(x is None for x in shoulders+hips):
        return float("nan")
    if min([ear[1],eye[1]]+[p[1] for p in shoulders+hips]) < .4:
        return float("nan")
    sh=(shoulders[0][0]+shoulders[1][0])/2
    hp=(hips[0][0]+hips[1][0])/2
    head=ear[0]*.6+eye[0]*.4
    angle=angle2d(hp,sh,head)
    return round(abs(180-angle),2) if math.isfinite(angle) else float("nan")

def _candidate_base(camera: str, side: str, part: str) -> dict:
    return {"camera": camera, "body_side": side, "part": part, "angle_deg": float("nan"),
            "min_visibility": 0.0, "quality": 0.0, "eligible": False,
            "warnings": "", "projected_length_ratio": float("nan"),
            "shoulder_width_torso_ratio": float("nan"),
            "forearm_upper_ratio": float("nan"), "interior_angle_deg": float("nan"),
            "neck_legacy_proxy_deg": float("nan"), "geometry_ok": False}


def candidates_from_landmarks(lm: dict[int, dict[str, float]] | None,
                              width: int, height: int, camera: str, side: str) -> list[dict]:
    """Compute candidate measures for one anatomical side from one camera.

    Original 222.py lower-arm angle = abs(180 - angle(shoulder, elbow, wrist));
    also store the elbow interior angle for annotation and manual comparability.
    Neck proxy follows v11.4: abs(180 - angle(hip, shoulder, ear)).
    """
    if camera not in CAMERAS or side not in SIDES:
        raise ValueError("Unknown camera or anatomical side")
    ids = SIDE_INDICES[side]
    pts = {k: _pt(lm or {}, idx, width, height) for k, idx in ids.items()}
    sh1 = _pt(lm or {}, 11, width, height)
    sh2 = _pt(lm or {}, 12, width, height)
    torso = (float(np.linalg.norm(pts["shoulder"][0] - pts["hip"][0]))
             if pts["shoulder"] is not None and pts["hip"] is not None else float("nan"))
    shoulder_width_ratio = (float(np.linalg.norm(sh1[0]-sh2[0]))/torso
                            if sh1 is not None and sh2 is not None and math.isfinite(torso) and torso > 1 else float("nan"))

    result = []
    for part in PARTS:
        c = _candidate_base(camera, side, part)
        c["shoulder_width_torso_ratio"] = shoulder_width_ratio
        if part == "前臂":
            names = ("shoulder", "elbow", "wrist")
        else:
            names = ("hip", "shoulder", "ear")
        if any(pts[k] is None for k in names):
            c["warnings"] = "必要關鍵點未找到"
            result.append(c)
            continue
        xyz = [pts[k][0] for k in names]
        vis = min(pts[k][1] for k in names)
        c["min_visibility"] = round(vis, 3)
        interior = angle2d(*xyz)
        if part == "前臂" and math.isfinite(interior):
            c["interior_angle_deg"] = round(interior,2)
        angle = abs(180.0 - interior) if math.isfinite(interior) else float("nan")
        c["angle_deg"] = round(angle, 2) if math.isfinite(angle) else float("nan")
        warnings = []
        if vis < .55:
            warnings.append("關鍵點可見度較低")
        if not math.isfinite(angle):
            warnings.append("投影角度無法計算")
        if part == "前臂":
            length_up = float(np.linalg.norm(xyz[0]-xyz[1]))
            length_fore = float(np.linalg.norm(xyz[1]-xyz[2]))
            ratio = length_fore / max(length_up, 1e-6)
            c["forearm_upper_ratio"] = round(ratio, 3)
            c["projected_length_ratio"] = round(length_fore / torso, 3) if math.isfinite(torso) and torso > 1 else float("nan")
            geometry_ok = (length_up >= 10 and length_fore >= 10 and .40 <= ratio <= 2.70 and
                           math.isfinite(torso) and torso >= 20)
            if not geometry_ok:
                warnings.append("肘腕投影短／肢段比例不合理，可能遮擋或離平面")
            # Frontal view can be useful for some elbow motions, but not automatically reliable.
            if camera == "Front":
                warnings.append("正面前臂夾角是否適合量測，需視動作平面判定")
        else:
            c["neck_legacy_proxy_deg"] = legacy_neck_proxy(lm,width,height,side)
            head_len = float(np.linalg.norm(xyz[2]-xyz[1]))
            ratio = head_len / max(torso, 1e-6)
            c["projected_length_ratio"] = round(ratio, 3)
            geometry_ok = (math.isfinite(torso) and torso >= 28 and head_len >= 8 and .08 <= ratio <= .92 and angle <= 85)
            if not geometry_ok:
                warnings.append("頸部參考線尺度／角度可疑")
            if camera == "Front":
                warnings.append("正面影像不直接用於矢狀面頸部前屈角度替換")
            elif math.isfinite(shoulder_width_ratio) and shoulder_width_ratio > .55:
                warnings.append("肩寬投影大，側視鏡頭可能已非正側面")
        c["geometry_ok"] = bool(geometry_ok)
        c["quality"] = round(0.65 * vis + (0.35 if geometry_ok else 0.0), 3)
        c["eligible"] = bool(vis >= .55 and geometry_ok and math.isfinite(angle) and
                             (part != "頸部" or (camera != "Front" and
                              math.isfinite(shoulder_width_ratio) and shoulder_width_ratio <= .55)))
        c["warnings"] = "；".join(warnings) if warnings else "通過初步幾何檢查；不代表實際角度準確"
        result.append(c)
    return result


def refine_relative_projection(candidates: list[dict]) -> list[dict]:
    """Cross-camera projected-length heuristic; do not treat as ground-truth 3D."""
    copied = [dict(c) for c in candidates]
    by_part_side = {(x["body_side"], x["part"]) for x in copied}
    for side, part in by_part_side:
        if part != "前臂":
            continue
        subset = [x for x in copied if x["body_side"] == side and x["part"] == part]
        lengths = [x["projected_length_ratio"] for x in subset if math.isfinite(x["projected_length_ratio"]) and x["geometry_ok"]]
        if len(lengths) < 2 or max(lengths) <= .01:
            continue
        max_len = max(lengths)
        for x in subset:
            ratio = x["projected_length_ratio"] / max_len if math.isfinite(x["projected_length_ratio"]) else float("nan")
            x["relative_projection"] = round(ratio, 3) if math.isfinite(ratio) else float("nan")
            if math.isfinite(ratio) and ratio < .60:
                x["eligible"] = False
                x["quality"] = round(max(0.0, x["quality"] - .22), 3)
                x["warnings"] += "；相對其他鏡頭前臂投影顯著縮短（僅啟發式）"
            elif math.isfinite(ratio) and ratio < .78:
                x["quality"] = round(max(0.0, x["quality"] - .09), 3)
                x["warnings"] += "；前臂投影偏短（留意）"
    return copied


def choose_candidate(candidates: list[dict], side: str, part: str) -> dict:
    """Conservative RESEARCH recommendation; never substitutes a formal RULA angle."""
    base_cam = "Left" if side == "左手側" else "Right"
    subset = [c for c in candidates if c["body_side"] == side and c["part"] == part]
    base = next((c for c in subset if c["camera"] == base_cam), None)
    good = sorted([c for c in subset if c["eligible"]], key=lambda x: x["quality"], reverse=True)
    choice = None
    if base and base["eligible"]:
        choice = base
        if good and good[0]["camera"] != base_cam and good[0]["quality"] > base["quality"] + .12:
            choice = good[0]
    elif good:
        choice = good[0]
    state = "無合格視角"
    caution = "三個鏡頭的品質條件均不充分；需人工評估或保留不確定"
    if choice:
        state = "維持原視角" if choice["camera"] == base_cam else "研究候選：跨鏡頭"
        caution = "品質門檻為工程假設，仍須人工角度驗證；不能直接代入正式RULA"
        if base is not None and math.isfinite(base["angle_deg"]) and choice["camera"] != base_cam:
            diff = abs(choice["angle_deg"] - base["angle_deg"])
            if diff >= 15:
                caution += f"；與原視角差 {diff:.1f}°，建議人工檢查"
        if part == "頸部":
            caution += "；仍是耳肩髖2D代理角，非解剖學頸椎前屈真值"
    return {"body_side": side, "part": part, "baseline_camera": base_cam,
            "baseline_angle_deg": base["angle_deg"] if base else float("nan"),
            "selected_camera": choice["camera"] if choice else "無",
            "selected_angle_deg": choice["angle_deg"] if choice else float("nan"),
            "selected_quality": choice["quality"] if choice else float("nan"),
            "decision": state, "decision_note": caution}


def evaluate_paired(df: pd.DataFrame, forearm_manual_kind: str = "flexion") -> pd.DataFrame:
    """Compare exact SAME cases. Convert manually measured elbow interior to 222.py flexion if requested.

    forearm_manual_kind = flexion | interior | unknown. Unknown blocks forearm MAE.
    """
    if forearm_manual_kind not in ("flexion", "interior", "unknown"):
        raise ValueError("Invalid manual elbow-angle definition")
    output = []
    for part, key in (("前臂", "forearm"), ("頸部", "neck")):
        cols = (f"manual_{key}_deg", f"baseline_{key}_deg", f"selected_{key}_deg")
        if not all(c in df.columns for c in cols):
            continue
        a, b, c = [pd.to_numeric(df[col], errors="coerce") for col in cols]
        if part == "前臂":
            if forearm_manual_kind == "interior":
                a = 180.0 - a
            elif forearm_manual_kind == "unknown":
                a = pd.Series(np.nan,index=a.index,dtype=float)
        mask = a.notna() & b.notna() & c.notna()
        n_manual = int(a.notna().sum()); n = int(mask.sum())
        mae_base = float((a[mask]-b[mask]).abs().mean()) if n else float("nan")
        mae_new = float((a[mask]-c[mask]).abs().mean()) if n else float("nan")
        output.append({"部位": part, "人工有效筆數": n_manual, "配對樣本數": n,
                       "原始MAE(度)": round(mae_base,3) if n else float("nan"),
                       "候選MAE(度)": round(mae_new,3) if n else float("nan"),
                       "改善(度)": round(mae_base-mae_new,3) if n else float("nan"),
                       "候選可用且有人工值": int((a.notna() & c.notna()).sum())})
    return pd.DataFrame(output)


def parse_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Accept exact template and limited Chinese aliases; reject ambiguous units."""
    aliases = {"樣本編號":"sample_id", "樣本序號":"sample_id", "時間(秒)":"time_sec",
               "時間（秒）":"time_sec", "時間(秒數)":"time_sec", "評估側":"body_side",
               "人工前臂角度":"manual_forearm_deg", "人工頸部角度":"manual_neck_deg",
               "系統前臂角度":"baseline_forearm_deg", "系統頸部角度":"baseline_neck_deg"}
    df = df.copy().rename(columns=aliases)
    if "time_sec" not in df.columns:
        raise ValueError("CSV 必須有 time_sec 或 時間(秒) 欄位；不能只靠樣本序號猜時間")
    if "sample_id" not in df.columns:
        df["sample_id"] = range(1, len(df)+1)
    if "body_side" not in df.columns:
        left = df.copy(); left["body_side"] = "左手側"
        right = df.copy(); right["body_side"] = "右手側"
        df = pd.concat([left, right], ignore_index=True)
    df["body_side"] = df["body_side"].replace({"左": "左手側", "右": "右手側", "左側": "左手側", "右側":"右手側"})
    for c in FIELDS:
        if c not in df.columns:
            df[c] = np.nan
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    for c in ("manual_forearm_deg", "manual_neck_deg", "baseline_forearm_deg", "baseline_neck_deg"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df["time_sec"].isna().any() or (df["time_sec"] < 0).any():
        raise ValueError("time_sec 包含非數值或負數")
    if not df["body_side"].isin(SIDES).all():
        raise ValueError("body_side 必須為『左手側』或『右手側』")
    if df.duplicated(["sample_id","body_side"]).any():
        raise ValueError("sample_id + body_side 必須唯一")
    if len(df) > 160:
        raise ValueError("單次最多 160 筆評估側（建議先測試 10 筆）")
    return df[list(FIELDS)].reset_index(drop=True)


def make_template(n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(1, n+1):
        for side in SIDES:
            rows.append({"sample_id":i,"time_sec":float(i),"body_side":side,
                         "manual_forearm_deg":np.nan,"manual_neck_deg":np.nan,
                         "baseline_forearm_deg":np.nan,"baseline_neck_deg":np.nan})
    return pd.DataFrame(rows,columns=FIELDS)
