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
            "neck_legacy_proxy_deg": float("nan"), "geometry_ok": False,
            "view_applicable": False, "view_status": "未判定", "view_reason": "",
            "cross_view_conflict": False, "relative_projection": float("nan"),
            "shoulder_line_visibility": float("nan")}


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

    shoulder_line_vis = min(sh1[1], sh2[1]) if sh1 is not None and sh2 is not None else float("nan")
    # This is a conservative camera-orientation screening proxy, NOT proof that an
    # individual forearm/neck is moving inside the image plane.
    # Treat unknown camera orientation as UNKNOWN, never silently as PASS.
    if camera == "Front":
        view_ok = False
        view_status = "正面僅供診斷"
        view_reason = "前臂屈伸與頸部前屈不得僅依正面2D投影自動替換；需經額外運動平面驗證"
    elif not math.isfinite(shoulder_width_ratio) or not math.isfinite(shoulder_line_vis):
        view_ok = False
        view_status = "視角不明"
        view_reason = "雙肩投影不足，無法排除軀幹旋轉或遮擋"
    elif shoulder_line_vis < .35:
        view_ok = False
        view_status = "轉向證據不足"
        view_reason = "其中一側肩點可見度太低，肩寬投影無法作為方向參考"
    elif shoulder_width_ratio > .55:
        view_ok = False
        view_status = "疑似偏離正側面"
        view_reason = f"肩寬／軀幹投影比 {shoulder_width_ratio:.2f} > 探索門檻0.55"
    else:
        view_ok = True
        view_status = "側向初篩通過"
        view_reason = "肩線投影較窄；仍無法單靠2D確認前臂／頸部真正運動平面"

    result = []
    for part in PARTS:
        c = _candidate_base(camera, side, part)
        c["shoulder_width_torso_ratio"] = shoulder_width_ratio
        c["shoulder_line_visibility"] = round(shoulder_line_vis, 3) if math.isfinite(shoulder_line_vis) else float("nan")
        c["view_applicable"] = view_ok
        c["view_status"] = view_status
        c["view_reason"] = view_reason
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
            if camera == "Front":
                warnings.append("正面前臂角只保留數值供觀察，不推薦為屈伸角度替代來源")
        else:
            c["neck_legacy_proxy_deg"] = legacy_neck_proxy(lm,width,height,side)
            head_len = float(np.linalg.norm(xyz[2]-xyz[1]))
            ratio = head_len / max(torso, 1e-6)
            c["projected_length_ratio"] = round(ratio, 3)
            geometry_ok = (math.isfinite(torso) and torso >= 28 and head_len >= 8 and .08 <= ratio <= .92 and angle <= 85)
            if not geometry_ok:
                warnings.append("頸部參考線尺度／角度可疑")
            if camera == "Front":
                warnings.append("正面耳肩髖角不直接用於矢狀面頸部前屈角度替換")
        c["geometry_ok"] = bool(geometry_ok)
        # Landmark/segment quality is separate from movement-plane applicability.
        # For example Front may have 0.997 landmark quality yet be rejected as
        # a sagittal elbow/neck replacement.
        c["quality"] = round(0.65 * vis + (0.35 if geometry_ok else 0.0), 3)
        c["eligible"] = bool(vis >= .55 and geometry_ok and math.isfinite(angle) and view_ok)
        if not view_ok:
            warnings.append(view_reason)
        c["warnings"] = "；".join(warnings) if warnings else "關鍵點與側向初篩通過；不代表實際角度準確"
        result.append(c)
    return result


def refine_relative_projection(candidates: list[dict]) -> list[dict]:
    """Only compare side-camera projections, never use a frontal length to veto a side view.

    Relative length is a *warning* proxy. It cannot infer out-of-plane arm rotation
    from a single camera. A questionable side image never becomes reliable merely
    because the other camera also looks questionable.
    """
    copied = [dict(c, relative_projection=c.get("relative_projection", float("nan"))) for c in candidates]
    by_part_side = {(x["body_side"], x["part"]) for x in copied}
    for side, part in by_part_side:
        if part != "前臂":
            continue
        subset = [x for x in copied if x["body_side"] == side and x["part"] == part]
        lengths = [x["projected_length_ratio"] for x in subset
                   if x["camera"] in ("Left", "Right") and x["eligible"]
                   and math.isfinite(x.get("projected_length_ratio", float("nan")))]
        if len(lengths) < 2 or max(lengths) <= .01:
            continue
        max_len = max(lengths)
        for x in subset:
            if x["camera"] not in ("Left", "Right"):
                continue
            length=x.get("projected_length_ratio", float("nan"))
            ratio = length / max_len if math.isfinite(length) else float("nan")
            x["relative_projection"] = round(ratio, 3) if math.isfinite(ratio) else float("nan")
            if math.isfinite(ratio) and ratio < .60:
                x["eligible"] = False
                x["warnings"] += "；對另一合格側面鏡頭，前臂投影顯著縮短（僅探索性檢查）"
            elif math.isfinite(ratio) and ratio < .78:
                x["warnings"] += "；與另一側面相比前臂投影偏短（留意，不自動扣準確率）"
    return copied


def refine_neck_cross_view_conflicts(candidates: list[dict], threshold_deg: float = 15.0) -> list[dict]:
    """Do not choose a 'winner' when two initially admissible side views disagree.

    This is a conflict alarm, not proof that either view is inaccurate. The
    ear/shoulder/hip proxy is inherently projection dependent.
    """
    if threshold_deg <= 0:
        raise ValueError("threshold_deg must be positive")
    copied=[dict(c) for c in candidates]
    for side in SIDES:
        possible=[c for c in copied if c["body_side"]==side and c["part"]=="頸部"
                  and c["camera"] in ("Left", "Right") and c["eligible"]
                  and math.isfinite(c["angle_deg"])]
        if len(possible)!=2:
            continue
        gap=abs(possible[0]["angle_deg"]-possible[1]["angle_deg"])
        if gap >= threshold_deg:
            for c in possible:
                c["eligible"]=False
                c["cross_view_conflict"]=True
                c["warnings"] += f"；兩側鏡頭對同一人體側別的頸角差 {gap:.1f}°，暫停自動選角"
    return copied


def choose_candidate(candidates: list[dict], side: str, part: str) -> dict:
    """Conservative research suggestion; frontal sagittal angle NEVER selected.

    Keep valid original side camera even if another side camera has a higher
    landmark-quality score. Other side camera is used only when original fails,
    and always labeled as an unverified research candidate.
    """
    base_cam = "Left" if side == "左手側" else "Right"
    opposite_cam = "Right" if base_cam == "Left" else "Left"
    subset = [c for c in candidates if c["body_side"] == side and c["part"] == part]
    base = next((c for c in subset if c["camera"] == base_cam), None)
    alternative = next((c for c in subset if c["camera"] == opposite_cam), None)
    valid = lambda c: (c is not None and c.get("eligible", False)
                       and c.get("view_applicable", True)
                       and math.isfinite(c.get("angle_deg", float("nan"))))
    choice = base if valid(base) else (alternative if valid(alternative) else None)
    state = "無合格側向視角"
    caution = "兩側鏡頭未通過初步檢查或出現角度衝突；保持角度不確定，需人工或校正3D資料確認"
    if choice:
        state = "維持原側向視角" if choice["camera"] == base_cam else "研究候選：另一側鏡頭（待人工驗證）"
        caution = "關鍵點品質與側向初篩通過；尚未證實前臂／頸部的實際運動平面，不是量測準確率"
        if choice["camera"] != base_cam:
            caution += "；原鏡頭不合格，改列另一側鏡頭的同一人體側別角度為研究候選，並非正式自動替換"
        if base is not None and math.isfinite(base["angle_deg"]) and choice["camera"] != base_cam:
            diff = abs(choice["angle_deg"] - base["angle_deg"])
            if diff >= 15:
                caution += f"；與原視角差 {diff:.1f}°，需人工檢查原影格"
        if part == "頸部":
            caution += "；耳肩髖2D代理角不等於解剖頸椎前屈角"
    elif any(c.get("cross_view_conflict") for c in subset):
        state = "頸部跨鏡頭分歧（待複核）"
        caution = "兩個初步合格側面鏡頭的頸角差超過探索門檻；不能靠可見度決定哪個正確"
    return {"body_side": side, "part": part, "baseline_camera": base_cam,
            "baseline_angle_deg": base["angle_deg"] if base else float("nan"),
            "selected_camera": choice["camera"] if choice else "無",
            "selected_angle_deg": choice["angle_deg"] if choice else float("nan"),
            "selected_quality": choice["quality"] if choice else float("nan"),
            "decision": state, "decision_note": caution}


def flag_shared_neck_conflicts(summary: pd.DataFrame, threshold_deg: float=15.0) -> pd.DataFrame:
    """Flag opposite anatomical sides at the same time if neck proxies diverge.

    Neck posture is shared but opposite ears and camera projections differ.
    Do NOT average the two or decide which angle is correct from this test.
    """
    result=summary.copy()
    result["neck_opposite_side_gap_deg"]=np.nan
    result["neck_shared_conflict"]=False
    result["neck_before_shared_check_deg"]=result["selected_neck_deg"].copy()
    result["neck_source_before_shared_check"]=result["source_neck"].copy()
    if result.empty:
        return result
    result["_time_key"]=pd.to_numeric(result["time_sec"],errors="coerce").round(4)
    for (_, _), grp in result.groupby(["sample_id","_time_key"], dropna=False):
        sides=set(grp["body_side"])
        if not set(SIDES).issubset(sides):
            continue
        la=grp[grp["body_side"]=="左手側"].iloc[0]
        ra=grp[grp["body_side"]=="右手側"].iloc[0]
        left=float(la["lab_base_neck_deg"])
        right=float(ra["lab_base_neck_deg"])
        if not (math.isfinite(left) and math.isfinite(right)):
            continue
        gap=abs(left-right)
        result.loc[grp.index,"neck_opposite_side_gap_deg"]=round(gap,2)
        if gap >= threshold_deg:
            result.loc[grp.index,"neck_shared_conflict"]=True
            result.loc[grp.index,"selected_neck_deg"]=np.nan
            result.loc[grp.index,"source_neck"]="無（跨側分歧）"
            result.loc[grp.index,"quality_neck"]=np.nan
            result.loc[grp.index,"decision_neck"]="頸部左右側分歧（待複核）"
            result.loc[grp.index,"reason_neck"]=(
                f"同時間左／右側基準頸部代理角相差 {gap:.1f}°（探索警戒值{threshold_deg:.0f}°）；"
                "各側耳肩髖投影可能不同，必須確認同步、轉向、關鍵點及角度基準；不採用任一候選")
    return result.drop(columns="_time_key")


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
