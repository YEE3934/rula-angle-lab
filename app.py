"""RULA Angle Lab | separate experimental Streamlit app. NOT official RULA output."""
from __future__ import annotations

import hashlib
import io
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from angle_lab_core import (
    CAMERAS, PARTS, SIDES, SIDE_INDICES, candidates_from_landmarks,
    choose_candidate, evaluate_paired, make_template, parse_rows,
    refine_relative_projection,
)

st.set_page_config(page_title="RULA Angle Lab｜跨視角量角研究", page_icon="🔬", layout="wide")
st.title("🔬 RULA Angle Lab｜跨視角角度測試")
st.caption("獨立研究工具 v1.0｜同側前臂／頸部 2D 角度比較，不會修改正式 RULA 系統或自動更動 Grand Score")
st.warning("這是測試版：視角品質門檻是尚待驗證的工程假設。候選視角 ≠ 真實 3D 關節角度；研究結果不可直接宣稱為官方 RULA 分數。", icon="⚠️")

LABELS = {"Left": "左側鏡頭 Left", "Front": "正面鏡頭 Front", "Right": "右側鏡頭 Right"}


def _fingerprint(upload):
    mv = upload.getbuffer()
    h = hashlib.sha256()
    h.update(bytes(mv[:min(len(mv),65536)]))
    h.update(bytes(mv[max(0,len(mv)-65536):]))
    h.update(str(len(mv)).encode())
    return f"{upload.name}_{len(mv)}_{h.hexdigest()[:16]}"


def materialize_video(upload, cam):
    key = _fingerprint(upload)
    state = st.session_state.setdefault("video_files", {})
    old = state.get(cam)
    if old and old["signature"] == key and os.path.isfile(old["path"]):
        return old["path"]
    if old:
        try: os.remove(old["path"])
        except OSError: pass
    ext = Path(upload.name).suffix.lower()
    if ext not in (".mp4", ".mov", ".avi", ".m4v"):
        ext = ".mp4"
    with tempfile.NamedTemporaryFile(prefix=f"rula_lab_{cam}_", suffix=ext, delete=False) as out:
        out.write(upload.getbuffer())
        path = out.name
    state[cam] = {"signature":key,"path":path}
    return path


def video_info(path):
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise RuntimeError("OpenCV 無法開啟這個檔案，請轉換成 H.264 MP4 再上傳")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps < 1 or count < 1:
            raise RuntimeError("無法取得有效 FPS 或影格數，請換成 H.264 MP4")
        return {"fps":fps,"count":count,"duration":count/fps,
                "width":int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height":int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def frame_at(cap, info, second, max_side=960):
    idx = int(round(float(second) * info["fps"]))
    if idx < 0 or idx >= info["count"]:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    h,w = frame.shape[:2]
    scale = min(1.,float(max_side)/max(w,h))
    if scale < 1:
        frame = cv2.resize(frame, (max(1,round(w*scale)),max(1,round(h*scale))), interpolation=cv2.INTER_AREA)
    return frame


def pose_from_frame(pose_model, frame):
    if frame is None:
        return None
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    result = pose_model.process(image_rgb)
    if result is None or result.pose_landmarks is None:
        return None
    return {i: {"x":float(pt.x),"y":float(pt.y),"visibility":float(pt.visibility)}
            for i,pt in enumerate(result.pose_landmarks.landmark)}


def overlay_bgr(frame, lm, side):
    if frame is None:
        return None
    draw = frame.copy()
    h,w=draw.shape[:2]
    ids = SIDE_INDICES[side]
    pairs = (("hip","shoulder"),("shoulder","ear"),("shoulder","elbow"),("elbow","wrist"))
    for a,b in pairs:
        if lm and ids[a] in lm and ids[b] in lm:
            p=lm[ids[a]]; q=lm[ids[b]]
            x=(round(p["x"]*w),round(p["y"]*h)); y=(round(q["x"]*w),round(q["y"]*h))
            cv2.line(draw,x,y,(220,180,50),2,cv2.LINE_AA)
    for key,idx in ids.items():
        if not lm or idx not in lm: continue
        pt=lm[idx]; xy=(round(pt["x"]*w),round(pt["y"]*h))
        cv2.circle(draw,xy,5,(60,225,105),-1,cv2.LINE_AA)
        cv2.putText(draw,f"{key}({idx})",(xy[0]+7,xy[1]-6),cv2.FONT_HERSHEY_SIMPLEX,.46,(245,245,245),2,cv2.LINE_AA)
    return cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)


def to_csv_bytes(df):
    return df.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig")


def make_zip(summary, detail, stats):
    mem=io.BytesIO()
    with zipfile.ZipFile(mem,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("analysis_summary.csv",to_csv_bytes(summary))
        z.writestr("all_camera_candidates.csv",to_csv_bytes(detail))
        z.writestr("paired_mae.csv",to_csv_bytes(stats))
        z.writestr("READ_ME.txt", "RULA Angle Lab v1.0 research output\n"
                   "This is a preliminary camera view selection heuristic, not calibrated 3D nor official RULA score.\n"
                   "The chosen camera always tracks the SAME anatomical side; body_side is fixed.\n"
                   "Neck values are abs(180-angle(hip,shoulder,ear)) 2D proxy, not anatomical cervical flexion.\n"
                   "Formal baseline angles must be entered independently from original system export.\n"
                   "MAE computed only from identical cases with manual, baseline and candidate values.\n"
                   "Projection / visibility scores are NOT probabilities of accuracy.\n"
                   "Do not use these numbers as confirmed ergonomic conclusions without manual verification.\n")
    return mem.getvalue()


with st.expander("研究設計與角度定義（先看）", expanded=False):
    st.markdown("""**人體側別固定**：左手側永遠使用 MediaPipe 左肩 11、左肘 13、左腕 15；右手側使用 12、14、16。更換鏡頭不更換評估側。\n
**前臂角度**：與正式程式一致，`abs(180-angle(肩,肘,腕))` 是屈曲代理角；同時保存肘內角供人工核對。**頸部角度**：沿用正式 `222.py` v11.4 的 `abs(180 - angle(髖, 肩, 耳))` 代理值，兩者都不是經校正的 3D 解剖真值。\n
**跨視角門檻**：MediaPipe 可見度、肢段投影長度／比例、側視肩寬比。候選分數只是工程性品質指標，不能反推準確率。正面頸部前屈不自動取代側面。\n
**研究比較**：請填入獨立人工角度以及「正式 RULA 主系統輸出的原始角度」。只有同一筆樣本的三種角度均存在才納入成對 MAE；沒有真值的樣本保留為待補。""")

st.subheader("① 上傳三支同期影片")
cols=st.columns(3)
uploads={}
for col,cam in zip(cols,CAMERAS):
    with col:
        uploads[cam]=st.file_uploader(LABELS[cam],type=["mp4","mov","avi","m4v"],key=f"video_{cam}")

st.caption("建議 MP4/H.264，短片先測試；每支影片通常不超過 200MB。三個影片必須拍同一次動作。不要把左側鏡頭誤認為人體左手。")
with st.expander("⚙️ 同步校正／影片鏡像",expanded=True):
    st.write("時間公式：**鏡頭讀取時間＝參考時間＋偏移秒數**。例如右鏡頭的同一動作比參考時間晚 0.30 秒才出現，就填 +0.30。")
    cols=st.columns(3)
    offsets={}; mirrors={}
    for col,cam in zip(cols,CAMERAS):
        with col:
            offsets[cam]=st.number_input(f"{cam} 偏移（秒）",value=0.0,step=.05,format="%.2f",key=f"offset_{cam}")
            mirrors[cam]=st.checkbox(f"{cam} 影片有鏡像（輸入時反轉還原）",value=False,key=f"mirror_{cam}")
    st.info("鏡像選項只在確定原影片遭水平鏡像時使用。若左右標記反了，先修正來源或確認鏡像，不能偷偷將左手分數改成右手。")

paths={}; infos={}; video_errors=[]
for cam,upload in uploads.items():
    if upload:
        try:
            paths[cam]=materialize_video(upload,cam)
            infos[cam]=video_info(paths[cam])
        except Exception as exc:
            video_errors.append(f"{cam}: {exc}")
if video_errors:
    st.error("；".join(video_errors))
if len(infos)==3:
    st.dataframe(pd.DataFrame([{ "鏡頭":LABELS[cam],"時長(秒)":round(m["duration"],2),"FPS":round(m["fps"],2),
                                "畫素":f'{m["width"]}×{m["height"]}'} for cam,m in infos.items()]),hide_index=True,use_container_width=True)

st.subheader("② 選擇要測試的時間點")
mode=st.radio("測試方式",["單一時間點","等間隔批次","匯入 CSV 樣本與人工角度"],horizontal=True)
reference=st.number_input("參考時間（秒，從影片開始算）",min_value=0.0,value=5.0,step=.2,format="%.2f")
rows=None
if mode=="單一時間點":
    rows=make_template(1)
    rows["time_sec"]=reference
elif mode=="等間隔批次":
    c1,c2,c3=st.columns(3)
    with c1: begin=st.number_input("起始秒數",min_value=0.,value=1.,step=.5)
    with c2: finish=st.number_input("結束秒數",min_value=0.,value=12.,step=.5)
    with c3: n=st.number_input("時間點數量",min_value=2,max_value=60,value=5,step=1)
    if finish > begin:
        rows=make_template(n)
        rows["time_sec"]=np.repeat(np.linspace(begin,finish,n),2)
    else:
        st.error("結束秒數必須大於起始秒數")
else:
    st.download_button("下載人工／系統比對 CSV 範本",to_csv_bytes(make_template(5)),
                       file_name="angle_lab_template.csv",mime="text/csv")
    input_csv=st.file_uploader("上傳樣本 CSV（必須包含 time_sec；可填人工與原系統角度）",type=["csv"],key="lab_csv")
    if input_csv:
        try:
            try: df=pd.read_csv(input_csv,encoding="utf-8-sig")
            except UnicodeDecodeError:
                input_csv.seek(0)
                df=pd.read_csv(input_csv,encoding="cp950")
            rows=parse_rows(df)
            st.success(f"讀取 {len(rows)} 筆側別評估資料")
        except Exception as exc:
            st.error(f"樣本 CSV 格式錯誤：{exc}")

if rows is not None:
    st.caption(f"目前準備測試：{len(rows)} 筆人體側別，{rows['time_sec'].nunique()} 個取樣時間（每個時間點可能包含左右手）。")

if len(infos)==3:
    if st.button("👀 預覽三視角同步畫面",type="secondary"):
        cols=st.columns(3)
        for col,cam in zip(cols,CAMERAS):
            cap=cv2.VideoCapture(paths[cam])
            try: frame=frame_at(cap,infos[cam],reference+offsets[cam])
            finally: cap.release()
            with col:
                st.markdown(f"**{LABELS[cam]}**")
                if frame is None:st.warning("超出時間範圍或無法讀取")
                else:
                    if mirrors[cam]: frame=cv2.flip(frame,1)
                    st.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),use_container_width=True)
        st.caption("請先確認三張影像是在做同一個動作。如果不同步，調整上面的秒數偏移再預覽。")

run_disabled = (len(infos)!=3 or rows is None or bool(video_errors))
if st.button("🚀 執行跨視角角度比較",type="primary",disabled=run_disabled):
    import mediapipe as mp
    if not hasattr(mp,"solutions") or not hasattr(mp.solutions,"pose"):
        st.error("MediaPipe 不支援 mp.solutions.pose；請核對 requirements.txt 的 mediapipe==0.10.21")
        st.stop()
    scan_rows=[]; scan_details=[]; captures={cam:cv2.VideoCapture(paths[cam]) for cam in CAMERAS}
    row_records=rows.to_dict("records")
    cache={}
    prog=st.progress(0,text="初始化預訓練 Pose 模型（只做2D候選比較）")
    try:
        with mp.solutions.pose.Pose(static_image_mode=True,model_complexity=1,
                                    enable_segmentation=False,min_detection_confidence=.50) as pose:
            for i,record in enumerate(row_records):
                side=record["body_side"]; sec=float(record["time_sec"])
                candidates=[]
                for cam in CAMERAS:
                    key=(cam,round(sec,4))
                    if key not in cache:
                        current_time=sec+offsets[cam]
                        f=frame_at(captures[cam],infos[cam],current_time)
                        if f is not None and mirrors[cam]: f=cv2.flip(f,1)
                        lm=pose_from_frame(pose,f)
                        wh=(f.shape[1],f.shape[0]) if f is not None else (1,1)
                        cache[key]=(lm,wh)
                    lm,(w,h)=cache[key]
                    candidates.extend(candidates_from_landmarks(lm,w,h,cam,side))
                candidates=refine_relative_projection(candidates)
                for cand in candidates:
                    scan_details.append({"sample_id":record["sample_id"],"time_sec":sec,**cand})
                decisions={part:choose_candidate(candidates,side,part) for part in PARTS}
                output=dict(record)
                for part,prefix in (("前臂","forearm"),("頸部","neck")):
                    choice=decisions[part]
                    output.update({f"lab_base_{prefix}_deg":choice["baseline_angle_deg"],
                                   f"selected_{prefix}_deg":choice["selected_angle_deg"],
                                   f"source_{prefix}":choice["selected_camera"],
                                   f"quality_{prefix}":choice["selected_quality"],
                                   f"decision_{prefix}":choice["decision"],
                                   f"reason_{prefix}":choice["decision_note"]})
                scan_rows.append(output)
                prog.progress((i+1)/len(row_records),text=f"已完成 {i+1} / {len(row_records)} 筆側別")
    except Exception as exc:
        st.exception(exc)
        st.error("分析未完成；保留正式 RULA 系統不受影響。建議先測試單一時間點及較小影片。")
    else:
        st.session_state["lab_result"]={"summary":pd.DataFrame(scan_rows),
            "details":pd.DataFrame(scan_details),"landmarks":cache,
            "offsets":dict(offsets),"mirrors":dict(mirrors),
            "signatures":{cam:_fingerprint(uploads[cam]) for cam in CAMERAS}}
        st.success(f"分析完成：{len(scan_rows)} 筆側別 × 2個部位，提供三鏡頭角度候選。")
    finally:
        for cap in captures.values():cap.release()

if "lab_result" in st.session_state:
    data=st.session_state["lab_result"]
    if any(cam not in uploads or uploads[cam] is None or _fingerprint(uploads[cam])!=data["signatures"].get(cam) for cam in CAMERAS):
        st.warning("影片來源已變動，舊分析結果僅供參考。請重新分析新影片。")
    if any(abs(offsets[cam]-data["offsets"][cam])>1e-6 or mirrors[cam]!=data["mirrors"][cam] for cam in CAMERAS):
        st.warning("同步偏移或鏡像設定已修改，下面仍是舊設定結果，需重新執行分析。")
    summary=data["summary"]
    details=data["details"]
    tabs=st.tabs(["🖼 三視角骨架檢視","📐 角度與視角建議","🧮 人工角度 MAE","📦 匯出結果"])
    with tabs[0]:
        index=st.selectbox("選擇樣本與人體評估側",options=list(summary.index),
                           format_func=lambda i:f"第 {summary.loc[i,'sample_id']} 筆｜{summary.loc[i,'body_side']}｜{summary.loc[i,'time_sec']:.2f}秒")
        row=summary.loc[index]; side=row["body_side"]; sec=float(row["time_sec"])
        cols=st.columns(3)
        for col,cam in zip(cols,CAMERAS):
            current_time=sec+data["offsets"][cam]
            cap=cv2.VideoCapture(paths[cam]) if cam in paths else None
            try: frame=frame_at(cap,infos[cam],current_time) if cap and cam in infos else None
            finally:
                if cap:cap.release()
            if frame is not None and data["mirrors"][cam]: frame=cv2.flip(frame,1)
            key=(cam,round(sec,4))
            lm=data["landmarks"].get(key,(None,None))[0]
            with col:
                st.markdown(f"**{LABELS[cam]}｜人體{side}**")
                if frame is not None: st.image(overlay_bgr(frame,lm,side),use_container_width=True)
                else: st.warning("無對應影格")
                for part in PARTS:
                    matched=details[(details["sample_id"]==row["sample_id"]) & (details["body_side"]==side) & (details["part"]==part) & (details["camera"]==cam)]
                    if not matched.empty:
                        item=matched.iloc[0]
                        st.caption(f"{part}：{item['angle_deg'] if np.isfinite(item['angle_deg']) else '無效'}°｜品質 {item['quality']:.2f}｜{'候選通過' if item['eligible'] else '不採用'}")
                        if part=="頸部" and np.isfinite(item.get("neck_legacy_proxy_deg",np.nan)):
                            st.caption(f"舊版眼耳＋軀幹中心線代理角 {item['neck_legacy_proxy_deg']:.1f}°（只供研究對照）")
                        if part=="前臂" and np.isfinite(item.get("interior_angle_deg",np.nan)):
                            st.caption(f"肘內角 {item['interior_angle_deg']:.1f}°（正式系統用的屈曲代理角是上方數值）")
        st.caption("骨架點是模型估測座標；同一個人體側別在不同鏡頭會有不同2D投影，且不保證任何一個是3D真值。")
    with tabs[1]:
        showcols=["sample_id","time_sec","body_side","lab_base_forearm_deg","selected_forearm_deg","source_forearm","decision_forearm",
                  "lab_base_neck_deg","selected_neck_deg","source_neck","decision_neck"]
        st.dataframe(summary[showcols],hide_index=True,use_container_width=True)
        st.caption("頸部各鏡頭另提供『舊版眼耳＋軀幹中心線代理角』作為 C 方法的參考線診斷，但**不會自動拿舊版角替換正式分數**。")
        with st.expander("各個鏡頭的完整品質診斷／所有候選角度"):
            st.dataframe(details,use_container_width=True,hide_index=True)
        st.info("『研究候選：跨鏡頭』不等於自動正確，不應拿其他攝影機的人體右側角度替換左側。此頁僅提供可檢驗候選，不直接修改正式系統分數。")
    with tabs[2]:
        st.write("在下表填上人工角度與**正式主系統原本輸出的角度**。若先前從 CSV 匯入，已經帶入；缺值維持空白，不能當0度。")
        definition=st.selectbox("你人工畫的前臂角度到底是哪種？（所有樣本必須同一規則）",
          ["尚未確認，先不算前臂MAE", "肘內角：肩－肘－腕畫出來的角（例如第18張100°）", "屈曲代理角：180°減肘內角（與222.py相同）"],
          index=0, key="manual_forearm_definition")
        manual_kind=("interior" if definition.startswith("肘內角") else "flexion" if definition.startswith("屈曲") else "unknown")
        st.caption("例如人工肘內角100°→系統屈曲代理角80°，不能直接把100°與系統45.9°相減；本程式不會偷偷修改你填的原始人工值。")
        columns=["sample_id","time_sec","body_side","manual_forearm_deg","baseline_forearm_deg","lab_base_forearm_deg","selected_forearm_deg","source_forearm",
                 "manual_neck_deg","baseline_neck_deg","lab_base_neck_deg","selected_neck_deg","source_neck"]
        edited=st.data_editor(summary[columns],key="angle_lab_editor",hide_index=True,use_container_width=True,
                              disabled=[k for k in columns if k not in ("manual_forearm_deg","baseline_forearm_deg","manual_neck_deg","baseline_neck_deg")],
                              num_rows="fixed")
        for c in ("manual_forearm_deg","baseline_forearm_deg","manual_neck_deg","baseline_neck_deg"):
            summary[c]=pd.to_numeric(edited[c],errors="coerce")
        data["summary"]=summary
        stats=evaluate_paired(summary,forearm_manual_kind=manual_kind)
        st.dataframe(stats,hide_index=True,use_container_width=True)
        st.caption("前臂人工定義："+definition+"。以上『原始MAE』使用你輸入的正式系統原始角度；『候選MAE』使用本實驗的視角候選。所有比較嚴格使用三者都有值的同一批紀錄，避免因排除高誤差影格而虛假改善。")
        if summary["manual_neck_deg"].notna().any() or summary["manual_forearm_deg"].notna().any():
            extra=[]
            for label,stem in (("頸部","neck"),("前臂","forearm")):
                a=summary[f"manual_{stem}_deg"];b=summary[f"lab_base_{stem}_deg"]
                if stem=="forearm":
                    a=(180-a) if manual_kind=="interior" else (pd.Series(np.nan,index=a.index) if manual_kind=="unknown" else a)
                m=a.notna() & b.notna()
                extra.append({"部位":label,"Lab同鏡頭可比筆數":int(m.sum()),
                              "Lab原視角MAE(度)":round(float((a[m]-b[m]).abs().mean()),3) if m.any() else float('nan')})
            st.markdown("**額外診斷：Lab重算原視角 vs 人工**（不是主系統正式原角度）")
            st.dataframe(pd.DataFrame(extra),hide_index=True)
        st.warning("頸部的『人工角度』若採不同解剖參考線／不同角度方向，不得直接算 MAE；請先統一耳－肩－髖2D代理角的定義。")
    with tabs[3]:
        definition=st.session_state.get("manual_forearm_definition","尚未確認")
        manual_kind=("interior" if definition.startswith("肘內角") else "flexion" if definition.startswith("屈曲") else "unknown")
        stats=evaluate_paired(summary,forearm_manual_kind=manual_kind)
        summary=summary.copy()
        summary["manual_forearm_definition"]=definition
        summary["manual_forearm_converted_deg"]=180-summary["manual_forearm_deg"] if manual_kind=="interior" else (summary["manual_forearm_deg"] if manual_kind=="flexion" else np.nan)
        st.download_button("📦 下載研究結果 ZIP",data=make_zip(summary,details,stats),
                           file_name="RULA_Angle_Lab_研究結果.zip",mime="application/zip",type="primary")
        st.download_button("下載比較總表 CSV",data=to_csv_bytes(summary),
                           file_name="angle_lab_comparison.csv",mime="text/csv")
        st.caption("ZIP 不包含影片與原始影像，只包含角度、選擇來源、品質診斷與成對 MAE。研究時請自行保存同時間戳對照照片。")

st.divider()
st.caption("RULA Angle Lab v1.0 | MediaPipe Legacy Pose (33 points) · Python · OpenCV · Streamlit | Independent research prototype")
