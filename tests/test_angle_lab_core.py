import math
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from angle_lab_core import (angle2d,candidates_from_landmarks,refine_relative_projection,
                            choose_candidate,evaluate_paired,parse_rows,make_template,SIDE_INDICES)


def pts():
    raw={
        7:(.35,.18,.97),11:(.35,.32,.97),13:(.32,.49,.96),15:(.27,.61,.96),23:(.36,.68,.95),
        8:(.66,.18,.91),12:(.66,.32,.95),14:(.75,.49,.95),16:(.82,.62,.94),24:(.66,.68,.93),
    }
    return {k:{'x':v[0],'y':v[1],'visibility':v[2]} for k,v in raw.items()}


def test_angles():
    assert angle2d([1,0],[0,0],[0,1])==pytest.approx(90)
    assert angle2d([0,0],[0,0],[1,1])!=angle2d([0,0],[0,0],[1,1])  # nan
    assert angle2d([-1,0],[0,0],[1,0])==pytest.approx(180)


def test_same_anatomical_side_across_cameras():
    a=pts()
    b={k:dict(v) for k,v in a.items()}
    # A change in RIGHT elbow should not change LEFT forearm angle
    b[14]['x']+=.08
    la=candidates_from_landmarks(a,640,480,'Left','左手側')
    rb=candidates_from_landmarks(b,640,480,'Right','左手側')
    assert la[0]['angle_deg']==rb[0]['angle_deg']
    assert la[0]['angle_deg']!=candidates_from_landmarks(b,640,480,'Right','右手側')[0]['angle_deg']


def test_neck_matches_v114_ear_shoulder_hip():
    p=pts()
    # Near-profile shoulder pair is projected close together.
    p[12]["x"] = .38
    c=candidates_from_landmarks(p,640,480,'Left','左手側')[1]
    assert c['angle_deg'] >= 0 and c['angle_deg']<=85
    assert c['eligible'] is True


def test_front_neck_is_not_eligible_as_sagittal_replacement():
    p=pts()
    c=candidates_from_landmarks(p,640,480,'Front','左手側')[1]
    assert c['eligible'] is False
    assert '正面' in c['warnings']


def test_visibility_low_cannot_qualify():
    p=pts();p[13]['visibility']=.2
    c=candidates_from_landmarks(p,640,480,'Right','左手側')[0]
    assert not c['eligible']


def test_no_qualified_angle_no_substitution():
    bad=[{'body_side':'左手側','part':'前臂','camera':cam,'eligible':False,
           'angle_deg':float('nan'),'quality':0} for cam in ('Left','Front','Right')]
    ch=choose_candidate(bad,'左手側','前臂')
    assert ch['selected_camera']=='無'
    assert math.isnan(ch['selected_angle_deg'])


def test_select_alternative_of_same_side_only():
    candidates=[]
    for cam,q in [('Left',.52),('Front',.71),('Right',.92)]:
        candidates.append({'body_side':'左手側','part':'前臂','camera':cam,
                           'eligible':cam!='Left','angle_deg':97.0 if cam=='Right' else 45.9,
                           'quality':q})
    # Decoy right anatomical side should never be selected
    candidates.append({'body_side':'右手側','part':'前臂','camera':'Left','eligible':True,
                       'angle_deg':12,'quality':1.0})
    ch=choose_candidate(candidates,'左手側','前臂')
    assert ch['selected_camera']=='Right'
    assert ch['selected_angle_deg']==97
    assert '人工' in ch['decision_note']


def test_quality_shortens_forearm_projection_cannot_claim_exactness():
    p=pts()
    c=[]
    for cam in ('Left','Front','Right'):
        c.extend(candidates_from_landmarks(p,640,480,cam,'左手側'))
    changed=refine_relative_projection(c)
    assert len(changed)==6
    assert all(isinstance(x['eligible'],bool) for x in changed)


def test_parse_two_sides_and_reject_duplicate():
    src=pd.DataFrame({'sample_id':[18],'time_sec':[9.4]})
    d=parse_rows(src)
    assert len(d)==2
    assert set(d.body_side)=={'左手側','右手側'}
    with pytest.raises(ValueError,match='唯一'):
        parse_rows(pd.DataFrame({'sample_id':[1,1],'time_sec':[1,1], 'body_side':['左手側','左手側']}))


def test_mae_uses_only_complete_paired_rows():
    df=pd.DataFrame({'manual_forearm_deg':[100.,120.,80.],
                     'baseline_forearm_deg':[45.9,np.nan,82.],
                     'selected_forearm_deg':[90.,116.,np.nan],
                     'manual_neck_deg':[20.,30.,np.nan],
                     'baseline_neck_deg':[30.,50.,20.],
                     'selected_neck_deg':[22.,35.,20.]})
    ans=evaluate_paired(df)
    f=ans[ans['部位']=='前臂'].iloc[0]
    n=ans[ans['部位']=='頸部'].iloc[0]
    assert f['配對樣本數']==1
    assert f['原始MAE(度)']==54.1
    assert f['候選MAE(度)']==10.0
    assert n['配對樣本數']==2
    assert n['原始MAE(度)']==15.0
    assert n['候選MAE(度)']==3.5


def test_rula_angle_is_only_2d_proxy_and_template():
    assert len(make_template(15))==30
    for side in ('左手側','右手側'):
        assert SIDE_INDICES[side]['shoulder']!=SIDE_INDICES[('右手側' if side=='左手側' else '左手側')]['shoulder']


def test_manual_elbow_interior_is_normalized_before_mae():
    df=pd.DataFrame({'manual_forearm_deg':[100.],
                     'baseline_forearm_deg':[45.9],
                     'selected_forearm_deg':[80.]})
    from angle_lab_core import evaluate_paired
    row=evaluate_paired(df,forearm_manual_kind='interior').iloc[0]
    assert row['原始MAE(度)']==34.1  # 180-100=80, not direct 100-45.9=54.1
    assert row['候選MAE(度)']==0.0
    no=evaluate_paired(df,forearm_manual_kind='unknown').iloc[0]
    assert no['配對樣本數']==0


def test_forearm_angle_is_180_minus_interior_to_match_222py():
    p=pts()
    c=candidates_from_landmarks(p,640,480,'Right','右手側')[0]
    assert c['angle_deg']==pytest.approx(abs(180-c['interior_angle_deg']),abs=.02)


def test_legacy_neck_is_extra_diagnostic_not_automatic_replacement():
    p=pts()
    p[2]={'x':.34,'y':.15,'visibility':.95}
    p[5]={'x':.66,'y':.15,'visibility':.91}
    c=candidates_from_landmarks(p,640,480,'Left','左手側')[1]
    assert 'neck_legacy_proxy_deg' in c
    assert math.isfinite(c['neck_legacy_proxy_deg'])
    assert c['part']=='頸部'
