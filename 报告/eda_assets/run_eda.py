from pathlib import Path
import json,csv,math
from collections import Counter,defaultdict
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(r"F:\wangxinbei")
DATA=ROOT/"data"/"B01赛题初赛数据集-stage1（仅供报名审核通过队伍下载）"/"B01-初赛数据集-stage1-发布版"
OUT=ROOT/"报告"/"eda_assets"; OUT.mkdir(parents=True,exist_ok=True)

def jl(path):
    a=[]; bad=[]
    for i,line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(),1):
        if not line.strip(): continue
        try:a.append(json.loads(line))
        except Exception as e:bad.append([i,str(e)])
    return a,bad
def v(x): return list(x.values())
def qs(x):
    a=np.asarray([float(z) for z in x if z is not None and math.isfinite(float(z))])
    if not len(a): return {}
    return {k:float(getattr(np,k)(a)) for k in ["min","mean","max"]}|{f"p{p}":float(np.percentile(a,p)) for p in [1,5,25,50,75,95,99]}|{"n":int(len(a))}

def tm(tr):
    vv=v(tr); pts=vv[-1]; p=[v(x) for x in pts]
    t=np.array([float(x[0]) for x in p]); lon=np.array([float(x[1]) for x in p]); lat=np.array([float(x[2]) for x in p])
    alt=np.array([float(x[3]) for x in p]); sp=np.array([float(x[4]) for x in p]); hd=np.array([float(x[5]) for x in p])
    d=np.diff(t)
    return {"id":str(vv[0]),"n":len(p),"duration":float(t.max()-t.min()),"rev":int((d<0).sum()),"zero":int((d==0).sum()),"dup":len(p)-len(set(tuple(x) for x in p)),
            "lon":lon.tolist(),"lat":lat.tolist(),"alt":alt.tolist(),"speed":sp.tolist(),"heading":hd.tolist(),
            "alt_stats":qs(alt),"speed_stats":qs(sp),"heading_stats":qs(hd),"dt":[float(x) for x in d]}
def pstats(rows):
    mm=[tm(v(r)[1]) for r in rows]
    return {"n":len(rows),"unique_traj":len({x["id"] for x in mm}),"points":qs([x["n"] for x in mm]),"duration":qs([x["duration"] for x in mm]),
            "altitude":qs([z for x in mm for z in x["alt"]]),"speed":qs([z for x in mm for z in x["speed"]]),
            "heading":qs([z for x in mm for z in x["heading"]]),"rev_questions":sum(x["rev"]>0 for x in mm),"rev_total":sum(x["rev"] for x in mm),
            "zero_questions":sum(x["zero"]>0 for x in mm),"dup_questions":sum(x["dup"]>0 for x in mm),"metrics":mm}

seg,bad_seg=jl(DATA/"task1"/"01_segment_questions.jsonl")
reg,bad_reg=jl(DATA/"task1"/"02_region_questions.jsonl")
ret,bad_ret=jl(DATA/"task2"/"01_retrieval_questions.jsonl")
S=pstats(seg); R=pstats(reg)
S["bad_lines"]=bad_seg; R["bad_lines"]=bad_reg

terms=["时间","高度","速度","区域","不满足","先","随后","盘旋","绕飞","巡航","直线","跑道","加速","减速","进入","离开","停留"]
def kw(rows,idx=-1):
    c=Counter()
    for r in rows:
        text=str(v(r)[idx])
        c.update({t:1 for t in terms if t in text})
    return dict(c)
S["keyword_counts"]=kw(seg); R["keyword_counts"]=kw(reg)
region_counts=[]; poly_vertices=[]; poly_area=[]
for r in reg:
    for z in v(r)[2]:
        coords=v(z)[-1]; region_counts.append(1); poly_vertices.append(len(coords))
        poly_area.append(abs(sum(coords[i][0]*coords[(i+1)%len(coords)][1]-coords[(i+1)%len(coords)][0]*coords[i][1] for i in range(len(coords)))/2))
R["regions_per_question"]=qs([len(v(r)[2]) for r in reg]); R["polygon_vertices"]=qs(poly_vertices); R["polygon_area_deg2"]=qs(poly_area)

RQ=[]; RK=Counter(); top=[]; nreg=[]
for r in ret:
    z=v(r); q=str(z[2]); RQ.append(q); top.append(int(z[3])); nreg.append(len(z[1])); RK.update({t:1 for t in terms if t in q})
TQ={"n":len(ret),"bad_lines":bad_ret,"query_chars":qs([len(x) for x in RQ]),"top_k":qs(top),"top_k_counts":dict(Counter(top)),"regions":qs(nreg),"region_counts":dict(Counter(nreg)),"keyword_counts":dict(RK)}

# CSV
def tsval(s):
    try:
        return float(s)
    except Exception:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
csvp=DATA/"task2"/"02_tracks.csv"; tracks=defaultdict(list); total=bad=0
with csvp.open(encoding="utf-8-sig",newline="") as f:
    rd=csv.reader(f); h=next(rd)
    def ci(words,default):
        for i,x in enumerate(h):
            if any(w in x.lower() for w in words): return i
        return default
    it=ci(["航迹","track","traj"],0); its=ci(["时间","time"],1); ilon=ci(["经度","lon"],2); ilat=ci(["纬度","lat"],3); ialt=ci(["高度","alt"],4); isp=ci(["速度","speed"],5); ih=ci(["航向","heading"],None)
    for row in rd:
        if not row: continue
        total+=1
        try: tracks[row[it]].append([tsval(row[its]),float(row[ilon]),float(row[ilat]),float(row[ialt]),float(row[isp]),float(row[ih]) if ih is not None else 0.0])
        except: bad+=1
M=[]; all_alt=[];all_sp=[];all_lon=[];all_lat=[];all_hd=[];all_dt=[]
for tid,p in tracks.items():
    if not p:
        continue
    p=sorted(p,key=lambda x:x[0]); t=np.array([x[0] for x in p]); d=np.diff(t)
    M.append({"track_id":tid,"n":len(p),"duration":float(t[-1]-t[0]),"zero":int((d==0).sum()),"dup":len(p)-len(set(tuple(x) for x in p)),
              "lon_min":min(x[1] for x in p),"lon_max":max(x[1] for x in p),"lat_min":min(x[2] for x in p),"lat_max":max(x[2] for x in p),
              "alt_mean":float(np.mean([x[3] for x in p])),"speed_mean":float(np.mean([x[4] for x in p]))})
    all_alt += [x[3] for x in p]; all_sp += [x[4] for x in p]; all_lon += [x[1] for x in p]; all_lat += [x[2] for x in p]; all_hd += [x[5] for x in p]; all_dt += [float(x) for x in d]
# original ordering quality
orig_rev=0; orig_rev_total=0; orig_zero=0
with csvp.open(encoding="utf-8-sig",newline="") as f:
    rd=csv.reader(f); h2=next(rd); groups=defaultdict(list)
    for row in rd:
        if row:
            try: groups[row[it]].append(tsval(row[its]))
            except: pass
for ts in groups.values():
    d=np.diff(ts); orig_rev += int((d<0).sum()>0); orig_rev_total += int((d<0).sum()); orig_zero += int((d==0).sum()>0)
TR={"rows":total,"bad_rows":bad,"tracks":len(M),"points":qs([x["n"] for x in M]),"duration":qs([x["duration"] for x in M]),"altitude":qs(all_alt),"speed":qs(all_sp),"longitude":qs(all_lon),"latitude":qs(all_lat),"heading":None,"heading_field_present":False,
    "original_reverse_tracks":orig_rev,"original_reverse_total":orig_rev_total,"original_zero_tracks":orig_zero,"duplicate_tracks":sum(x["dup"]>0 for x in M),"metrics":M}

stats={"structure":{"segment_rows":len(seg),"region_rows":len(reg),"retrieval_rows":len(ret),"segment_bad":len(bad_seg),"region_bad":len(bad_reg),"retrieval_bad":len(bad_ret),"csv_rows":total,"csv_bad":bad},
       "segment":S,"region":R,"retrieval":TQ,"tracks":TR}
(OUT/"eda_stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
with (OUT/"track_summary.csv").open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(M[0])); w.writeheader(); w.writerows(M)

plt.rcParams["figure.dpi"]=150; plt.rcParams["axes.grid"]=True; plt.rcParams["grid.alpha"]=.25
fig,ax=plt.subplots(2,3,figsize=(14,8))
ax[0,0].hist([x["n"] for x in M],bins=20); ax[0,0].set_title("Track point count")
ax[0,1].hist([x["duration"]/3600 for x in M],bins=20); ax[0,1].set_title("Track duration (hours)")
ax[0,2].hist(all_sp,bins=40); ax[0,2].set_title("Point speed (m/s)")
ax[1,0].hist(all_alt,bins=40); ax[1,0].set_title("Point altitude (m)")
ax[1,1].scatter([x["n"] for x in M],[x["duration"]/3600 for x in M],s=8); ax[1,1].set_title("Length vs duration")
ax[1,2].scatter([x["speed_mean"] for x in M],[x["alt_mean"] for x in M],s=8); ax[1,2].set_title("Mean speed vs altitude")
fig.tight_layout(); fig.savefig(OUT/"tracks_distributions.png"); plt.close(fig)

fig,ax=plt.subplots(2,3,figsize=(14,8))
for a,x,t in [(ax[0,0],[m["n"] for m in S["metrics"]],"Segment points"),(ax[0,1],[m["duration"]/3600 for m in S["metrics"]],"Segment duration"),(ax[0,2],[z for m in S["metrics"] for z in m["speed"]],"Segment speed"),(ax[1,0],[m["n"] for m in R["metrics"]],"Region points"),(ax[1,1],[m["duration"]/3600 for m in R["metrics"]],"Region duration"),(ax[1,2],[len(v(r)[2]) for r in reg],"Regions per question")]:
    a.hist(x,bins=20); a.set_title(t)
fig.tight_layout(); fig.savefig(OUT/"task1_distributions.png"); plt.close(fig)

fig,ax=plt.subplots(2,2,figsize=(12,8))
ax[0,0].hist([len(x) for x in RQ],bins=15); ax[0,0].set_title("Query length")
ax[0,1].bar(list(TQ["top_k_counts"]),list(TQ["top_k_counts"].values())); ax[0,1].set_title("top_k")
ax[1,0].bar(list(TQ["region_counts"]),list(TQ["region_counts"].values())); ax[1,0].set_title("Regions/query")
kwkeys=["时间","高度","速度","区域","不满足","先","随后","盘旋","绕飞","加速"]
kwplot=["time","altitude","speed","region","negation","first","then","circle","orbit","accelerate"]
ax[1,1].bar(kwplot,[RK.get(k,0) for k in kwkeys]); ax[1,1].set_title("Keyword presence"); ax[1,1].tick_params(axis="x",rotation=45)
fig.tight_layout(); fig.savefig(OUT/"task2_query_distributions.png"); plt.close(fig)

order=sorted(tracks); ids=[order[int(i*(len(order)-1)/8)] for i in range(9)]
fig,ax=plt.subplots(figsize=(11,7))
for tid in ids:
    p=sorted(tracks[tid],key=lambda x:x[0]); ax.plot([x[1] for x in p],[x[2] for x in p],linewidth=.8,label=tid)
ax.set_title("Sampled trajectories"); ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend(fontsize=6,ncol=3)
fig.tight_layout(); fig.savefig(OUT/"sampled_trajectory_map.png"); plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(10,4))
ax[0].bar(["reverse tracks","zero-gap tracks","duplicate tracks"],[orig_rev,orig_zero,TR["duplicate_tracks"]]); ax[0].set_title("Track data quality")
ax[1].hist(all_dt,bins=50); ax[1].set_title("Sorted positive time gaps"); ax[1].set_xlabel("seconds")
fig.tight_layout(); fig.savefig(OUT/"track_quality.png"); plt.close(fig)

print(json.dumps({"stats":stats},ensure_ascii=False,indent=2))

