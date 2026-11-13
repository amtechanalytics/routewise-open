import json,csv,math,sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
R=os.path.join(os.path.dirname(__file__),'..','results')+'/'; OUT=os.path.join(os.path.dirname(__file__),'..','figures')+'/'; os.makedirs(OUT,exist_ok=True)
C=['#2a78d6','#eb6834','#1baf7a','#eda100']; INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.edgecolor':INK2,'axes.labelcolor':INK,
 'xtick.color':INK2,'ytick.color':INK2,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
 'grid.color':GRID,'grid.linewidth':0.6,'axes.axisbelow':True,'legend.frameon':False})
# Fig1
cfg={r['config']:r for r in csv.DictReader(open(R+'t1_configs.csv'))}
d=json.load(open(R+'deltas.json'))
fig,ax=plt.subplots(figsize=(4.6,3.2))
fams=['base','mask','dilute','merge']; mk=['o','s','^','D']
for i,f in enumerate(fams):
    xs=[float(cfg[k]['mention_rate']) for k in d if k.endswith('.'+f)]
    ys=[d[k]['route'] for k in d if k.endswith('.'+f)]
    ax.scatter(xs,ys,s=[46,40,30,30][i],marker=mk[i],facecolor=C[i] if i<2 else 'none',edgecolor=C[i] if i>=2 else 'white',linewidth=1.3,label=f,zorder=3+i)
for k in d:
    if k.endswith('.base'):
        ax.annotate(k.split('.')[0],(float(cfg[k]['mention_rate']),d[k]['route']),xytext=(5,2),textcoords='offset points',fontsize=7,color=INK2)
ax.axhline(0,color=INK2,lw=0.8)
ax.set_xlabel('Share of questions that mention their document'); ax.set_ylabel('Δ evidence recall, route − flat')
ax.legend(title='Configuration family',fontsize=7,title_fontsize=7,loc='lower right')
fig.tight_layout(); fig.savefig(OUT+'fig1_mention.pdf'); fig.savefig(OUT+'fig1_mention.png',dpi=200); plt.close()
# Fig2
arms={}
for r in csv.DictReader(open(R+'v2_t2_arms.csv')): arms[(r['config'],r['arm'])]=r
cells=['fin.L2','fda.L2','cuad.L2','mhrag.L2','finmy.L0']; lab=['fin L2','fda L2','cuad L2','mhrag L2','fin-multiyear L0']
meth=[('pdoc','pdoc'),('pdoc-rr','pdoc-rr'),('R','ROUTEWISE-R')]
fig,ax=plt.subplots(figsize=(5.6,2.9)); w=0.26
for j,(a,n) in enumerate(meth):
    v=[float(arms[(c,a)]['top1_acc']) for c in cells]
    xs=[i+(j-1)*w for i in range(len(cells))]
    ax.bar(xs,v,width=w-0.03,color=C[j],label=n,zorder=3)
    for x,y in zip(xs,v): ax.text(x,y+0.015,f'{y:.2f}',ha='center',fontsize=6.3,color=INK2)
ax.set_xticks(range(len(cells))); ax.set_xticklabels(lab); ax.set_ylabel('Top-1 document accuracy'); ax.set_ylim(0,1.05)
ax.legend(ncol=3,loc='upper left',fontsize=7)
fig.tight_layout(); fig.savefig(OUT+'fig2_resolution.pdf'); fig.savefig(OUT+'fig2_resolution.png',dpi=200); plt.close()
# Fig3
strong={'fin.L2':'meta','fda.L2':'sac','cuad.L2':'sac','mhrag.L2':'flat','cuad.L1':'meta'}
cells=['fin.L2','fda.L2','cuad.L2','mhrag.L2','cuad.L1']; lab=['fin L2','fda L2','cuad L2','mhrag L2','cuad L1\n(exploratory)']
fig,ax=plt.subplots(figsize=(5.6,2.9))
ser=[('Strongest baseline',lambda c:strong[c]),('ROUTEWISE-R',lambda c:'R'),('Gold-document oracle',lambda c:'oracle')]
for j,(n,f) in enumerate(ser):
    v=[float(arms[(c,f(c))]['correct']) for c in cells]
    xs=[i+(j-1)*w for i in range(len(cells))]
    ax.bar(xs,v,width=w-0.03,color=[C[3],C[2],'#b9b8b3'][j],label=n,zorder=3)
    for x,y in zip(xs,v): ax.text(x,y+0.012,f'{y:.2f}',ha='center',fontsize=6.3,color=INK2)
ax.set_xticks(range(len(cells))); ax.set_xticklabels(lab); ax.set_ylabel('Answer correctness'); ax.set_ylim(0,0.72)
ax.legend(ncol=3,loc='upper left',fontsize=7)
fig.tight_layout(); fig.savefig(OUT+'fig3_correctness.pdf'); fig.savefig(OUT+'fig3_correctness.png',dpi=200); plt.close()
# Fig4
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return c-h,c+h
H=R+'human/'
key={r['item']:r for r in csv.DictReader(open(H+'s3_rating_key.csv',encoding='utf-8-sig'))}
def valid_counts(sheet):
    sh={r['item']:r for r in csv.DictReader(open(H+sheet,encoding='utf-8-sig'))}
    ok=lambda r:all(r[c].strip().lower()=='yes' for c in ('same_answer','identifiable','no_leak','plausible'))
    cnt={};tot={}
    for i,k in key.items():
        if k['screened']!='accepted': continue
        cell=k['source']+' '+k['level']
        for c in (cell,'All accepted'):
            tot[c]=tot.get(c,0)+1; cnt[c]=cnt.get(c,0)+ok(sh[i])
    return cnt,tot
c1,tot=valid_counts('s3_rating_sheet_rater1.csv'); c2,_=valid_counts('s3_rating_sheet_rater2.csv')
order=sorted([c for c in tot if c!='All accepted'],key=lambda c:(-c1[c]/tot[c],c))+['All accepted']
rows=[(c,c1[c],tot[c]) for c in order]
r2={c:c2[c] for c in order}
fig,ax=plt.subplots(figsize=(4.8,3.0))
for i,(n,k,m) in enumerate(rows[::-1]):
    p=k/m; lo,hi=wilson(k,m)
    ax.barh(i,p,height=0.6,color=C[0] if n!='All accepted' else INK2,zorder=3)
    ax.plot([lo,hi],[i,i],color=INK,lw=1,zorder=4)
    ax.scatter([r2[n]/m],[i],marker='D',s=26,color=C[3],edgecolor=INK,linewidth=0.6,zorder=5,label='rater 2' if i==0 else None)
    ax.text(max(hi,p)+0.02,i,f'{k}/{m}',va='center',fontsize=7,color=INK2)
ax.axvline(0.70,color=C[1],lw=1.2,ls='--'); ax.text(0.71,len(rows)-0.6,'gate: lower bound ≥ 0.70',fontsize=6.8,color=INK2)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]]); ax.set_xlim(0,1.1)
ax.set_xlabel('Valid on all four criteria (rater 1 share, 95 % Wilson CI)'); ax.grid(axis='y',visible=False)
ax.legend(loc='lower right',fontsize=7)
fig.tight_layout(); fig.savefig(OUT+'fig4_human.pdf'); fig.savefig(OUT+'fig4_human.png',dpi=200); plt.close()
print('ok')
