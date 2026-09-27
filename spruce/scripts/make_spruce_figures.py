"""Regenerate the Norway spruce OCS figures from the HBLUP (Hjd17) runs.
Outputs *_hblup.pdf/.png into ../../manuscript/ .  Run: python make_spruce_figures.py"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,'..','output'); FIG=os.path.join(HERE,'..','..','manuscript')
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4']; INK='#333333'; MUT='#777777'
plt.rcParams.update({'font.size':8,'axes.edgecolor':MUT,'axes.labelcolor':INK,'xtick.color':MUT,
    'ytick.color':MUT,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans',
    'legend.frameon':False})
def lab(a,t): a.text(-0.16,1.04,t,transform=a.transAxes,fontweight='bold',fontsize=10,color=INK)
def save(fig,name):
    for e in ('pdf','png'): fig.savefig(os.path.join(FIG,f'{name}.{e}'),dpi=300,bbox_inches='tight')
    print('wrote',name)

# ---------------------------------------------------------------- Figure 1
raw1=pd.read_csv(os.path.join(OUT,'single_gamma','spruce_rank_comparison_raw.csv'))
s=pd.read_csv(os.path.join(OUT,'single_gamma','spruce_rank_comparison_summary.csv'))
r=s[s.method=='PCA Randomized'].set_index('rank').sort_index()
st=s[s.method=='PCA Standard'].set_index('rank').sort_index()
R=raw1[raw1.method=='PCA Randomized']; S_=raw1[raw1.method=='PCA Standard']
fig,ax=plt.subplots(1,3,figsize=(9.6,3.4))
jit=lambda x: np.asarray(x)*np.exp(np.random.default_rng(0).normal(0,.012,len(x)))
# A: speedup, every replicate shown
a=ax[0]
a.scatter(jit(S_['rank']),S_.speedup,s=7,color=C[1],alpha=.45,lw=0)
a.scatter(jit(R['rank']),R.speedup,s=7,color=C[0],alpha=.45,lw=0)
a.plot(st.index,st.mean_speedup,'-s',color=C[1],ms=4,lw=2,label='PCA Standard')
a.plot(r.index,r.mean_speedup,'-o',color=C[0],ms=4,lw=2,label='PCA Randomized')
a.set_xscale('log'); a.set_yscale('log'); a.set_xticks([5,10,20,30,50,100]); a.set_xticklabels([5,10,20,30,50,100])
a.set_xlabel('Rank $k$'); a.set_ylabel('Speedup ($\\times$)')
a.legend(loc='upper right',fontsize=7); lab(a,'A')
# B: accuracy-speed frontier, SD in both directions
a=ax[1]
a.errorbar(r.mean_gain_diff_pct,r.mean_speedup,xerr=r.sd_gain_diff_pct,yerr=r.sd_speedup,
           fmt='o',color=C[0],ms=5,lw=1,capsize=2,ecolor=C[0],elinewidth=1,alpha=.9)
OFF={5:(-34,4),10:(8,4),15:(6,-12),20:(9,2),25:(-40,-10),30:(9,3),40:(9,2),50:(9,2),75:(9,2),100:(-6,-14)}
for k in r.index:
    dx,dy=OFF[k]
    a.annotate(f'$k$={k}',(r.mean_gain_diff_pct[k],r.mean_speedup[k]),textcoords='offset points',
               xytext=(dx,dy),fontsize=7,color=INK)
a.axvline(1.0,ls='--',color=MUT,lw=.8); a.text(1.05,r.mean_speedup.min()*1.05,'1% gain',fontsize=7,color=MUT)
a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel('Genetic gain difference (%)'); a.set_ylabel('Speedup ($\\times$)'); lab(a,'B')
# C: selection overlap + per-replicate spread
a=ax[2]
a.scatter(jit(R['rank']),R.overlap_pct,s=7,color=C[2],alpha=.45,lw=0)
a.plot(r.index,r.mean_overlap_pct,'-^',color=C[2],ms=5,lw=2)
a.axhline(100,ls=':',color=MUT,lw=.8)
a.set_xscale('log'); a.set_xticks([5,10,20,30,50,100]); a.set_xticklabels([5,10,20,30,50,100])
a.set_xlabel('Rank $k$'); a.set_ylabel('Selection overlap with full dense (%)')
a.set_ylim(50,104); lab(a,'C')
fig.tight_layout(w_pad=3.0); save(fig,'Figure1_spruce_rank_selection_hblup')

# ---------------------------------------------------------- gamma x rank
raw=pd.read_csv(os.path.join(OUT,'gamma_rank_sweep','gamma_rank_sweep_raw_spruce.csv'))
elb=pd.read_csv(os.path.join(OUT,'gamma_rank_sweep','gamma_rank_sweep_elbow_spruce.csv'))
g=raw.groupby(['gamma','rank']).agg(gain=('gain_diff_pct','mean'),gsd=('gain_diff_pct','std'),
    sp=('speedup','mean'),spsd=('speedup','std')).reset_index()
fig,ax=plt.subplots(1,3,figsize=(9.6,3.2))
for i,(gam,d) in enumerate(g.groupby('gamma')):
    ax[0].fill_between(d['rank'],d.gain-d.gsd,d.gain+d.gsd,color=C[i],alpha=.15,lw=0)
    ax[0].plot(d['rank'],d.gain,'-o',color=C[i],ms=3,lw=1.6,label=f'$\\gamma$={gam:g}')
    ax[1].fill_between(d['rank'],d.sp-d.spsd,d.sp+d.spsd,color=C[i],alpha=.15,lw=0)
    ax[1].plot(d['rank'],d.sp,'-o',color=C[i],ms=3,lw=1.6,label=f'$\\gamma$={gam:g}')
    e=elb[elb.gamma==gam].iloc[0]
    ax[0].plot([e.rec_rank],[d.set_index('rank').gain[e.rec_rank]],'o',color=C[i],ms=9,mfc='none',mew=1.6)
ax[0].axhline(1.0,ls='--',color=MUT,lw=.8); ax[0].set_yscale('log')
ax[0].set_xlabel('Rank $k$'); ax[0].set_ylabel('Genetic gain difference (%)'); ax[0].legend(fontsize=7); lab(ax[0],'A')
ax[1].set_yscale('log'); ax[1].set_xlabel('Rank $k$'); ax[1].set_ylabel('Speedup ($\\times$)'); lab(ax[1],'B')
a=ax[2]
for i,(_,e) in enumerate(elb.iterrows()):
    a.plot(e.n_sel_baseline,e.mean_speedup,'o',color=C[i],ms=8)
    a.annotate(f'$k$={int(e.rec_rank)}',(e.n_sel_baseline,e.mean_speedup),textcoords='offset points',
               xytext=(6,6),fontsize=7,color=INK)
b=np.polyfit(np.log(elb.n_sel_baseline),np.log(elb.mean_speedup),1)
xs=np.linspace(np.log(elb.n_sel_baseline.min()),np.log(elb.n_sel_baseline.max()),20)
a.plot(np.exp(xs),np.exp(np.polyval(b,xs)),ls='--',color=MUT,lw=1)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('Individuals selected (full dense)')
a.set_ylabel('Speedup at recommended rank ($\\times$)'); lab(a,'C')
fig.tight_layout(w_pad=3.0); save(fig,'Figure_gamma_rank_sweep_spruce_hblup')

# ------------------------------------------------------------ concordance
c=pd.read_csv(os.path.join(OUT,'single_gamma','spruce_concordance_rank30.csv'))
T=1e-4; b_,ps,pr=c.baseline_contribution,c.pca_standard_contribution,c.pca_randomized_contribution
rk=lambda x,y: np.corrcoef(pd.Series(x).rank(),pd.Series(y).rank())[0,1]
kt=lambda x,y: pd.Series(np.asarray(x)).corr(pd.Series(np.asarray(y)),method='kendall')
fig,ax=plt.subplots(2,2,figsize=(7.2,6.0)); ax=ax.ravel()
for k,(x,y,xl,yl,t) in enumerate([(b_,ps,'Full dense','PCA Standard','A'),(b_,pr,'Full dense','PCA Randomized','B'),
                                  (ps,pr,'PCA Standard','PCA Randomized','C')]):
    j=c[(x>T)&(y>T)]; xv,yv=x[j.index]*100,y[j.index]*100
    a=ax[k]; a.scatter(xv,yv,s=14,color=C[0],alpha=.7,lw=0)
    lo,hi=min(xv.min(),yv.min()),max(xv.max(),yv.max()); a.plot([lo,hi],[lo,hi],ls='--',color=MUT,lw=.8)
    a.set_xlabel(f'{xl} contribution (%)'); a.set_ylabel(f'{yl} contribution (%)')
    a.text(.03,.96,f'$\\rho$ = {rk(xv,yv):.3f}\n$\\tau$ = {kt(xv,yv):.3f}\n$n$ = {len(j)}',
           transform=a.transAxes,va='top',color=INK); lab(a,t)
j=c[(b_>T)&(pr>T)].copy()
sh=(j.pca_randomized_contribution.rank(ascending=False)-j.baseline_contribution.rank(ascending=False))
a=ax[3]; a.hist(sh,bins=np.arange(sh.min()-.5,sh.max()+1.5),color=C[0])
a.axvline(0,color=MUT,lw=.8,ls='--')
a.set_xlabel('Rank shift (RSVD $-$ full dense)'); a.set_ylabel('Individuals')
a.text(.03,.96,f'mean |shift| = {sh.abs().mean():.1f}\nmax = {int(sh.abs().max())}\n$n$ = {len(j)}',
       transform=a.transAxes,va='top',color=INK); lab(a,'D')
fig.tight_layout(h_pad=2.4,w_pad=3.0); save(fig,'Figure2_spruce_solution_concordance_hblup')
