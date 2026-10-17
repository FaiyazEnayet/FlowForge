"""Animate source series and analytical curves, never a scrolling HTML screenshot."""
import sys,json,base64,io,re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'portfolio_ui'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,dates as mdates
from PIL import Image
from fontTools.ttLib import TTFont
from datetime import datetime
from configs import CONFIGS
out=ROOT;tmp=ROOT/'.cache/gif-fonts';tmp.mkdir(parents=True,exist_ok=True)
fonts=json.loads((ROOT/'portfolio_ui/fonts.json').read_text(encoding='utf8'))
def register(family):
 p=tmp/(family.replace(' ','')+'.ttf')
 if not p.exists():
  b=base64.b64decode(re.findall(r'base64,([^\)]+)',fonts[family])[-1]);f=TTFont(io.BytesIO(b));f.flavor=None;f.save(p)
 font_manager.fontManager.addfont(str(p));return font_manager.FontProperties(fname=str(p)).get_name()
def values(a):
 if isinstance(a,list):return np.asarray(a)
 return np.frombuffer(base64.b64decode(a['bdata']),dtype=np.dtype(a['dtype']))
def evidence(repo):return json.loads((ROOT/'portfolio_ui/animation-data.json').read_text(encoding='utf8'))
def save(repo,frames,kind,source):
 dest=ROOT/'docs/demo.gif';dest.parent.mkdir(exist_ok=True)
 frames[0].save(dest,save_all=True,append_images=frames[1:],duration=[155]*(len(frames)-1)+[1500],loop=0,optimize=True,disposal=2)
 return dict(repo=repo,kind=kind,source='portfolio_ui/animation-data.json',frames=Image.open(dest).n_frames,size=dest.stat().st_size)
reports=[]
for repo in ['FlowForge']:
 c=CONFIGS[repo];bg,surface,text,muted,accent,line=c['colors'];body=register(c['body']);title=register(c['display']);d=evidence(repo);figures=d['figures'];frames=[]
 plt.rcParams.update({'font.family':body,'font.size':10,'text.color':text,'axes.labelcolor':muted,'xtick.color':muted,'ytick.color':muted,'axes.edgecolor':line,'axes.facecolor':surface,'figure.facecolor':bg,'grid.color':line,'grid.alpha':.5,'legend.facecolor':surface,'legend.edgecolor':line,'legend.labelcolor':text})
 for frame in range(32):
  progress=(frame+1)/32
  if repo=='FlowForge':fig,axes=plt.subplots(3,1,figsize=(10,6.8),sharex=True);fig.subplots_adjust(left=.13,right=.96,top=.83,bottom=.1,hspace=.3)
  elif repo in ['toxicflow','wattwise']:fig,axes=plt.subplots(2,1,figsize=(10,6.4),sharex=True);fig.subplots_adjust(left=.13,right=.96,top=.81,bottom=.12,hspace=.35)
  else:fig,ax=plt.subplots(figsize=(10,6));axes=[ax];fig.subplots_adjust(left=.13,right=.96,top=.81,bottom=.14)
  fig.text(.075,.94,c['name'],fontfamily=title,fontsize=25,color=text)
  for ax in np.atleast_1d(axes):ax.grid(True,axis='both');ax.spines[['top','right']].set_visible(False)
  if repo=='neural-cppi':
   f=figures[1];ax=axes[0];n=len(values(f['data'][0]['x']));k=max(1,int(n*progress));yy=[]
   for i,color in [(0,c['series'][0]),(2,c['series'][1])]:
    trace=f['data'][i];x=values(trace['x']);y=values(trace['y']);band=values(f['data'][i+1]['y']);a,b=band[:n],band[n:][::-1];yy.extend([a.min(),b.min(),a.max(),b.max()]);ax.plot(x[:k],y[:k],lw=2,color=color,label=trace['name']);ax.fill_between(x[:k],np.minimum(a,b)[:k],np.maximum(a,b)[:k],color=color,alpha=.14)
   ax.axhline(90,color=muted,lw=1,ls='--',label='Protection floor 90');ax.set(xlim=(0,float(x[-1])),ylim=(min(89,min(yy)-1),max(yy)+1),xlabel='Trading day',ylabel='Wealth · initial 100');ax.legend(loc='upper left',fontsize=9);fig.text(.075,.875,'Block-bootstrap scenarios · median and 10th–90th percentile bands',fontsize=10,color=muted)
  elif repo=='FlowForge':
   f=figures[0];labels=['Rebased midprice','Spread · bp','Book imbalance'];n=len(values(f['data'][0]['x']));k=max(1,int(n*progress))
   for j,ax in enumerate(axes):
    for i,label,color in [(j,'Observed #186',c['series'][0]),(24+j,'Generated #140',c['series'][1])]:
     tr=f['data'][i];x,y=values(tr['x']),values(tr['y']);ax.plot(x[:k],y[:k],color=color,lw=1.8,label=label)
    all_y=np.r_[values(f['data'][j]['y']),values(f['data'][24+j]['y'])];pad=max(float(np.ptp(all_y))*.12,1e-6);ax.set(xlim=(float(x[0]),float(x[-1])),ylim=(float(all_y.min())-pad,float(all_y.max())+pad),ylabel=labels[j])
   axes[0].legend(loc='upper right',fontsize=9,ncol=2);axes[-1].set_xlabel('Offset within separate 60-second windows · seconds');fig.text(.075,.88,'Observed market data and DDPM output · independent samples, aligned by offset',fontsize=10,color=muted)
  elif repo=='toxicflow':
   f=figures[0];first=np.array([mdates.date2num(datetime.fromisoformat(x)) for x in f['data'][0]['x']]);end=first[0]+progress*(first[-1]-first[0])
   for ax,i,label,color in [(axes[0],0,'BTCUSDT midprice · USD',c['series'][0]),(axes[1],2,'VPIN · unitless',c['series'][1])]:
    tr=f['data'][i];x=np.array([mdates.date2num(datetime.fromisoformat(v)) for v in tr['x']]);y=np.array(tr['y']);mask=x<=end;ax.plot(x[mask],y[mask],color=color,lw=1.8);pad=max(float(np.ptp(y))*.12,.01);ax.set(xlim=(first[0],first[-1]),ylim=(float(y.min())-pad,float(y.max())+pad),ylabel=label);ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
   axes[-1].set_xlabel('Recorded session time · source timestamp clock');fig.text(.075,.875,'Recorded price and equal-volume toxicity · chronological reveal, no interpolation',fontsize=10,color=muted)
  elif repo=='sukuklab':
   cf=d['sukuk']['cashflows'];times=np.array([r['t'] for r in cf]);amount=np.array([r['amount'] for r in cf]);ys=np.linspace(.02,.09,120);prices=np.array([(amount/(1+y/2)**(2*times)).sum() for y in ys]);y=.02+.07*progress;price=float((amount/(1+y/2)**(2*times)).sum());ax=axes[0];ax.plot(ys*100,prices,color=accent,lw=2);ax.scatter([y*100],[price],color=accent,s=80,zorder=3);ax.axvline(y*100,color=muted,lw=1,ls=':');ax.set(xlabel='Market yield · nominal semiannual · %',ylabel='Price · currency units per 1,000 face');fig.text(.075,.875,'Illustrative fixed-profit cash flows · 5.06% annual profit · Actual/365',fontsize=10,color=muted);ax.text(.97,.93,f'Yield {y:.2%}\nPrice {price:,.2f}',transform=ax.transAxes,ha='right',va='top',fontsize=14,color=text)
  elif repo=='wattwise':
   f=figures[0];k=max(1,int(24*progress))
   for i,color in [(0,c['series'][0]),(1,c['series'][1])]:
    tr=f['data'][i];x,y=np.array(tr['x']),np.array(tr['y']);axes[0].plot(x[:k],y[:k],lw=2,color=color,label=tr['name'])
   ys=np.r_[f['data'][0]['y'],f['data'][1]['y']];axes[0].set(ylim=(0,float(ys.max())*1.15),ylabel='Average hourly load · kWh');axes[0].legend(loc='upper left',ncol=2,fontsize=9);price=np.array(f['data'][2]['y']);axes[1].step(x[:k],price[:k],color=c['series'][2],where='mid',lw=2);axes[1].set(xlim=(0,23),ylim=(0,float(price.max())*1.2),xlabel='Hour of day',ylabel='TOU tariff · USD/kWh');fig.text(.075,.875,'Seeded household profile · actual and optimized source curves · $5.18 shifting saving',fontsize=10,color=muted)
  else:
   data=json.loads((ROOT/'data/sbengaged-events.json').read_text(encoding='utf8'));events=sorted([e for e in data['events'] if e['food_probability']>=.5],key=lambda x:x['score'],reverse=True)[:8];ax=axes[0];fig.subplots_adjust(left=.44,right=.96,top=.81,bottom=.14);k=max(1,int(8*progress));import textwrap
   y=np.arange(8);ax.barh(y[:k],[e['score'] for e in events[:k]],color=accent,height=.55);ax.set_yticks(y,['\n'.join(textwrap.wrap(e['title'],32)) for e in events],fontsize=10);ax.invert_yaxis();ax.set(xlim=(0,max(e['score'] for e in events)*1.18),xlabel='Model probability × food-quality weight');fig.text(.075,.875,'Public SB Engaged snapshot · '+data['fetched_at'][:10]+' · food availability unverified',fontsize=10,color=muted)
   for i,e in enumerate(events[:k]):ax.text(e['score']+.015,i,f"{e['food_probability']:.0%}",va='center',fontsize=10,color=text)
  buf=io.BytesIO();fig.savefig(buf,format='png',dpi=105,facecolor=bg);plt.close(fig);buf.seek(0);frames.append(Image.open(buf).convert('RGB').quantize(colors=192,method=Image.Quantize.MEDIANCUT))
 reports.append(save(repo,frames,'Animated source series or original analytical cash-flow calculation','portfolio_ui/animation-data.json'));print(repo,reports[-1]['size'],flush=True)
print('Wrote docs/demo.gif from source evidence')
