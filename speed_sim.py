"""
Timing model of the real robot:  piece spawns at row 0 -> bot sees it (latency) -> presses are struck one
every `press_period` s -> gravity moves the piece down one row every `spr` s -> it locks when it can't
move down -> next piece spawns after ARE.   Policies compare:
  old    : no time budget, presses of a landed piece keep running into the next piece (old controller)
  cancel : no time budget, stale presses cancelled when the piece lands
  budget : cancel + AI only picks placements whose presses can finish before the piece lands
"""
import os, sys, random, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tetris_ai_v2 as A
from tetris_shapes import SHAPES, rotate_piece
SPAWN={'I':[[1,1,1,1]],'O':[[1,1],[1,1]],'T':[[1,1,1],[0,1,0]],'S':[[0,1,1],[1,1,0]],'Z':[[1,1,0],[0,1,1]],'J':[[1,1,1],[0,0,1]],'L':[[1,1,1],[1,0,0]]}
TAB={'I':{0:3,1:5,2:3,3:5},'O':{0:4,1:4,2:4,3:4},'T':{0:4,1:4,2:4,3:5},'S':{0:4,1:5,2:4,3:5},'Z':{0:4,1:5,2:4,3:5},'J':{0:4,1:4,2:4,3:5},'L':{0:4,1:4,2:4,3:5}}
def trim(m):
    rows=[r for r in m if any(r)]; cols=[c for c in range(len(rows[0])) if any(r[c] for r in rows)]
    return [[r[c] for c in cols] for r in rows]
ORIENT={}
for p,m in SPAWN.items():
    cur=m; ORIENT[p]=[]
    for k in range(4):
        t=trim(cur); ORIENT[p].append((len(t),len(t[0]),tuple(sum(1<<c for c in range(len(t[0])) if row[c]) for row in t)))
        cur=rotate_piece(cur)
def real_clicks(p,rot):
    tgt=SHAPES[p]
    for _ in range(rot): tgt=rotate_piece(tgt)
    tgt=trim(tgt); cur=SPAWN[p]
    for c in range(4):
        if trim(cur)==tgt: return c
        cur=rotate_piece(cur)
    return rot%4
def sequence(p,rot,col):
    c=real_clicks(p,rot)
    if p in('I','S','Z'): c%=2
    if p=='O': c=0
    seq=[]
    if c==1: seq=['CW']
    elif c==2: seq=['CW','CW']
    elif c==3: seq=['CCW']
    sc=TAB[p][c]; dx=col-sc
    seq+=(['R']*dx) if dx>0 else (['L']*(-dx))
    return seq,c,sc
def collide(rows,masks,R,L):
    if L<0: return True
    for i,m in enumerate(masks):
        r=R+i
        if r>=20: return True
        if (m<<L)>>10: return True          # outside right wall
        if rows[r]&(m<<L): return True
    return False
def play(seed,spr,policy,P=0.285,strike=0.065,lat=0.25,are=0.30,limit=300):
    rng=random.Random(seed); names=list(SPAWN)
    rows=[0]*20; lines=0; ok=0; placed=0; leak_free=0.0; leftovers=[]
    t=0.0
    for n in range(limit):
        p=rng.choice(names); clicks=0; L=TAB[p][0]; R=0
        h,w,masks=ORIENT[p][0]
        if collide(rows,masks,0,L): return placed,lines,ok
        Ts=t
        board=A.rows_to_board(rows)
        elapsed=lat
        budget=None
        if policy=='budget':
            budget=dict(spr=spr,elapsed=elapsed,press_period=P,strike=strike,lead=0.0,margin=0.1)
        pf=lambda name,rot,col:len(sequence(name,rot,col)[0])
        mv,_,_=A.find_best_move(board,p,None,None,press_fn=pf if budget else None,budget=budget)
        if mv is None: return placed,lines,ok
        seq,tc,_sc=sequence(p,*mv); tL=mv[1]
        start=Ts+lat
        if policy=='old': start=max(start,leak_free)
        presses=[(start+i*P,a) for i,a in enumerate(seq)]
        if policy=='old' and leftovers: presses=sorted(leftovers+presses)   # stale presses from the last piece keep firing
        leftovers=[]
        ev=sorted([(Ts+k*spr,'g') for k in range(1,60)]+[(pt,a) for pt,a in presses])
        locked=False; tl=Ts
        for et,a in ev:
            if a=='g':
                if collide(rows,ORIENT[p][clicks%4][2],R+1,L): locked=True; tl=et; break
                R+=1
            else:
                if et<Ts: continue
                if a in('L','R'):
                    nl=L+(-1 if a=='L' else 1)
                    if not collide(rows,ORIENT[p][clicks%4][2],R,nl): L=nl
                else:
                    nc=(clicks+(1 if a=='CW' else -1))%4
                    nL=TAB[p][nc]
                    if not collide(rows,ORIENT[p][nc][2],R,nL): clicks=nc; L=nL
        if not locked: return placed,lines,ok
        h,w,masks=ORIENT[p][clicks%4]
        for i,m in enumerate(masks): rows[R+i]|=(m<<L)
        placed+=1
        if clicks==tc and L==tL: ok+=1
        rows,cl=A._clear(rows); lines+=cl
        if policy=='old':
            leftovers=[(pt,a) for pt,a in presses if pt>tl]
            leak_free=max([tl]+[pt+P for pt,_ in presses])
        t=tl+are
    return placed,lines,ok
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description="Model: how does fall speed (seconds per row) vs button-press time affect the bot?")
    ap.add_argument("--press-ms",type=float,nargs="+",default=[285,150],help="press period(s) to compare: strike + recharge, ms")
    ap.add_argument("--spr",type=float,nargs="+",default=[0.9,0.5,0.3,0.2,0.12,0.08],help="seconds per row to test")
    ap.add_argument("--pieces",type=int,default=200)
    ap.add_argument("--seeds",type=int,default=4)
    a=ap.parse_args()
    pols=[('cancel',"cancel stale presses only"),('budget',"cancel + time-budgeted AI")]
    print("MODEL (not a measurement). pieces survived out of %d (mean of %d seeds) / %% of pieces landing where the AI intended"%(a.pieces,a.seeds))
    for pms in a.press_ms:
        print("\nbutton press period %.0f ms"%pms)
        print("  sec/row | "+" | ".join("%-30s"%n for _,n in pols))
        for spr in a.spr:
            cells=[]
            for pol,_ in pols:
                res=[play(s,spr,pol,P=pms/1000.0,limit=a.pieces) for s in range(1,a.seeds+1)]
                surv=statistics.mean(r[0] for r in res); okp=100*sum(r[2] for r in res)/max(1,sum(r[0] for r in res))
                cells.append("%5.0f survived, %3.0f%% intended"%(surv,okp))
            print("  %7.2f | "%spr+" | ".join("%-30s"%c for c in cells),flush=True)
