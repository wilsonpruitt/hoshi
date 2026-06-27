const Hoshi = (function(){
  function makeConfig(N){
    // Deploy radius. 9x9 keeps ratio ~2.3 (R 4). 13x13 uses R 5 (ratio 2.6):
    // MCTS self-play shows it's healthier than R 6 (perfectly first-player fair,
    // capture path alive, best balance objective) AND a centered trooper no
    // longer blankets the whole board, so placement carries risk. Too sparse
    // (ratio >>2.7, e.g. 9x9 at R 3) kills the capture path — don't go there.
    const R = N>=13 ? 5 : Math.max(3, Math.round(N/2.3));
    // encircle: local-encirclement trooper capture — a deployed trooper is immune
    // to group-liberty capture and falls only when every on-board neighbor is an
    // enemy stone (drones still die normally). Off by default = classic Go capture.
    // mobileDrones: drones gain a 'move' action (relocate within a ready trooper's
    // shadow). moveMode 'step' = one orthogonal cell (wriggle, can't escape range);
    // 'redeploy' = jump anywhere in shadow. Off by default = classic static play.
    return { N, R, troopers:5, captureToWin:3, maxPlies:N*N*4, encircle:false,
             mobileDrones:false, moveMode:'step' };
  }
  // positional-superko key: full board layout + whose turn it is to play next
  function posKey(board, toMove){
    let s=''; for(const c of board) s += c ? (''+c.c+c.k) : '_';
    return s+'|'+toMove;
  }
  // Setup variants are a "hidden rule" (just a starting position) — they add no
  // rules to learn but are the game's main source of strategic variety: how many
  // troopers you commit forward at the start. -1 = fog (scatter all).
  const SETUP_HOME = {beachhead:0, vanguard:1, garrison:2, bastion:3, fog:-1};
  function spreadCols(N,r,k){
    const out=[]; for(let t=0;t<k;t++){
      const c=Math.min(N-1,Math.max(0,Math.round((t+1)*N/(k+1))));
      out.push(r*N+c);
    } return out;
  }
  const idx=(N,r,c)=>r*N+c;
  const rc=(N,i)=>[Math.floor(i/N), i%N];
  const cheb=(N,i,j)=>{const[a,b]=rc(N,i),[d,e]=rc(N,j);return Math.max(Math.abs(a-d),Math.abs(b-e));};
  function orth(N,i){const[r,c]=rc(N,i),o=[];
    if(r>0)o.push(i-N); if(r<N-1)o.push(i+N); if(c>0)o.push(i-1); if(c<N-1)o.push(i+1); return o;}

  function groupLib(N,board,i){
    const color=board[i].c, seen=new Set([i]), st=[i]; let libs=0; const lib=new Set();
    while(st.length){const p=st.pop();
      for(const q of orth(N,p)){
        if(board[q]==null){ if(!lib.has(q)){lib.add(q);libs++;} }
        else if(board[q].c===color && !seen.has(q)){seen.add(q);st.push(q);}
      }}
    return {group:[...seen], libs};
  }
  function ready(s,i){ const p=s.placedPly[i]; return p===undefined ? true : s.ply >= p+2; }
  function readyTroopers(s,color){
    const out=[]; for(let i=0;i<s.board.length;i++){const b=s.board[i];
      if(b&&b.c===color&&b.k==='T'&&ready(s,i))out.push(i);} return out;}
  function legalDrone(s,i){
    if(s.board[i]!=null)return false;
    const rt=readyTroopers(s,s.toMove);
    return rt.some(t=>cheb(s.N,t,i)<=s.R);
  }
  function legalTrooper(s,i){ return s.board[i]==null && s.reserve[s.toMove]>0; }

  function clone(s){
    return {N:s.N,R:s.R,troopers:s.troopers,captureToWin:s.captureToWin,maxPlies:s.maxPlies,
      encircle:s.encircle, mobileDrones:s.mobileDrones, moveMode:s.moveMode,
      board:s.board.map(x=>x?{c:x.c,k:x.k}:null),
      reserve:[...s.reserve], lostTroopers:[...s.lostTroopers],
      placedPly:Object.assign({},s.placedPly), history:new Set(s.history),
      toMove:s.toMove, ply:s.ply, passes:s.passes, over:s.over, winner:s.winner};
  }

  function areaScore(s){
    const N=s.N, stones=[0,0]; for(const b of s.board) if(b)stones[b.c]++;
    const seen=new Set(), terr=[0,0];
    for(let i=0;i<N*N;i++){
      if(s.board[i]!=null||seen.has(i))continue;
      const reg=[i], st=[i], border=new Set(); seen.add(i);
      while(st.length){const p=st.pop();
        for(const q of orth(N,p)){
          if(s.board[q]!=null)border.add(s.board[q].c);
          else if(!seen.has(q)){seen.add(q);reg.push(q);st.push(q);}
        }}
      if(border.size===1)terr[[...border][0]]+=reg.length;
    }
    return [stones[0]+terr[0], stones[1]+terr[1]];
  }
  function endTerritory(ns){
    const [a,b]=areaScore(ns); ns.over=true; ns.winner = a>b?0:(b>a?1:-1);
  }

  // Captures from a stone landing at index `i` (drop OR mobile-drone move). Mutates
  // ns, tallies captured troopers, returns the removed Set. Shared so the drop and
  // move actions can never drift apart.
  function applyCaptures(ns, i, me){
    const dead=new Set();
    for(const q of orth(ns.N,i)){
      if(ns.board[q]&&ns.board[q].c!==me&&!dead.has(q)){
        const {group,libs}=groupLib(ns.N,ns.board,q);
        if(libs===0)group.forEach(g=>dead.add(g));
      }}
    if(ns.encircle){
      // troopers are immune to group-liberty capture: from a dead group only the
      // drones fall. A trooper dies only when every on-board neighbor is an enemy
      // stone — which `i` can only complete for a trooper beside it. (Check the
      // pre-removal board: a friendly-drone neighbor, even a dying one, means the
      // trooper is not yet walled in.)
      for(const d of [...dead]) if(ns.board[d].k==='T') dead.delete(d);
      for(const q of orth(ns.N,i)){
        const cell=ns.board[q];
        if(cell&&cell.c!==me&&cell.k==='T' &&
           orth(ns.N,q).every(r=>ns.board[r]&&ns.board[r].c===me))
          dead.add(q);
      }
    }
    dead.forEach(d=>{ if(ns.board[d].k==='T'){ns.lostTroopers[1-me]++; delete ns.placedPly[d];}
                      ns.board[d]=null; });
    return dead;
  }

  // move = {type:'trooper'|'drone'|'pass', i} | {type:'move', from, to}
  function apply(s, move){
    const ns=clone(s), me=s.toMove;
    if(move.type==='pass'){           // Go rule: TWO passes in a row end the game & score
      ns.passes=(ns.passes||0)+1; ns.toMove^=1; ns.ply++;
      if(ns.passes>=2) endTerritory(ns);
      return ns;
    }

    if(move.type==='move'){              // mobile drone: relocate within trooper shadow
      if(!ns.mobileDrones)return null;
      const {from,to}=move, cell=ns.board[from];
      if(!cell||cell.c!==me||cell.k!=='D'||ns.board[to]!=null)return null;
      if(ns.moveMode==='step' && !orth(ns.N,from).includes(to))return null;
      const ready=readyTroopers(s,me);
      if(!ready.some(t=>cheb(ns.N,t,to)<=ns.R))return null;
      ns.board[from]=null; ns.board[to]={c:me,k:'D'};
      const dead=applyCaptures(ns,to,me);
      const {libs}=groupLib(ns.N,ns.board,to);
      if(libs===0 && dead.size===0)return null;
      const key=posKey(ns.board, me^1);
      if(ns.history.has(key)) return null;
      ns.history.add(key);
      ns.passes=0;
      if(ns.lostTroopers[1-me]>=ns.captureToWin){ ns.over=true; ns.winner=me; }
      ns.toMove^=1; ns.ply++;
      if(!ns.over && ns.ply>=ns.maxPlies) endTerritory(ns);
      return ns;
    }

    const i=move.i;
    if(ns.board[i]!=null)return null;
    if(move.type==='trooper'){ if(ns.reserve[me]<=0)return null; }
    else { if(!legalDrone(s,i))return null; }
    ns.board[i]={c:me,k:move.type==='trooper'?'T':'D'};
    const dead=applyCaptures(ns,i,me);
    // suicide
    const {libs}=groupLib(ns.N,ns.board,i);
    if(libs===0 && dead.size===0)return null;
    // positional superko: a move may not recreate a previous position (stops
    // endless capture/recapture cycles). Pass moves are exempt.
    const key=posKey(ns.board, me^1);
    if(ns.history.has(key)) return null;
    ns.history.add(key);
    // bookkeeping
    if(move.type==='trooper'){ ns.reserve[me]--; ns.placedPly[i]=ns.ply; }
    ns.passes=0;
    if(ns.lostTroopers[1-me]>=ns.captureToWin){ ns.over=true; ns.winner=me; }
    ns.toMove^=1; ns.ply++;
    if(!ns.over && ns.ply>=ns.maxPlies) endTerritory(ns);   // safety cap -> score
    return ns;
  }

  function legalMoves(s){
    const out=[{type:'pass'}];
    if(s.reserve[s.toMove]>0)
      for(let i=0;i<s.board.length;i++) if(legalTrooper(s,i)) out.push({type:'trooper',i});
    for(let i=0;i<s.board.length;i++) if(legalDrone(s,i)) out.push({type:'drone',i});
    if(s.mobileDrones){           // relocate a drone within a ready trooper's shadow
      const ready=readyTroopers(s,s.toMove);
      if(ready.length){
        const inShadow=i=>s.board[i]==null && ready.some(t=>cheb(s.N,t,i)<=s.R);
        for(let f=0;f<s.board.length;f++){
          const c=s.board[f]; if(!c||c.c!==s.toMove||c.k!=='D')continue;
          const cands = s.moveMode==='step' ? orth(s.N,f).filter(inShadow)
                                            : (()=>{const a=[];for(let i=0;i<s.board.length;i++)if(inShadow(i))a.push(i);return a;})();
          for(const t of cands) if(apply(s,{type:'move',from:f,to:t})) out.push({type:'move',from:f,to:t});
        }
      }
    }
    return out;
  }

  function newState(N, variant){
    const cfg=makeConfig(N);
    const s={...cfg, board:new Array(N*N).fill(null),
      reserve:[cfg.troopers,cfg.troopers], lostTroopers:[0,0], placedPly:{},
      history:new Set(), toMove:0, ply:0, passes:0, over:false, winner:null};
    const home = (variant in SETUP_HOME) ? SETUP_HOME[variant] : 2;
    for(const color of [0,1]){
      let pts=[];
      if(home<0){                                   // fog: scatter all troopers
        while(pts.length<cfg.troopers){ const i=Math.floor(Math.random()*N*N);
          if(s.board[i]==null && !pts.includes(i)) pts.push(i); }
      } else {                                       // commit `home` to the back row
        pts = spreadCols(N, color===0?0:N-1, home);
      }
      for(const i of pts){ if(s.board[i]!=null)continue;
        s.board[i]={c:color,k:'T'}; s.reserve[color]--; s.placedPly[i]=-10; }
    }
    s.history.add(posKey(s.board, 0));   // seed: the opening position can't recur
    return s;
  }

  return {makeConfig,newState,apply,legalMoves,legalDrone,legalTrooper,
          readyTroopers,ready,areaScore,groupLib,cheb,orth,idx,rc};
})();
if(typeof module!=='undefined') module.exports=Hoshi;
