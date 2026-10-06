"""Local, single-session PACE viewer: python watch.py runs/example/checkpoint.pt."""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

from pace import Config, Pace, cautious
from train import History, advance, load


HTML = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PACE</title>
<style>
body{max-width:900px;margin:32px auto;padding:0 18px;color:#222;background:#fff;font:14px ui-monospace,monospace}
header,nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap}header{justify-content:space-between}h1{font:inherit;margin:0}a{color:inherit}
#objective{font:clamp(16px,3vw,23px) Georgia,serif;margin:28px 0}button,select,input{font:inherit;color:inherit;background:none;border:1px solid #ddd;padding:5px}input{width:112px}
 .chart{width:100%;height:360px;margin:20px 0 8px}.mini{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.mini div{height:120px}p{line-height:1.8}small{color:#777}#values{white-space:pre-wrap}#meta{overflow-wrap:anywhere}.safety{color:#00835b}
</style>
<header><h1><a href="https://www.paradigm.xyz/research/pace/">PACE</a> / πθ</h1><small>local simulation</small></header>
<script src="https://code.highcharts.com/highcharts.js"></script>
<p id="objective">maxπ E[(C₁(T) − αC₂(T))/20 − β𝟙crash]</p>
<nav><label>π <select id="mode" aria-label="Policy"><option>PPO</option></select></label><label>seed <input id="seed" type="number" min="0" max="4294967295" step="1" value="2147483648"></label><button id="restart">↺ Reset</button><button id="play" disabled>Play</button><select id="speed" aria-label="Playback speed"><option value="1">1×</option><option value="4">4×</option><option value="10">10×</option></select></nav>
<div id="graph" class="chart" aria-label="Agent deployment d1, opponent deployment d2, and safety s over time"></div>
<small>― d₁ agent &nbsp; ┄ d₂ opponent &nbsp; <span class="safety">― s safety</span></small>
<div class="mini"><div id="cash1"></div><div id="cash2"></div><div id="gap"></div></div>
<p id="values" role="status">Loading…</p><small id="meta"></small>
<script>
const $=id=>document.getElementById(id);let charts={};
let s, points=[], playing=false, reset=true, endedAt=0;
async function api(path,data={}){
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const value=await r.json();if(!r.ok)throw Error(value.error||r.statusText);return value;
}
function draw(){
  const times=points.map(p=>p.t), lines={d1:points.map(p=>p.deployed[0]),d2:points.map(p=>p.deployed[1]),s:points.map(p=>p.safety),c1:points.map(p=>p.cash[0]),c2:points.map(p=>p.cash[1]),gap:points.map(p=>p.cash[0]-p.cash[1])};
  function chart(id,title,series,height){
    if(!charts[id]) charts[id]=Highcharts.chart(id,{chart:{height,animation:false},title:{text:title,style:{fontSize:height<150?'11px':'14px'}},credits:{enabled:false},legend:{enabled:height>150},xAxis:{visible:false},yAxis:{title:{text:null}},plotOptions:{series:{animation:false,marker:{enabled:false}}},series:series.map(x=>({name:x[0],data:x[1].map((v,i)=>[times[i],v]),color:x[2],dashStyle:x[3]||'Solid'}))});
    else series.forEach((x,i)=>charts[id].series[i].setData(x[1].map((v,j)=>[times[j],v]),false));
    charts[id]?.redraw();
  }
  chart('graph','capability / safety',[['d₁',lines.d1,'#222'],['d₂',lines.d2,'#999','Dash'],['s',lines.s,'#00835b']],360);
  chart('cash1','C₁',[['agent',lines.c1,'#222']],120);chart('cash2','C₂',[['opponent',lines.c2,'#999']],120);chart('gap','ΔC',[['agent − opponent',lines.gap,'#00835b']],120);
  const gap=s.cash[0]-s.cash[1], outcome=s.crashed?'crash':s.done?(gap>0?'win':gap<0?'loss':'tie'):s.settling?'settling':`u=${s.action%2}, share=${Number(s.action>=2)}`;
  $('objective').textContent=`maxπ E[(C₁(T) − ${s.alpha}C₂(T))/20 − ${s.beta}𝟙crash]`;
  $('values').textContent=`t = ${s.t.toFixed(1)}s   C₁ = ${s.cash[0].toFixed(2)}   C₂ = ${s.cash[1].toFixed(2)}   ΔC = ${gap.toFixed(2)} ($B)\n${outcome}`;
  $('values').style.color=s.crashed?'#b33':'#222';$('meta').textContent=`${s.version} · ${s.sync}`;
  $('play').textContent=playing?'Pause':'Play';$('play').disabled=false;
}
$('restart').onclick=$('mode').onchange=$('seed').onchange=()=>{reset=true;};
$('play').onclick=()=>{playing=!playing;if(s.done&&playing)reset=true;draw();};
async function tick(){
  const start=performance.now();
  try{
    if(reset){
      reset=false;const first=!s;s=await api('/start',{mode:$('mode').value,seed:$('seed').value});points=[s];
      if(first)playing=s.following;
      const selected=$('mode').value;$('mode').replaceChildren(...s.modes.map(m=>new Option(m,m,m===selected,m===selected)));draw();
    }else if(playing){
      if(s.done){if(s.following&&start-endedAt>2000)reset=true;else if(!s.following){playing=false;draw();}}
      else{s=await api('/step');if(s.t===0)points=[];points.push(s);if(s.done)endedAt=performance.now();draw();}
    }
  }catch(e){playing=false;$('values').textContent=e.message;}
  setTimeout(tick,Math.max(10,1000*(s?.dt||.1)/Number($('speed').value)-(performance.now()-start)));
}
tick();
</script></html>"""


class Game:
    def __init__(self, checkpoint, allow_missing=False):
        self.checkpoint = checkpoint
        if not allow_missing and not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        self.start("PPO", 2**31)

    def start(self, mode, seed):
        imitation = self.checkpoint.with_name("imitation.pt")
        self.modes = ["Teacher", "PPO"] + (["Imitation"] if imitation.exists() else [])
        if mode not in self.modes or not 0 <= seed < 2**32:
            raise ValueError("Choose an available policy and a seed in [0, 2³²).")
        self.mode, self.seed = mode, seed
        path = self.checkpoint if mode != "Imitation" and self.checkpoint.exists() else imitation
        self.model, saved = load(path) if path.exists() else (None, {})
        if mode == "Teacher":
            self.model = None
        self.config = Config(**saved.get("config", {}))
        self.reward = {"alpha": 0, "beta": 0, **saved.get("reward_config", {})}
        self.version = ("teacher" if mode == "Teacher" else "awaiting checkpoint" if not self.model
                        else f"n = {saved.get('transitions', 0):,}")
        self.env = Pace([seed], self.config, legacy=self.model is not None and self.model.config.input_dim == 23)
        self.rng = torch.Generator().manual_seed(seed)
        self.history = History(self.env.observe(), self.model.config.context if self.model else 1, "cpu")
        self.action = 0
        return self.state()

    @torch.no_grad()
    def step(self):
        if self.mode != "Teacher" and self.model is None:
            return self.start(self.mode, self.seed)
        e = self.env
        if not e.done[0]:
            if self.model:
                distribution, _ = self.model(self.history.tokens, self.history.lengths)
                action = torch.multinomial(distribution.probs, 1, generator=self.rng).flatten()
            else:
                action = torch.as_tensor(cautious(e))
            self.action = int(action[0])
            advance(e, self.history, action)
        return self.state()

    def state(self):
        e = self.env
        return dict(t=float(e.ticks[0] * e.c.dt), safety=float(e.safety[0]),
                    deployed=e.d[0].tolist(), cash=e.cash[0].tolist(), action=self.action,
                    done=bool(e.done[0]), crashed=bool(e.crashed[0]),
                    dt=e.c.decision_dt, duration=e.c.duration + e.c.delay, settling=bool(e.ticks[0] >= e.horizon),
                    modes=self.modes, version=self.version,
                    alpha=self.reward["alpha"], beta=self.reward["beta"])


def follow(run, checkpoint, status):
    import modal
    volume = modal.Volume.from_name("pace-rl-runs")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            for name in ("imitation.pt", "checkpoint.pt"):
                local = checkpoint.with_name(name)
                if name == "imitation.pt" and local.exists():
                    continue
                try:
                    data = b"".join(volume.read_file(f"{run}/{name}"))
                except FileNotFoundError:
                    continue  # Training has not saved this file yet.
                if local.exists() and local.read_bytes() == data:
                    continue
                temporary = local.with_suffix(".download")
                temporary.write_bytes(data)
                _, saved = load(temporary)
                Config(**saved["config"])
                temporary.replace(local)  # Readers always see a complete checkpoint.
            status["sync"] = "sync " + time.strftime("%H:%M:%S")
        except Exception as error:
            status["sync"] = f"Sync retry in 60s: {type(error).__name__}"
            print(status["sync"], flush=True)
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--follow-run", help="Modal run ID; poll saved checkpoints every 60 seconds")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.follow_run and (len(args.follow_run) != 32 or any(c not in "0123456789abcdef" for c in args.follow_run)):
        parser.error("follow-run must be a 32-character hexadecimal run ID")
    if args.follow_run:
        expected = Path("runs") / args.follow_run / "checkpoint.pt"
        if args.checkpoint and args.checkpoint != expected:
            parser.error("with follow-run, omit the checkpoint path")
        args.checkpoint = expected
    if not args.checkpoint:
        parser.error("provide a checkpoint path or --follow-run RUN_ID")
    status = {"sync": "Connecting to training…" if args.follow_run else "Local checkpoint",
              "following": bool(args.follow_run)}
    game = Game(args.checkpoint, allow_missing=bool(args.follow_run))
    if args.follow_run:
        threading.Thread(target=follow, args=(args.follow_run, args.checkpoint, status), daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, data, content_type="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            self.reply(HTML.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            if self.headers.get("Origin") not in (None, f"http://{self.headers.get('Host')}"):
                self.send_error(403)
                return
            try:
                if self.path == "/start":
                    data = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 1024)))
                    state = game.start(data["mode"], int(data["seed"]))
                elif self.path == "/step":
                    state = game.step()
                else:
                    self.send_error(404)
                    return
                self.reply(json.dumps({**state, **status}).encode())
            except (ValueError, KeyError) as error:
                self.reply(json.dumps({"error": str(error)}).encode(), status=400)

        def log_message(self, *_):
            pass

    print(f"Watch at http://127.0.0.1:{args.port} (Ctrl-C to stop)", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
