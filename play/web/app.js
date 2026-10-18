"use strict";

// Card ids follow the engine: id = rank * 4 + suit for 0..51 (rank 0 = "2",
// 12 = "A"; suit 0..3 = spade, heart, club, diamond), 52 = small joker,
// 53 = big joker. The heart of the level rank is the wild card (逢人配).
const RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"];
const SUITS = ["♠", "♥", "♣", "♦"];
const SEAT_NAMES = ["你", "下家", "队友", "上家"];
const OUT_NAMES = ["", "头游", "二游", "三游", "末游"];
const HUMAN = 0;
const PHASE_PLAY = 3, PHASE_ROUND_END = 4;

const $ = (id) => document.getElementById(id);
const SPEED = /[?&]fast\b/.test(location.search) ? 0.02 : 1;   // ?fast for testing
const sleep = (ms) => new Promise((r) => setTimeout(r, ms * SPEED));

const S = {
  gameId: null,
  eventCount: 0,
  view: null,
  round: 0,
  level: 0,
  hand: [],            // [{uid, id}]
  groups: [],          // custom columns: [[uid...]]
  selected: new Set(),
  fresh: new Set(),    // uids received by tribute this round
  nextUid: 1,
  busy: false,
  hints: null, hintIdx: -1, hintKey: -1,
  sfIdx: -1,
  log: [],
};

// ---------- card helpers ----------
const isJoker = (id) => id >= 52;
const rankOf = (id) => (id >= 52 ? -1 : id >> 2);
const suitOf = (id) => id & 3;
const isWild = (id) => id < 52 && rankOf(id) === S.level && suitOf(id) === 1;
function power(id) {
  if (id === 53) return 16;
  if (id === 52) return 15;
  return rankOf(id) === S.level ? 13 : rankOf(id);
}
function cardName(id) {
  if (id === 53) return "大王";
  if (id === 52) return "小王";
  return SUITS[suitOf(id)] + RANKS[rankOf(id)];
}

function cardEl(id) {
  const el = document.createElement("div");
  el.className = "card";
  if (id >= 52) {
    el.classList.add("joker");
    if (id === 53) el.classList.add("red");
    el.innerHTML = `<span class="rk">JOKER</span><span class="big">${id === 53 ? "大" : "小"}</span>`;
    return el;
  }
  const s = suitOf(id);
  if (s === 1 || s === 3) el.classList.add("red");
  if (rankOf(id) === S.level) el.classList.add("level");
  el.innerHTML = `<span class="corner"><span class="rk">${RANKS[rankOf(id)]}</span><span class="st">${SUITS[s]}</span></span><span class="big">${SUITS[s]}</span>`;
  if (isWild(id)) {
    el.classList.add("wild");
    el.insertAdjacentHTML("beforeend", `<span class="wild-tag">配</span>`);
  }
  return el;
}

function miniRow(ids) {
  const row = document.createElement("div");
  row.className = "mini-row";
  ids.forEach((id) => row.appendChild(cardEl(id)));
  return row;
}

// ---------- network ----------
async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({ error: "服务器无响应" }));
  if (data.lost) {
    localStorage.removeItem("gz-game");
    S.gameId = null;
  }
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// ---------- hand bookkeeping ----------
function syncHand(ids, round) {
  if (round !== S.round) {
    S.round = round;
    S.hand = [];
    S.groups = [];
    S.selected.clear();
    S.fresh.clear();
  }
  const want = new Map();
  ids.forEach((id) => want.set(id, (want.get(id) || 0) + 1));
  const have = new Map();
  S.hand.forEach((c) => have.set(c.id, (have.get(c.id) || 0) + 1));
  // Remove surplus copies, selected ones first (those were just played).
  const order = [...S.hand].sort((a, b) => (S.selected.has(b.uid) ? 1 : 0) - (S.selected.has(a.uid) ? 1 : 0));
  const removed = new Set();
  for (const c of order) {
    const surplus = (have.get(c.id) || 0) - (want.get(c.id) || 0);
    if (surplus > 0) {
      removed.add(c.uid);
      have.set(c.id, have.get(c.id) - 1);
    }
  }
  S.hand = S.hand.filter((c) => !removed.has(c.uid));
  const added = [];
  for (const [id, n] of want) {
    for (let k = have.get(id) || 0; k < n; k++) {
      const c = { uid: S.nextUid++, id };
      S.hand.push(c);
      added.push(c.uid);
    }
  }
  if (S.hand.length !== added.length) added.forEach((u) => S.fresh.add(u)); // only mid-round arrivals
  removed.forEach((u) => { S.selected.delete(u); S.fresh.delete(u); });
  S.groups = S.groups.map((g) => g.filter((u) => !removed.has(u))).filter((g) => g.length);
}

function columns() {
  const byUid = new Map(S.hand.map((c) => [c.uid, c]));
  const inGroup = new Set(S.groups.flat());
  const cols = S.groups.map((g) => ({ custom: true, cards: g.map((u) => byUid.get(u)).filter(Boolean) }));
  const buckets = new Map();
  for (const c of S.hand) {
    if (inGroup.has(c.uid)) continue;
    const key = power(c.id);
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(c);
  }
  [...buckets.keys()].sort((a, b) => b - a).forEach((k) => {
    const cards = buckets.get(k).sort((a, b) => (isWild(a.id) - isWild(b.id)) || (suitOf(a.id) - suitOf(b.id)));
    cols.push({ custom: false, cards });
  });
  return cols;
}

// ---------- rendering ----------
function fitHand(ncols) {
  const wrap = $("hand-wrap");
  const avail = wrap.clientWidth - 8;
  const small = window.innerWidth <= 760 || window.innerHeight <= 560;
  const maxW = small ? 50 : 66;
  let w = Math.floor(avail / (Math.max(ncols, 1) * 0.82 + 0.18 + S.groups.length * 0.2));
  w = Math.max(30, Math.min(maxW, w));
  const hand = $("hand");
  hand.style.setProperty("--card-w", w + "px");
  hand.style.setProperty("--card-h", Math.round(w * 1.42) + "px");
  hand.style.fontSize = Math.round(w * 0.24) + "px";
  return { w, h: Math.round(w * 1.42), step: Math.round(w * (small ? 0.42 : 0.38)) };
}

function renderHand() {
  const cols = columns();
  const { h, step } = fitHand(cols.length);
  const hand = $("hand");
  hand.innerHTML = "";
  for (const col of cols) {
    const el = document.createElement("div");
    el.className = "col" + (col.custom ? " custom" : "");
    el.style.height = h + (col.cards.length - 1) * step + "px";
    col.cards.forEach((c, i) => {
      const ce = cardEl(c.id);
      ce.style.top = i * step + "px";
      ce.style.zIndex = i + 1;
      ce.dataset.uid = c.uid;
      if (S.selected.has(c.uid)) ce.classList.add("sel");
      if (S.fresh.has(c.uid)) ce.classList.add("new");
      el.appendChild(ce);
    });
    hand.appendChild(el);
  }
}

function setZone(seat, entry, fresh) {
  const z = $("zone-" + seat);
  z.innerHTML = "";
  z.classList.remove("fresh");
  if (!entry) return;
  if (entry.pass) {
    z.innerHTML = `<div class="zone-pass">${entry.forced ? "要不起" : "不出"}</div>`;
  } else {
    const label = document.createElement("div");
    label.className = "zone-label" + (/炸|同花顺|天王/.test(entry.type || "") ? " bomb" : "");
    label.textContent = entry.type || "";
    z.appendChild(label);
    z.appendChild(miniRow(entry.cards));
  }
  if (fresh) { void z.offsetWidth; z.classList.add("fresh"); }
}

function setCount(seat, n) {
  if (seat === HUMAN) return;
  const el = $("count-" + seat);
  el.textContent = n;
  el.classList.toggle("low", n > 0 && n <= 10);
}

function setBadge(seat, text) {
  (seat === HUMAN ? $("me-badge") : $("badge-" + seat)).textContent = text || "";
}

function setTurn(seat) {
  for (const s of [1, 2, 3]) {
    $("seat-" + s).classList.toggle("turn", s === seat);
    $("seat-" + s).classList.toggle("thinking", s === seat && S.busy);
  }
}

function renderLevels(v) {
  S.level = RANKS.indexOf(v.level);
  $("lvl-now").textContent = "打 " + v.level;
  $("lvl-us").textContent = v.levels[0];
  $("lvl-them").textContent = v.levels[1];
  $("round-no").textContent = `第 ${v.round} 局`;
}

function renderSnapshot(v) {
  S.view = v;
  renderLevels(v);
  syncHand(v.hand, v.round);
  for (const s of [1, 2, 3]) setCount(s, v.counts[s]);
  for (let s = 0; s < 4; s++) {
    const pos = v.finished.indexOf(s);
    setBadge(s, pos >= 0 ? OUT_NAMES[pos + 1] : "");
    setZone(s, v.trick[String(s)], false);
  }
  renderHand();
  renderControls();
}

function renderControls() {
  const v = S.view;
  const p = v && v.pending;
  const myTurn = !!p && !S.busy;
  $("btn-pass").classList.toggle("hidden", !(p && p.kind === "play"));
  $("btn-hint").classList.toggle("hidden", !(p && p.kind === "play"));
  $("btn-play").classList.toggle("hidden", !(p && p.kind === "play"));
  $("btn-tribute").classList.toggle("hidden", !(p && p.kind !== "play"));
  $("btn-pass").disabled = !myTurn || !p.can_pass;
  $("btn-hint").disabled = !myTurn;
  $("btn-play").disabled = !myTurn;
  $("btn-tribute").disabled = !myTurn;
  let text = "";
  if (!v) text = "";
  else if (v.match_over) text = "比赛结束";
  else if (v.phase === PHASE_ROUND_END) text = "本局结束";
  else if (S.busy) text = v.to_move > 0 ? `${SEAT_NAMES[v.to_move]}思考中…` : "";
  else if (p && p.kind === "play") text = p.lead ? "轮到你出牌（你先出）" : "轮到你出牌";
  else if (p && p.kind === "tribute") { text = "请选一张最大的牌进贡"; $("btn-tribute").textContent = "进贡"; }
  else if (p && p.kind === "back") { text = "请选一张 10 以下的非级牌还贡"; $("btn-tribute").textContent = "还贡"; }
  else if (v.to_move > 0) text = `等待${SEAT_NAMES[v.to_move]}…`;
  $("prompt").textContent = text;
  setTurn(v && !v.pending && v.phase !== PHASE_ROUND_END ? v.to_move : -1);
}

// ---------- events ----------
function logLine(html, cls) {
  S.log.push(`<div class="${cls || ""}">${html}</div>`);
  if (S.log.length > 600) S.log.splice(0, S.log.length - 600);
  if (!$("log-panel").classList.contains("hidden")) renderLog();
}
function renderLog() {
  const body = $("log-body");
  body.innerHTML = S.log.join("");
  body.scrollTop = body.scrollHeight;
}

let toastTimer = null;
function toast(text, ms = 1800) {
  const t = $("toast");
  t.textContent = text;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), ms);
}

async function playEvents(events) {
  for (const ev of events) {
    if (ev.kind === "round_start") {
      for (let s = 0; s < 4; s++) { setZone(s, null); setBadge(s, ""); }
      [1, 2, 3].forEach((s) => setCount(s, 27));
      renderLevels({ ...S.view, ...ev });
      logLine(`第 ${ev.round} 局 · 打 ${ev.level}（我方 ${ev.levels[0]} / 对方 ${ev.levels[1]}）`, "round-sep");
      toast(`第 ${ev.round} 局 · 打 ${ev.level}`, 1400);
      await sleep(700);
    } else if (ev.kind === "anti_tribute") {
      logLine("抗贡：本局不进贡");
      toast("抗贡！本局不进贡", 1800);
      await sleep(1200);
    } else if (ev.kind === "tribute" || ev.kind === "back") {
      const verb = ev.kind === "tribute" ? "进贡" : "还贡";
      const msg = `${SEAT_NAMES[ev.from]} ${verb} ${cardName(ev.card)} 给 ${SEAT_NAMES[ev.to]}`;
      logLine(msg, ev.from === HUMAN || ev.to === HUMAN ? "me" : "");
      $("center-msg").innerHTML = "";
      $("center-msg").append(msg, miniRow([ev.card]));
      await sleep(1300);
      $("center-msg").innerHTML = "";
    } else if (ev.kind === "play") {
      if (ev.new_trick) for (let s = 0; s < 4; s++) setZone(s, null);
      setZone(ev.seat, ev, true);
      setCount(ev.seat, ev.left);
      const who = SEAT_NAMES[ev.seat];
      if (ev.pass) logLine(`${who}：${ev.forced ? "要不起" : "不出"}`, ev.seat === HUMAN ? "me" : "");
      else logLine(`${who}：${ev.type} ${ev.cards.map(cardName).join(" ")}`, ev.seat === HUMAN ? "me" : "");
      if (ev.out) {
        setBadge(ev.seat, OUT_NAMES[ev.out]);
        logLine(`${who} ${OUT_NAMES[ev.out]}！`);
        toast(`${who} ${OUT_NAMES[ev.out]}！`, 1400);
      }
      const next = (ev.seat + 1) % 4;
      setTurn(next === HUMAN ? -1 : next);
      if (ev.seat === HUMAN) await sleep(250);
      else await sleep(ev.forced ? 380 : ev.out ? 1200 : 850);
    } else if (ev.kind === "round_end") {
      await sleep(500);
    }
  }
}

async function handle(data) {
  const events = data.events || [];
  S.eventCount = data.event_count;
  // Hand changes are the player's own; show them at once, then replay the table.
  S.view = data;
  syncHand(data.hand, data.round);
  for (const ev of events) {
    if ((ev.kind === "tribute" || ev.kind === "back") && ev.to === HUMAN) {
      const c = S.hand.find((x) => x.id === ev.card && !S.fresh.has(x.uid));
      if (c) S.fresh.add(c.uid);
    }
  }
  renderHand();
  S.busy = true;
  renderControls();
  try {
    await playEvents(events);
  } finally {
    S.busy = false;
  }
  renderSnapshot(data);
  refreshAI();
  if (data.result && (data.phase === PHASE_ROUND_END || data.match_over)) showResult(data);
}

async function send(path, extra) {
  if (S.busy) return null;
  S.busy = true;
  renderControls();
  try {
    const data = await api(path, { id: S.gameId, since: S.eventCount, ...extra });
    S.busy = false;
    return data;
  } catch (e) {
    S.busy = false;
    renderControls();
    toast(e.message, 2600);
    if (!S.gameId) showWelcome();
    return null;
  }
}

// ---------- actions ----------
function selectedIds() {
  return S.hand.filter((c) => S.selected.has(c.uid)).map((c) => c.id);
}

async function doPlay(reading) {
  const cards = selectedIds();
  if (!cards.length) { toast("请先选牌"); return; }
  const data = await send("/api/play", { cards, reading });
  if (!data) return;
  if (data.ok === false && data.choose) {
    S.view = data;
    renderControls();
    chooseReading(data.choose);
    return;
  }
  S.hints = null;
  S.fresh.clear();
  await handle(data);
}

async function doPass() {
  const data = await send("/api/play", { cards: [] });
  if (!data) return;
  S.selected.clear();
  S.hints = null;
  await handle(data);
}

async function doTribute() {
  const cards = selectedIds();
  if (cards.length !== 1) { toast("请选择一张牌"); return; }
  const data = await send("/api/play", { cards });
  if (!data) return;
  await handle(data);
}

async function loadHints() {
  if (!S.hints || S.hintKey !== S.eventCount) {
    const key = S.eventCount;
    const data = await api("/api/hint", { id: S.gameId });
    if (key !== S.eventCount) return [];
    S.hints = data.hints;
    S.hintKey = key;
    S.hintIdx = -1;
  }
  return S.hints;
}

async function doHint() {
  if (S.busy) return;
  let plays;
  try { plays = (await loadHints()).filter((h) => !h.pass); } catch (e) { toast(e.message); return; }
  if (!plays.length) { toast("没有能大过的牌，只能不出"); return; }
  S.hintIdx = (S.hintIdx + 1) % plays.length;
  selectIds(plays[S.hintIdx].cards);
  if (S.hintIdx === 0) toast("提示按 AI 推荐顺序排列，再点一次换下一种", 1600);
}

// ---------- AI probability panel ----------
S.showAI = localStorage.getItem("gz-ai") !== "0";

function myPlayTurn() {
  const v = S.view;
  return !!(v && v.pending && v.pending.kind === "play" && !S.busy);
}

async function refreshAI() {
  const panel = $("ai-panel");
  panel.classList.toggle("hidden", !S.showAI);
  document.body.classList.toggle("ai-open", S.showAI);
  $("btn-ai").classList.toggle("on", S.showAI);
  if (!S.showAI) return;
  const body = $("ai-body");
  if (!myPlayTurn()) {
    body.innerHTML = `<div class="ai-empty">轮到你出牌时，这里显示 AI 在你这个位置上出每一手牌的概率。</div>`;
    return;
  }
  let list;
  try { list = await loadHints(); } catch (e) { body.textContent = e.message; return; }
  if (!myPlayTurn()) return;
  body.innerHTML = "";
  const top = list.filter((h, i) => i < 3 || h.prob >= 0.005).slice(0, 8);
  for (const h of top) {
    const row = document.createElement("div");
    row.className = "ai-row" + (h.pass ? " pass" : "");
    const pct = h.prob < 0.001 ? "<0.1" : (h.prob * 100).toFixed(h.prob >= 0.1 ? 0 : 1);
    row.innerHTML = `<div class="ai-bar" style="width:${Math.max(1, h.prob * 100)}%"></div>
      <span class="ai-pct">${pct}%</span><span class="ai-type">${h.type}</span>`;
    if (h.cards.length) row.appendChild(miniRow(h.cards));
    row.onclick = () => {
      if (h.pass) { S.selected.clear(); renderHand(); toast("AI 认为可以不出：点「不出」"); }
      else selectIds(h.cards);
    };
    body.appendChild(row);
  }
  const rest = list.slice(top.length).reduce((a, h) => a + h.prob, 0);
  if (list.length > top.length) {
    body.insertAdjacentHTML("beforeend",
      `<div class="ai-empty">其余 ${list.length - top.length} 种出法合计 ${rest < 0.001 ? "<0.1" : (rest * 100).toFixed(1)}%</div>`);
  }
}

function selectIds(ids) {
  S.selected.clear();
  const pool = [...S.hand];
  // Prefer cards already grouped together, so a hint picks from the player's arrangement.
  for (const id of ids) {
    const i = pool.findIndex((c) => c.id === id);
    if (i >= 0) { S.selected.add(pool[i].uid); pool.splice(i, 1); }
  }
  renderHand();
}

function straightFlushes() {
  const wilds = S.hand.filter((c) => isWild(c.id));
  const out = [];
  for (let suit = 0; suit < 4; suit++) {
    for (let start = -1; start <= 8; start++) {
      const pick = [];
      let missing = 0;
      for (let k = 0; k < 5; k++) {
        const r = start + k < 0 ? 12 : start + k;
        const c = S.hand.find((x) => x.id === r * 4 + suit && !isWild(x.id) && !pick.includes(x));
        if (c) pick.push(c); else missing++;
      }
      if (missing <= wilds.length && missing < 5) {
        out.push({ missing, cards: [...pick, ...wilds.slice(0, missing)], start });
      }
    }
  }
  out.sort((a, b) => a.missing - b.missing || b.start - a.start);
  return out;
}

function doSF() {
  const list = straightFlushes();
  if (!list.length) { toast("手里没有同花顺"); return; }
  S.sfIdx = (S.sfIdx + 1) % list.length;
  S.selected = new Set(list[S.sfIdx].cards.map((c) => c.uid));
  renderHand();
  toast(`同花顺 ${S.sfIdx + 1}/${list.length}${list[S.sfIdx].missing ? "（用了逢人配）" : ""}，可点「理牌」放到一起`, 2000);
}

function doGroup() {
  if (!S.selected.size) { toast("先选中要放在一起的牌"); return; }
  const pick = S.hand.filter((c) => S.selected.has(c.uid)).map((c) => c.uid);
  S.groups = S.groups.map((g) => g.filter((u) => !S.selected.has(u))).filter((g) => g.length);
  S.groups.push(pick);
  S.selected.clear();
  renderHand();
}

function doReset() {
  S.groups = [];
  S.selected.clear();
  renderHand();
}

// ---------- modals ----------
function modal(html, buttons) {
  $("modal-body").innerHTML = "";
  if (typeof html === "string") $("modal-body").innerHTML = html;
  else $("modal-body").appendChild(html);
  const actions = $("modal-actions");
  actions.innerHTML = "";
  for (const b of buttons || []) {
    const el = document.createElement("button");
    el.className = "act " + (b.cls || "gold");
    el.textContent = b.text;
    el.onclick = () => { closeModal(); b.onclick && b.onclick(); };
    actions.appendChild(el);
  }
  $("modal").classList.remove("hidden");
}
function closeModal() { $("modal").classList.add("hidden"); }

function chooseReading(options) {
  const box = document.createElement("div");
  box.innerHTML = `<h2>这手牌有几种出法</h2><p>你用到了逢人配，请选择要出的牌型：</p>`;
  const list = document.createElement("div");
  list.className = "choice-list";
  for (const o of options) {
    const b = document.createElement("button");
    b.textContent = o.label;
    b.onclick = () => { closeModal(); doPlay(o.index); };
    list.appendChild(b);
  }
  box.appendChild(list);
  modal(box, [{ text: "取消", cls: "gray" }]);
}

function showResult(v) {
  const r = v.result;
  const usWin = r.winning_team === 0;
  const box = document.createElement("div");
  const title = v.match_over
    ? (v.winner === 0 ? "🎉 比赛结束：你们赢了！" : "比赛结束：AI 对手赢了")
    : (usWin ? `本局我方胜，升 ${r.gain} 级` : `本局对方胜，升 ${r.gain} 级`);
  box.innerHTML = `<h2 class="${(v.match_over ? v.winner === 0 : usWin) ? "win" : ""}">${title}</h2>`;
  const table = document.createElement("table");
  r.order.forEach((seat, i) => {
    const tr = document.createElement("tr");
    const team = seat % 2 === 0 ? "我方" : "对方";
    tr.innerHTML = `<td>${OUT_NAMES[i + 1]}</td><td>${SEAT_NAMES[seat]}（${team}）</td><td></td>`;
    if (r.hands[seat].length) tr.lastChild.appendChild(miniRow(r.hands[seat]));
    table.appendChild(tr);
  });
  box.appendChild(table);
  const p = document.createElement("p");
  p.textContent = `级数：我方 ${r.levels[0]}，对方 ${r.levels[1]}。`;
  box.appendChild(p);
  if (v.match_over) {
    modal(box, [{ text: "再来一场", onclick: newGame }]);
  } else {
    modal(box, [{ text: "下一局", onclick: nextRound }]);
  }
}

async function nextRound() {
  const data = await send("/api/next", {});
  if (data) await handle(data);
}

function showWelcome() {
  modal(`<h2>和 GuanZero AI 打掼蛋</h2>
    <p>你坐在下方，<b>队友（上）</b>和两个<b>对手（左右）</b>都是同一个 AI 模型（GuanZero 最新版）。一整场从 2 打到 A，可以随时刷新页面继续。</p>
    <p><b>选牌</b>：点击或在牌上滑动选牌；<b>理牌</b>：选中几张牌后点「理牌」把它们放成一列，「还原」恢复自动排序；「同花顺」会帮你找同花顺。</p>
    <p><b>提示</b>按 AI 认为最好的顺序给出可出的牌。红桃级牌（标「配」）是逢人配万能牌。</p>`,
    [{ text: "开始比赛", onclick: newGame }]);
}

function showHelp() {
  modal(`<h2>规则速览</h2>
    <p><b>人数</b>：4 人两队，对家是队友，两副牌 108 张，每人 27 张。</p>
    <p><b>级牌</b>：本局打几，几就是级牌，大小排在 A 之上、小王之下。红桃级牌是<b>逢人配</b>，可以当除大小王外的任何牌。</p>
    <p><b>牌型</b>：单张、对子、三张、三带二、顺子（5 张）、三连对、钢板（两个连续三张）、炸弹（4 张及以上同点）、同花顺、天王炸（四个王）。</p>
    <p><b>大小</b>：天王炸 > 6 张以上炸弹 > 同花顺 > 5 张炸 > 4 张炸 > 其他牌型（同牌型比大小）。</p>
    <p><b>升级</b>：头游和二游同队升 3 级，头游和三游升 2 级，头游和末游升 1 级。打过 A 赢得比赛。</p>
    <p><b>进贡</b>：上一局末游向头游进贡最大的牌（红桃级牌除外），头游还一张 10 以下的牌；双下要两人都进贡。手握两个大王可以抗贡。</p>`,
    [{ text: "知道了" }]);
}

async function newGame() {
  closeModal();
  try {
    S.busy = true;
    const data = await api("/api/new", {});
    S.busy = false;
    S.gameId = data.id;
    localStorage.setItem("gz-game", data.id);
    S.eventCount = 0;
    S.round = 0;
    S.log = [];
    await handle(data);
  } catch (e) {
    S.busy = false;
    toast(e.message, 3000);
  }
}

// ---------- input ----------
function setupHandInput() {
  const hand = $("hand");
  let dragging = false, mode = true, touched = new Set();
  const cardAt = (x, y) => {
    const el = document.elementFromPoint(x, y);
    return el && el.closest ? el.closest(".card[data-uid]") : null;
  };
  const apply = (el) => {
    if (!el) return;
    const uid = Number(el.dataset.uid);
    if (touched.has(uid)) return;
    touched.add(uid);
    if (mode) S.selected.add(uid); else S.selected.delete(uid);
    el.classList.toggle("sel", mode);
  };
  hand.addEventListener("pointerdown", (e) => {
    const el = cardAt(e.clientX, e.clientY);
    if (!el) return;
    e.preventDefault();
    dragging = true;
    touched = new Set();
    mode = !S.selected.has(Number(el.dataset.uid));
    apply(el);
  });
  window.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    apply(cardAt(e.clientX, e.clientY));
  });
  window.addEventListener("pointerup", () => { dragging = false; });
  hand.addEventListener("dblclick", (e) => {
    const el = e.target.closest(".card[data-uid]");
    if (!el) return;
    const col = el.parentElement;
    col.querySelectorAll(".card").forEach((c) => S.selected.add(Number(c.dataset.uid)));
    renderHand();
  });
  $("table").addEventListener("click", () => { if (S.selected.size) { S.selected.clear(); renderHand(); } });
}

function setup() {
  $("btn-play").onclick = () => doPlay(null);
  $("btn-pass").onclick = doPass;
  $("btn-hint").onclick = doHint;
  $("btn-tribute").onclick = doTribute;
  $("btn-group").onclick = doGroup;
  $("btn-reset").onclick = doReset;
  $("btn-sf").onclick = doSF;
  $("btn-clear").onclick = () => { S.selected.clear(); renderHand(); };
  $("btn-help").onclick = showHelp;
  $("btn-new").onclick = () => modal("<h2>开始新比赛？</h2><p>当前比赛进度会丢失。</p>",
    [{ text: "取消", cls: "gray" }, { text: "开始新比赛", onclick: newGame }]);
  $("btn-ai").onclick = () => {
    S.showAI = !S.showAI;
    localStorage.setItem("gz-ai", S.showAI ? "1" : "0");
    refreshAI();
  };
  $("btn-log").onclick = () => { $("log-panel").classList.toggle("hidden"); renderLog(); };
  document.querySelectorAll("[data-close]").forEach((b) => {
    b.onclick = () => $(b.dataset.close).classList.add("hidden");
  });
  window.addEventListener("keydown", (e) => {
    if (!$("modal").classList.contains("hidden")) return;
    if (e.key === "Enter") $("btn-play").classList.contains("hidden") ? $("btn-tribute").click() : $("btn-play").click();
    else if (e.key === " ") { e.preventDefault(); $("btn-pass").click(); }
    else if (e.key === "h" || e.key === "H") $("btn-hint").click();
  });
  window.addEventListener("resize", () => S.view && renderHand());
  setupHandInput();
}

async function boot() {
  setup();
  const saved = localStorage.getItem("gz-game");
  if (saved) {
    try {
      S.gameId = saved;
      const data = await api("/api/state", { id: saved, since: 1e9 });
      S.eventCount = data.event_count;
      renderSnapshot(data);
      refreshAI();
      if (data.result && (data.phase === PHASE_ROUND_END || data.match_over)) showResult(data);
      toast("已恢复上次的比赛", 1500);
      return;
    } catch (e) {
      localStorage.removeItem("gz-game");
      S.gameId = null;
    }
  }
  showWelcome();
}

boot();
