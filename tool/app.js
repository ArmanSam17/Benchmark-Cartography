/* Benchmark Cartography: linked data map, question inspector and leaderboard view. */
"use strict";

const FIELDS = {
  confidence: "Confidence in official key (mean P over models)",
  variability: "Variability of P(key) across models",
  top_confidence: "Top-tier confidence in official key",
  correctness: "Share of models choosing official key",
  top_correctness: "Share of top-tier models choosing key",
  ability_alignment: "Ability alignment (corr. of P(key) with accuracy)",
  irt_a: "2PL discrimination a",
  irt_b: "2PL difficulty b (clipped to ±6)",
  vote_entropy: "Vote entropy over models (bits)",
  top_vote_entropy: "Vote entropy, top tier (bits)",
  consensus_share: "Top-tier share on consensus alternative",
};
const GROUPS = [
  { id: "unchecked", label: "Not reviewed by Redux", color: "--context" },
  { id: "clean", label: "Redux: fine", color: "--clean" },
  { id: "other", label: "Redux: other error", color: "--other-err" },
  { id: "wrong", label: "Redux: wrong key", color: "--wrong-key" },
];
const LETTERS = ["A", "B", "C", "D"];
const DRAWS = 200;
// Study mode (?study): hides every MMLU-Redux label so experts judge questions from model evidence alone.
const BLIND = new URLSearchParams(location.search).has("study");
let STUDY = [];

const S = {
  x: "confidence", y: "variability", color: "redux", subject: -1,
  hidden: new Set(), sel: null, selList: [], focus: -1,
  mode: "remove", subset: "all", models: [], isTop: null, lbToken: 0, hoverModel: -1,
};
let Q, M, MAT, N, NM, NS, groupOf, topIdx, topShares, allShares, diffBin, ORIG;

const f2 = d3.format(".2f"), f3 = d3.format(".3f");
const fmt2 = (v) => (Number.isFinite(v) ? f2(v) : "—"), fmt3 = (v) => (Number.isFinite(v) ? f3(v) : "—"), fmtPct = d3.format(".0%"), fmtInt = d3.format(",");
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const vote = (m, q) => MAT[m * N + q];
const pKey = (m, q) => MAT[NM * N + m * N + q] / 255;
const shortName = (name) => name.split("/").pop();
const tooltip = document.getElementById("tooltip");

/* ---------- loading ---------- */

async function load() {
  const [q, m, buf] = await Promise.all([
    fetch("data/questions.json").then((r) => r.json()),
    fetch("data/models.json").then((r) => r.json()),
    fetch("data/matrix.bin").then((r) => r.arrayBuffer()),
  ]);
  if (BLIND) STUDY = (await fetch("data/study_set.json").then((r) => r.json())).qids;
  Q = q; M = m; MAT = new Uint8Array(buf);
  for (const k of Object.keys(FIELDS)) Q[k] = Q[k].map((v) => (v === null ? NaN : v));
  N = Q.n; NM = M.n; NS = Q.subjects.length;
  if (MAT.length !== 2 * N * NM) throw new Error("matrix.bin size does not match the JSON files");

  groupOf = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const e = Q.error[i];
    groupOf[i] = BLIND ? 0 : e === 0 ? 0 : e === 1 ? 1 : e === 2 ? 3 : 2;
  }
  // Exact scores from the Python pipeline, used as-is for the full model set.
  ORIG = {};
  for (const k of MODEL_SCORES) ORIG[k] = Q[k];
  S.sel = new Uint8Array(N);
  computeScores(subsetModels("all"));
}

/* ---------- model subsets ---------- */

// Scores that depend on which models are read. The 2PL fit (irt_a, irt_b) stays the all-model one.
const MODEL_SCORES = ["confidence", "variability", "correctness", "ability_alignment", "top_confidence",
  "top_correctness", "consensus_alt", "consensus_share", "vote_entropy", "top_vote_entropy"];
const TOP_SHARE = 0.2;
const SUBSETS = [
  { id: "all", label: "All models" },
  { id: "base", label: "Pretrained base models only" },
  { id: "family", label: "One per family (strongest)" },
  { id: "halfA", label: "Random half A (split within families)" },
  { id: "halfB", label: "Random half B (split within families)" },
  { id: "noMmlu", label: "Exclude models named after MMLU" },
];

function subsetModels(id) {
  const all = d3.range(NM);
  if (id === "base") return all.filter((m) => M.type[m] === "pretrained");
  if (id === "noMmlu") return all.filter((m) => !M.mentions_mmlu[m]);
  if (id === "family") {
    return Array.from(d3.group(all, (m) => M.family[m]).values(), (ms) => d3.greatest(ms, (m) => M.macro_acc[m])).sort(d3.ascending);
  }
  if (id === "halfA" || id === "halfB") {
    // Shuffle within each family with a fixed seed, then alternate sides so both halves span every family.
    const rnd = mulberry32(3001), side = new Uint8Array(NM);
    let flip = 0;
    for (const ms of d3.group(all, (m) => M.family[m]).values()) {
      const order = sample(ms.slice().sort((a, b) => d3.ascending(M.name[a], M.name[b])), ms.length, rnd);
      order.forEach((m) => { side[m] = flip; flip ^= 1; });
    }
    return all.filter((m) => side[m] === (id === "halfA" ? 0 : 1));
  }
  return all;
}

// Mirrors pipeline/05_scores.py on the chosen models. P(key) is read from the 8-bit matrix,
// so subset scores carry up to 0.002 of rounding; the full set uses the exact pipeline values.
function computeScores(models) {
  const n = models.length;
  S.models = models;
  const nTop = Math.max(1, Math.round(TOP_SHARE * n));
  topIdx = models.slice().sort((a, b) => M.macro_acc[b] - M.macro_acc[a]).slice(0, nTop);
  S.isTop = new Uint8Array(NM);
  for (const m of topIdx) S.isTop[m] = 1;

  const conf = new Float64Array(N), sq = new Float64Array(N), cov = new Float64Array(N);
  const topConf = new Float64Array(N), corr = new Float64Array(N), topCorr = new Float64Array(N);
  allShares = new Float32Array(N * 4);
  topShares = new Float32Array(N * 4);
  const meanAcc = d3.mean(models, (m) => M.macro_acc[m]);
  let accSS = 0;
  for (const m of models) {
    const a = M.macro_acc[m] - meanAcc, top = S.isTop[m], off = m * N, poff = NM * N + m * N;
    accSS += a * a;
    for (let i = 0; i < N; i++) {
      const p = MAT[poff + i] / 255, v = MAT[off + i], c = v === Q.gold[i] ? 1 : 0;
      conf[i] += p; sq[i] += p * p; cov[i] += a * p; corr[i] += c;
      allShares[i * 4 + v] += 1 / n;
      if (top) { topConf[i] += p; topCorr[i] += c; topShares[i * 4 + v] += 1 / nTop; }
    }
  }
  const entropy = (arr, i) => {
    let h = 0;
    for (let k = 0; k < 4; k++) { const x = arr[i * 4 + k]; if (x > 1e-9) h -= x * Math.log2(x); }
    return Math.max(0, h);
  };
  const out = Object.fromEntries(MODEL_SCORES.map((k) => [k, new Array(N)]));
  for (let i = 0; i < N; i++) {
    const mean = conf[i] / n;
    const ssP = Math.max(0, sq[i] - n * mean * mean);
    out.confidence[i] = mean;
    out.variability[i] = Math.sqrt(ssP / n);
    out.correctness[i] = corr[i] / n;
    out.ability_alignment[i] = accSS > 0 && ssP > 1e-12 ? cov[i] / Math.sqrt(accSS * ssP) : 0;
    out.top_confidence[i] = topConf[i] / nTop;
    out.top_correctness[i] = topCorr[i] / nTop;
    let alt = -1, best = -1;
    for (let k = 0; k < 4; k++) if (k !== Q.gold[i] && topShares[i * 4 + k] > best) { best = topShares[i * 4 + k]; alt = k; }
    out.consensus_alt[i] = alt;
    out.consensus_share[i] = best;
    out.vote_entropy[i] = entropy(allShares, i);
    out.top_vote_entropy[i] = entropy(topShares, i);
  }
  for (const k of MODEL_SCORES) Q[k] = S.subset === "all" && ORIG ? ORIG[k] : out[k];
  S.computed = out;

  diffBin = new Uint8Array(N);
  for (let i = 0; i < N; i++) diffBin[i] = Math.min(9, Math.floor(Q.correctness[i] * 10));
  document.getElementById("n-models").textContent = n;
}

function setSubset(id) {
  S.subset = id;
  computeScores(subsetModels(id));
  layoutMap();
  renderSelection();
  if (S.focus >= 0) focusQuestion(S.focus);
  renderDetection();
  updateLeaderboard();
  document.getElementById("status").textContent = `${fmtInt(N)} questions · ${S.models.length} models · ${topIdx.length} top tier`;
}

// Average precision with tied scores grouped, as scikit-learn computes it.
function auprc(score, label, idx) {
  const order = idx.slice().sort((a, b) => score[b] - score[a]);
  const pos = d3.sum(order, (i) => label[i]);
  let tp = 0, seen = 0, prevRecall = 0, ap = 0;
  for (let k = 0; k < order.length; ) {
    let j = k;
    while (j < order.length && score[order[j]] === score[order[k]]) { tp += label[order[j]]; seen++; j++; }
    const recall = tp / pos;
    ap += (recall - prevRecall) * (tp / seen);
    prevRecall = recall; k = j;
  }
  return ap;
}

function renderDetection() {
  if (BLIND) return;
  const reviewed = d3.range(N).filter((i) => Q.error[i] >= 1);
  const wrong = Uint8Array.from({ length: N }, (_, i) => (Q.error[i] === 2 ? 1 : 0));
  const anyErr = Uint8Array.from({ length: N }, (_, i) => (Q.error[i] >= 2 ? 1 : 0));
  const neg = (k) => Q[k].map((v) => (Number.isFinite(v) ? -v : -Infinity));
  const rows = [
    ["Low top-tier confidence", neg("top_confidence")],
    ["Low confidence", neg("confidence")],
    ["Negative ability alignment", neg("ability_alignment")],
    ["Low share correct", neg("correctness")],
    ["Low 2PL discrimination*", neg("irt_a")],
  ];
  const base = (lab) => d3.mean(reviewed, (i) => lab[i]);
  const body = rows.map(([name, sc]) => `<tr><td>${name}</td><td class="num">${f3(auprc(sc, wrong, reviewed))}</td><td class="num">${f3(auprc(sc, anyErr, reviewed))}</td></tr>`).join("");
  document.getElementById("detect").innerHTML =
    `<table><thead><tr><th>Detection on ${fmtInt(reviewed.length)} Redux-reviewed questions (AUPRC)</th><th class="num">Wrong key (base ${f3(base(wrong))})</th><th class="num">Any error (base ${f3(base(anyErr))})</th></tr></thead><tbody>${body}</tbody></table>`
    + `<p class="note">${S.models.length} models, top tier = strongest ${topIdx.length}. *2PL is fitted once on all ${NM} models.</p>`;
}

/* ---------- controls ---------- */

function initControls() {
  for (const id of ["x-field", "y-field"]) {
    const el = document.getElementById(id);
    for (const [k, label] of Object.entries(FIELDS)) el.add(new Option(label, k));
    el.value = id === "x-field" ? S.x : S.y;
    el.onchange = () => { S[id[0]] = el.value; layoutMap(); };
  }
  const color = document.getElementById("color-field");
  if (BLIND) {
    color.querySelector('option[value="redux"]').remove();
    S.color = color.value;
    document.querySelectorAll('[data-preset="wrong"], [data-preset="anyerr"], [data-mode="redux"], .detect-wrap').forEach((el) => el.remove());
    document.querySelectorAll("#selection-table th:nth-child(2)").forEach((el) => el.remove());
    const study = document.createElement("button");
    study.dataset.preset = "study";
    study.textContent = `Study set (${STUDY.length})`;
    document.querySelector('[data-preset="flag"]').parentElement.before(study);
    document.querySelector(".sub").insertAdjacentHTML("beforeend", ' <b>Study mode: review labels are hidden.</b>');
  }
  color.onchange = () => { S.color = color.value; renderLegend(); drawMap(); };
  const subj = document.getElementById("subject");
  Q.subjects.forEach((s, i) => subj.add(new Option(s.replace(/_/g, " "), i)));
  subj.onchange = () => { S.subject = +subj.value; layoutMap(); };

  document.querySelectorAll("[data-preset]").forEach((b) => {
    b.onclick = () => {
      const p = b.dataset.preset;
      const t = +document.getElementById("flag-threshold").value;
      const pick = (f) => d3.range(N).filter((i) => visible(i) && f(i));
      if (p === "study") setSelection(STUDY.map((id) => Q.qid.indexOf(id)).filter((i) => i >= 0), "study set");
      else if (p === "wrong") setSelection(pick((i) => Q.error[i] === 2), "Redux wrong keys");
      else if (p === "anyerr") setSelection(pick((i) => Q.error[i] >= 2), "all Redux errors");
      else if (p === "flag") setSelection(pick((i) => Q.top_confidence[i] < t), `top-tier confidence < ${t}`);
      else setSelection([], "");
    };
  });
  document.querySelectorAll("[data-mode]").forEach((b) => {
    b.onclick = () => {
      S.mode = b.dataset.mode;
      document.querySelectorAll("[data-mode]").forEach((x) => x.classList.toggle("on", x === b));
      updateLeaderboard();
    };
  });
  const subset = document.getElementById("subset");
  for (const o of SUBSETS) subset.add(new Option(`${o.label} (${subsetModels(o.id).length})`, o.id));
  subset.onchange = () => setSubset(subset.value);
}

function visible(i) {
  return Number.isFinite(Q[S.x][i]) && Number.isFinite(Q[S.y][i]) && (S.subject < 0 || Q.subject[i] === S.subject) && !S.hidden.has(GROUPS[groupOf[i]].id);
}

function renderLegend() {
  const el = document.getElementById("legend");
  el.innerHTML = "";
  if (S.color === "redux") {
    const counts = d3.rollup(d3.range(N), (v) => v.length, (i) => groupOf[i]);
    GROUPS.forEach((g, gi) => {
      const item = document.createElement("span");
      item.className = "item" + (S.hidden.has(g.id) ? " off" : "");
      item.title = "Click to show or hide";
      item.innerHTML = `<span class="swatch" style="background:var(${g.color})"></span>${g.label} <span class="t-muted">${fmtInt(counts.get(gi) || 0)}</span>`;
      item.onclick = () => { S.hidden.has(g.id) ? S.hidden.delete(g.id) : S.hidden.add(g.id); renderLegend(); layoutMap(); };
      el.appendChild(item);
    });
  } else {
    const [lo, hi] = S.color === "consensus_share" ? ["0", "1"] : ["0", "2 bits"];
    el.innerHTML = `<span class="item">${FIELDS[S.color]}</span><span class="item">${lo} <span class="ramp" style="background:linear-gradient(90deg,var(--seq-lo),var(--seq-hi))"></span> ${hi}</span>`;
  }
}

/* ---------- data map ---------- */

const map = { margin: { l: 52, r: 14, t: 10, b: 40 } };

function initMap() {
  const box = document.getElementById("map");
  map.box = box;
  map.canvas = d3.select(box).append("canvas").node();
  map.svg = d3.select(box).append("svg");
  map.gx = map.svg.append("g").attr("class", "axis");
  map.gy = map.svg.append("g").attr("class", "axis");
  map.tx = map.svg.append("text").attr("class", "axis-title").attr("text-anchor", "end");
  map.ty = map.svg.append("text").attr("class", "axis-title").attr("text-anchor", "start");
  map.brushG = map.svg.append("g");
  map.brush = d3.brush().on("end", onBrushEnd);
  map.svg.on("mousemove", onMapHover).on("mouseleave", hideTip);
  new ResizeObserver(() => layoutMap()).observe(box);
}

function layoutMap() {
  const { box, margin: mg } = map;
  const w = box.clientWidth, h = box.clientHeight;
  if (!w || !Q) return;
  const dpr = window.devicePixelRatio || 1;
  map.canvas.width = w * dpr; map.canvas.height = h * dpr;
  map.canvas.style.width = w + "px"; map.canvas.style.height = h + "px";
  map.svg.attr("width", w).attr("height", h);
  map.w = w; map.h = h; map.dpr = dpr;

  const xs = Q[S.x], ys = Q[S.y];
  map.xScale = d3.scaleLinear().domain(d3.extent(xs)).nice().range([mg.l, w - mg.r]);
  map.yScale = d3.scaleLinear().domain(d3.extent(ys)).nice().range([h - mg.b, mg.t]);
  map.gx.attr("transform", `translate(0,${h - mg.b})`).call(d3.axisBottom(map.xScale).ticks(8).tickSizeOuter(0));
  map.gy.attr("transform", `translate(${mg.l},0)`).call(d3.axisLeft(map.yScale).ticks(6).tickSizeOuter(0));
  map.tx.attr("x", w - mg.r).attr("y", h - 6).text(FIELDS[S.x]);
  map.ty.attr("x", mg.l + 6).attr("y", mg.t + 12).text(FIELDS[S.y]);

  map.px = new Float32Array(N); map.py = new Float32Array(N);
  const pts = [];
  for (let i = 0; i < N; i++) {
    map.px[i] = map.xScale(xs[i]); map.py[i] = map.yScale(ys[i]);
    if (visible(i)) pts.push(i);
  }
  map.quad = d3.quadtree(pts, (i) => map.px[i], (i) => map.py[i]);
  map.brush.extent([[mg.l, mg.t], [w - mg.r, h - mg.b]]);
  map.brushG.call(map.brush);
  drawMap();
}

function pointColor(i, colors, seq) {
  if (S.color === "redux") return colors[groupOf[i]];
  const v = S.color === "consensus_share" ? Q.consensus_share[i] : Q.top_vote_entropy[i] / 2;
  return seq(Math.max(0, Math.min(1, v)));
}

function drawMap() {
  if (!map.px) return;
  const ctx = map.canvas.getContext("2d");
  ctx.setTransform(map.dpr, 0, 0, map.dpr, 0, 0);
  ctx.clearRect(0, 0, map.w, map.h);
  // light grid
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  for (const t of map.xScale.ticks(8)) { const x = map.xScale(t); ctx.beginPath(); ctx.moveTo(x, map.margin.t); ctx.lineTo(x, map.h - map.margin.b); ctx.stroke(); }
  for (const t of map.yScale.ticks(6)) { const y = map.yScale(t); ctx.beginPath(); ctx.moveTo(map.margin.l, y); ctx.lineTo(map.w - map.margin.r, y); ctx.stroke(); }

  const colors = GROUPS.map((g) => css(g.color));
  const seq = d3.interpolateRgb(css("--seq-lo"), css("--seq-hi"));
  const hasSel = S.selList.length > 0;
  // Draw order: unreviewed, fine, other errors, wrong keys, so rare labels stay on top.
  const order = d3.range(N).filter(visible).sort((a, b) => groupOf[a] - groupOf[b]);
  const surface = css("--surface-1");
  const ring = css("--select-ring");
  const pass = (selected) => {
    for (const i of order) {
      if (hasSel && !!S.sel[i] !== selected) continue;
      const g = groupOf[i];
      const r = g >= 2 && S.color === "redux" ? 2.6 : 1.8;
      ctx.globalAlpha = hasSel && !selected ? 0.18 : g === 0 && S.color === "redux" ? 0.55 : 0.85;
      ctx.fillStyle = pointColor(i, colors, seq);
      ctx.beginPath(); ctx.arc(map.px[i], map.py[i], selected && hasSel ? r + 0.6 : r, 0, 2 * Math.PI); ctx.fill();
      if (g >= 2 && S.color === "redux") { ctx.globalAlpha = 1; ctx.strokeStyle = surface; ctx.lineWidth = 0.6; ctx.stroke(); }
      else if (selected && hasSel && S.color !== "redux") { ctx.globalAlpha = 1; ctx.strokeStyle = ring; ctx.lineWidth = 1; ctx.stroke(); }
    }
  };
  pass(false);
  if (hasSel) pass(true);
  ctx.globalAlpha = 1;
  if (S.focus >= 0 && visible(S.focus)) {
    ctx.strokeStyle = css("--select-ring"); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(map.px[S.focus], map.py[S.focus], 7, 0, 2 * Math.PI); ctx.stroke();
  }
}

function onBrushEnd(ev) {
  if (!ev.sourceEvent) return;
  if (!ev.selection) {
    const [px, py] = d3.pointer(ev.sourceEvent, map.svg.node());
    const hit = map.quad.find(px, py, 10);
    if (hit !== undefined) focusQuestion(hit);
    return;
  }
  const [[x0, y0], [x1, y1]] = ev.selection;
  const picked = [];
  map.quad.visit((node, nx0, ny0, nx1, ny1) => {
    if (!node.length) {
      do { const i = node.data; if (map.px[i] >= x0 && map.px[i] <= x1 && map.py[i] >= y0 && map.py[i] <= y1) picked.push(i); } while ((node = node.next));
    }
    return nx0 > x1 || ny0 > y1 || nx1 < x0 || ny1 < y0;
  });
  const add = ev.sourceEvent.shiftKey && S.selList.length;
  setSelection(add ? Array.from(new Set([...S.selList, ...picked])) : picked, add ? "brushed region (added)" : "brushed region");
  map.brushG.call(map.brush.move, null);
}

function onMapHover(ev) {
  if (!map.quad) return;
  const [px, py] = d3.pointer(ev, map.svg.node());
  const i = map.quad.find(px, py, 8);
  if (i === undefined) return hideTip();
  const q = Q.question[i];
  showTip(ev, `<b>${Q.qid[i]}</b> <span class="t-muted">${GROUPS[groupOf[i]].label}</span><br>${escapeHtml(q.length > 180 ? q.slice(0, 180) + "…" : q)}<br><span class="t-muted">${FIELDS[S.x].split(" (")[0]}: ${fmt3(Q[S.x][i])} · ${FIELDS[S.y].split(" (")[0]}: ${fmt3(Q[S.y][i])}</span>`);
}

/* ---------- selection ---------- */

function setSelection(list, label) {
  S.sel.fill(0);
  for (const i of list) S.sel[i] = 1;
  S.selList = list;
  S.selLabel = label;
  renderSelection();
  drawMap();
  updateLeaderboard();
}

function renderSelection() {
  const el = document.getElementById("selection-summary");
  const list = S.selList;
  if (BLIND && list.length) {
    el.innerHTML = `<b>${fmtInt(list.length)}</b> questions selected (${S.selLabel}).`;
  } else if (!list.length) {
    el.innerHTML = "Nothing selected. Drag on the map to select (hold Shift to add), or use a preset above.";
  } else {
    const c = [0, 0, 0, 0];
    for (const i of list) c[groupOf[i]]++;
    const reviewed = c[1] + c[2] + c[3];
    const prec = reviewed ? ` Of the ${fmtInt(reviewed)} reviewed by Redux, <b>${fmtPct((c[2] + c[3]) / reviewed)}</b> are flawed and <b>${fmtPct(c[3] / reviewed)}</b> have a wrong key (base rates 6.5% and 1.9%).` : "";
    el.innerHTML = `<b>${fmtInt(list.length)}</b> questions selected (${S.selLabel}): ${c[3]} wrong key, ${c[2]} other error, ${c[1]} fine, ${fmtInt(c[0])} not reviewed.${prec}`;
  }
  const rows = list.slice().sort((a, b) => Q.top_confidence[a] - Q.top_confidence[b]).slice(0, 300);
  const tbody = d3.select("#selection-table tbody");
  tbody.selectAll("tr").data(rows, (d) => d).join("tr")
    .classed("active", (d) => d === S.focus)
    .html((i) => `<td>${Q.qid[i]}</td>${BLIND ? "" : `<td>${badge(i)}</td>`}<td class="num">${fmt3(Q.top_confidence[i])}</td><td class="num">${fmt2(Q.consensus_share[i])}</td><td class="num">${fmt2(Q.irt_a[i])}</td>`)
    .on("click", (_, i) => focusQuestion(i));
}

function badge(i) {
  if (BLIND) return "";
  const g = GROUPS[groupOf[i]];
  const detail = Q.error[i] >= 2 ? Q.error_codes[Q.error[i]].replace(/_/g, " ") : g.label.replace("Redux: ", "");
  return `<span class="badge"><span class="dot" style="background:var(${g.color})"></span>${detail}</span>`;
}

/* ---------- inspector ---------- */

function focusQuestion(i) {
  S.focus = i;
  drawMap();
  d3.selectAll("#selection-table tbody tr").classed("active", (d) => d === i);
  const el = document.getElementById("inspector");
  const gold = Q.gold[i], fix = Q.redux_correct[i], alt = Q.consensus_alt[i];
  const topPlural = d3.greatestIndex([0, 1, 2, 3], (k) => topShares[i * 4 + k]);
  const opts = Q.options[i].map((o, k) => {
    const tags = [];
    if (k === gold) tags.push('<span class="tag key">official key</span>');
    if (!BLIND && k === fix && fix !== gold) tags.push('<span class="tag fix">Redux correction</span>');
    if (k === alt) tags.push(`<span class="tag cons">consensus alternative</span>`);
    if (k === topPlural) tags.push('<span class="tag">top-tier plurality</span>');
    const a = allShares[i * 4 + k], t = topShares[i * 4 + k];
    return `<div class="opt"><span class="letter">${LETTERS[k]}</span><div>${escapeHtml(o)}<div class="tags">${tags.join("")}</div></div>
      <div class="bars"><span>All ${S.models.length}</span><span class="bar all"><span style="width:${a * 100}%"></span></span><span class="num">${fmtPct(a)}</span>
      <span>Top ${topIdx.length}</span><span class="bar top"><span style="width:${t * 100}%"></span></span><span class="num">${fmtPct(t)}</span></div></div>`;
  }).join("");
  const sc = (k, v) => `<div class="score"><div class="k">${k}</div><div class="v">${v}</div></div>`;
  el.innerHTML = `
    <div class="meta"><b>${Q.qid[i]}</b><span>${Q.subjects[Q.subject[i]].replace(/_/g, " ")}</span>${badge(i)}</div>
    ${!BLIND && Q.redux_reason[i] ? `<p class="reason">Redux note: ${escapeHtml(Q.redux_reason[i])}</p>` : ""}
    <p class="qtext">${escapeHtml(Q.question[i])}</p>
    ${opts}
    <div class="scores">
      ${sc("Confidence", fmt3(Q.confidence[i]))}${sc("Top-tier conf.", fmt3(Q.top_confidence[i]))}
      ${sc("Variability", fmt3(Q.variability[i]))}${sc("Ability align.", fmt2(Q.ability_alignment[i]))}
      ${sc("Share correct", fmtPct(Q.correctness[i]))}${sc("Top-tier entropy", fmt2(Q.top_vote_entropy[i]))}
      ${sc("2PL a", fmt2(Q.irt_a[i]))}${sc("2PL b", fmt2(Q.irt_b[i]))}
    </div>
    <h3>Each model's probability on the official key, by model accuracy</h3>
    <div id="model-strip"></div>`;
  drawModelStrip(i);
}

function drawModelStrip(i) {
  const box = document.getElementById("model-strip");
  const w = Math.max(260, box.clientWidth), h = 186, mg = { l: 34, r: 8, t: 14, b: 34 };
  const svg = d3.select(box).append("svg").attr("width", w).attr("height", h);
  const x = d3.scaleLinear().domain(d3.extent(M.macro_acc)).nice().range([mg.l, w - mg.r]);
  const y = d3.scaleLinear().domain([0, 1]).range([h - mg.b, mg.t]);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${h - mg.b})`).call(d3.axisBottom(x).ticks(5).tickFormat(fmtPct).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${mg.l},0)`).call(d3.axisLeft(y).ticks(4).tickSizeOuter(0));
  svg.append("text").attr("x", w - mg.r).attr("y", h - 2).attr("text-anchor", "end").text("model macro accuracy");
  svg.append("text").attr("x", 2).attr("y", 9).text("P(official key)");
  svg.append("g").selectAll("circle").data(S.models.slice().sort((a, b) => S.isTop[a] - S.isTop[b])).join("circle")
    .attr("cx", (m) => x(M.macro_acc[m])).attr("cy", (m) => y(pKey(m, i))).attr("r", 4)
    .attr("fill", (m) => (S.isTop[m] ? "var(--accent)" : "var(--clean)"))
    .attr("fill-opacity", (m) => (S.isTop[m] ? 0.95 : 0.55))
    .attr("stroke", "var(--surface-1)").attr("stroke-width", 1)
    .on("mousemove", (ev, m) => showTip(ev, `<b>${shortName(M.name[m])}</b><br>accuracy ${fmtPct(M.macro_acc[m])}${S.isTop[m] ? " · top tier" : ""}<br>P(key) ${fmt2(pKey(m, i))} · chose ${LETTERS[vote(m, i)]}`))
    .on("mouseleave", hideTip);
}

/* ---------- leaderboard ---------- */

function activeModels() {
  return S.models;
}

function baseCounts(models) {
  const C = new Float64Array(models.length * NS), Ns = new Float64Array(NS);
  for (let i = 0; i < N; i++) Ns[Q.subject[i]]++;
  models.forEach((m, j) => {
    const off = m * N, row = j * NS;
    for (let i = 0; i < N; i++) if (MAT[off + i] === Q.gold[i]) C[row + Q.subject[i]]++;
  });
  return { C, Ns };
}

// Macro accuracy (mean of per-subject accuracy) after removing or re-keying questions.
function scoreModels(models, base, removed, rekeyQ, rekeyK) {
  const C = base.C.slice(), Ns = base.Ns.slice();
  const nm = models.length;
  for (const i of removed) {
    const s = Q.subject[i], g = Q.gold[i];
    Ns[s]--;
    for (let j = 0; j < nm; j++) if (MAT[models[j] * N + i] === g) C[j * NS + s]--;
  }
  for (let r = 0; r < rekeyQ.length; r++) {
    const i = rekeyQ[r], k = rekeyK[r], s = Q.subject[i], g = Q.gold[i];
    for (let j = 0; j < nm; j++) {
      const v = MAT[models[j] * N + i];
      C[j * NS + s] += (v === k) - (v === g);
    }
  }
  const acc = new Float64Array(nm);
  for (let j = 0; j < nm; j++) {
    let sum = 0, cnt = 0;
    for (let s = 0; s < NS; s++) if (Ns[s] > 0) { sum += C[j * NS + s] / Ns[s]; cnt++; }
    acc[j] = sum / cnt;
  }
  return acc;
}

function ranksOf(acc) {
  const order = d3.range(acc.length).sort((a, b) => acc[b] - acc[a]);
  const r = new Int32Array(acc.length);
  order.forEach((j, k) => (r[j] = k + 1));
  return r;
}

function discordant(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) for (let j = i + 1; j < a.length; j++) if ((a[i] - a[j]) * (b[i] - b[j]) < 0) d++;
  return d;
}

// The questions the current mode actually changes, and the key each one gets.
function plannedChange(list) {
  if (S.mode === "remove") return { removed: list, rekeyQ: [], rekeyK: [] };
  if (S.mode === "consensus") return { removed: [], rekeyQ: list, rekeyK: list.map((i) => Q.consensus_alt[i]) };
  const qs = list.filter((i) => Q.redux_correct[i] >= 0 && Q.redux_correct[i] !== Q.gold[i]);
  return { removed: [], rekeyQ: qs, rekeyK: qs.map((i) => Q.redux_correct[i]) };
}

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sample(pool, k, rnd) {
  if (k >= pool.length) return pool.slice();
  const a = pool.slice();
  for (let i = 0; i < k; i++) { const j = i + Math.floor(rnd() * (a.length - i)); [a[i], a[j]] = [a[j], a[i]]; }
  return a.slice(0, k);
}

async function updateLeaderboard() {
  const token = ++S.lbToken;
  const models = activeModels();
  const base = baseCounts(models);
  const acc0 = scoreModels(models, base, [], [], []);
  const r0 = ranksOf(acc0);
  const change = plannedChange(S.selList);
  const changed = S.mode === "remove" ? change.removed : change.rekeyQ;
  const acc1 = scoreModels(models, base, change.removed, change.rekeyQ, change.rekeyK);
  const r1 = ranksOf(acc1);
  const nPairs = (models.length * (models.length - 1)) / 2;
  const D = discordant(acc0, acc1);
  const moved = d3.range(models.length).filter((j) => r0[j] !== r1[j]);
  const maxMove = d3.max(moved, (j) => Math.abs(r0[j] - r1[j])) || 0;
  const top10a = new Set(d3.range(models.length).filter((j) => r0[j] <= 10));
  const top10Changed = d3.range(models.length).filter((j) => r1[j] <= 10 && !top10a.has(j)).length;
  const lb = { models, acc0, acc1, r0, r1, D, nPairs, changed };

  document.getElementById("lb-action-word").textContent = S.mode === "remove" ? "removals" : "re-keys to the consensus alternative";
  renderStats(lb, moved.length, maxMove, top10Changed, null);
  renderSlope(lb);
  renderMovers(lb);

  const note = document.getElementById("baseline-note");
  const host = d3.select("#baseline");
  host.selectAll("*").remove();
  if (!changed.length) {
    note.textContent = S.mode === "redux" && S.selList.length
      ? "None of the selected questions has a Redux correction that differs from the official key."
      : "Select questions to compare their effect with random questions of the same difficulty.";
    return;
  }

  // Difficulty-matched baseline: same count per decile of "share of models correct",
  // drawn from questions outside the selection that Redux has not flagged.
  const need = new Int32Array(10);
  for (const i of changed) need[diffBin[i]]++;
  const pools = Array.from({ length: 10 }, () => []);
  const poolAll = [];
  for (let i = 0; i < N; i++) {
    if (S.sel[i] || Q.error[i] >= 2) continue;
    pools[diffBin[i]].push(i); poolAll.push(i);
  }
  const short = d3.sum(d3.range(10), (b) => Math.max(0, need[b] - pools[b].length));
  const rnd = mulberry32(20261121);
  const matched = [], unmatched = [];
  note.textContent = `Running ${DRAWS} draws…`;
  const t0 = performance.now();
  for (let d = 0; d < DRAWS; d++) {
    const draw = [];
    for (let b = 0; b < 10; b++) if (need[b]) draw.push(...sample(pools[b], need[b], rnd));
    const draw2 = sample(poolAll, changed.length, rnd);
    for (const [arr, out] of [[draw, matched], [draw2, unmatched]]) {
      const acc = S.mode === "remove"
        ? scoreModels(models, base, arr, [], [])
        : scoreModels(models, base, [], arr, arr.map((i) => Q.consensus_alt[i]));
      out.push(discordant(acc0, acc));
    }
    if (performance.now() - t0 > 30) { await new Promise((r) => setTimeout(r)); if (token !== S.lbToken) return; }
  }
  if (token !== S.lbToken) return;
  const pM = (1 + matched.filter((x) => x >= D).length) / (DRAWS + 1);
  const pU = (1 + unmatched.filter((x) => x >= D).length) / (DRAWS + 1);
  note.innerHTML = `${DRAWS} draws of ${fmtInt(changed.length)} unflagged questions with the same difficulty profile (deciles of share correct)${short ? `; ${short} could not be matched because the pool ran out` : ""}. `
    + `Matched p = <b>${fmt3(pM)}</b>; unmatched random p = ${fmt3(pU)}.`
    + (S.mode !== "remove" ? " Baseline questions are re-keyed to their own consensus alternatives." : "");
  renderStats(lb, moved.length, maxMove, top10Changed, { pM, matched });
  renderBaseline(matched, unmatched, D);
}

function renderStats(lb, nMoved, maxMove, top10Changed, base) {
  const card = (k, v, d) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;
  const has = lb.changed.length > 0;
  const tau = 1 - (2 * lb.D) / lb.nPairs;
  const verb = S.mode === "remove" ? "removed" : "re-keyed";
  document.getElementById("lb-stats").innerHTML =
    card(`Questions ${verb}`, fmtInt(lb.changed.length), `${lb.models.length} models ranked`) +
    card("Pairs that swap", has ? fmtInt(lb.D) : "—", `of ${fmtInt(lb.nPairs)} model pairs`) +
    card("Kendall τ vs. original", has ? fmt3(tau) : "—", "1.000 means no change") +
    card("Models that move", has ? fmtInt(nMoved) : "—", has ? `largest move ${maxMove} ${maxMove === 1 ? "place" : "places"}` : "") +
    card("New in top 10", has ? top10Changed : "—", "models entering the top 10") +
    card("Matched baseline", base ? `p = ${fmt3(base.pM)}` : has ? "…" : "—", base ? `median ${d3.median(base.matched)} ${d3.median(base.matched) === 1 ? "swap" : "swaps"} by chance` : "difficulty-matched random");
}

function renderSlope(lb) {
  const box = document.getElementById("slope");
  box.innerHTML = "";
  const n = lb.models.length, w = Math.max(320, box.clientWidth), rowH = 4.2;
  const mg = { t: 26, b: 10 }, h = mg.t + mg.b + n * rowH;
  const xl = 64, xr = w - 64;
  const y = d3.scaleLinear().domain([1, n]).range([mg.t, h - mg.b]);
  const svg = d3.select(box).append("svg").attr("width", w).attr("height", h);
  svg.append("text").attr("class", "col-title").attr("x", xl).attr("y", 14).attr("text-anchor", "middle").text("Original rank");
  svg.append("text").attr("class", "col-title").attr("x", xr).attr("y", 14).attr("text-anchor", "middle").text(S.mode === "remove" ? "After removal" : "After re-keying");
  for (const [x, anchor, dx] of [[xl, "end", -8], [xr, "start", 8]]) {
    svg.append("g").selectAll("text").data([1, 10, 25, 50, 75, 100, 125, n].filter((r) => r <= n)).join("text")
      .attr("x", x + dx).attr("y", (r) => y(r) + 3).attr("text-anchor", anchor).text((r) => r);
  }
  const ids = d3.range(n).sort((a, b) => Math.abs(lb.r1[a] - lb.r0[a]) - Math.abs(lb.r1[b] - lb.r0[b]));
  const stroke = (j) => (lb.r1[j] < lb.r0[j] ? "var(--up)" : lb.r1[j] > lb.r0[j] ? "var(--down)" : "var(--context)");
  const lines = svg.append("g").selectAll("line").data(ids).join("line")
    .attr("x1", xl).attr("x2", xr).attr("y1", (j) => y(lb.r0[j])).attr("y2", (j) => y(lb.r1[j]))
    .attr("stroke", stroke).attr("stroke-width", (j) => (lb.r0[j] === lb.r1[j] ? 1 : 1.6))
    .attr("stroke-opacity", (j) => (lb.r0[j] === lb.r1[j] ? 0.25 : 0.95));
  svg.append("g").selectAll("line").data(ids).join("line")
    .attr("x1", xl).attr("x2", xr).attr("y1", (j) => y(lb.r0[j])).attr("y2", (j) => y(lb.r1[j]))
    .attr("stroke", "transparent").attr("stroke-width", 6)
    .on("mousemove", (ev, j) => {
      lines.attr("stroke-opacity", (k) => (k === j ? 1 : 0.15)).attr("stroke-width", (k) => (k === j ? 3 : 1));
      const m = lb.models[j];
      showTip(ev, `<b>${shortName(M.name[m])}</b> <span class="t-muted">${M.family[m]} · ${M.type[m]}</span><br>rank ${lb.r0[j]} → ${lb.r1[j]}<br>accuracy ${fmt3(lb.acc0[j])} → ${fmt3(lb.acc1[j])}`);
    })
    .on("mouseleave", () => {
      lines.attr("stroke-opacity", (j) => (lb.r0[j] === lb.r1[j] ? 0.25 : 0.95)).attr("stroke-width", (j) => (lb.r0[j] === lb.r1[j] ? 1 : 1.6));
      hideTip();
    });
}

function renderMovers(lb) {
  const rows = d3.range(lb.models.length).filter((j) => lb.r0[j] !== lb.r1[j])
    .sort((a, b) => Math.abs(lb.r1[b] - lb.r0[b]) - Math.abs(lb.r1[a] - lb.r0[a]) || lb.r0[a] - lb.r0[b]).slice(0, 12);
  d3.select("#movers tbody").selectAll("tr").data(rows).join("tr").html((j) => {
    const d = lb.r0[j] - lb.r1[j];
    return `<td title="${M.name[lb.models[j]]}">${shortName(M.name[lb.models[j]])}</td><td class="num">${lb.r0[j]} → ${lb.r1[j]}</td>`
      + `<td class="num" style="color:var(${d > 0 ? "--up" : "--down"})">${d > 0 ? "▲" : "▼"} ${Math.abs(d)}</td><td class="num">${fmt3(lb.acc1[j])}</td>`;
  });
  if (!rows.length) d3.select("#movers tbody").html('<tr><td colspan="4" class="t-muted">No model changes rank.</td></tr>');
}

function renderBaseline(matched, unmatched, D) {
  const box = document.getElementById("baseline");
  const w = Math.max(260, box.clientWidth), h = 150, mg = { l: 30, r: 10, t: 14, b: 30 };
  const svg = d3.select(box).append("svg").attr("width", w).attr("height", h);
  const x = d3.scaleLinear().domain([0, Math.max(D, d3.max(matched), d3.max(unmatched)) * 1.08 || 1]).nice().range([mg.l, w - mg.r]);
  const bins = d3.bin().domain(x.domain()).thresholds(x.ticks(24))(matched);
  const y = d3.scaleLinear().domain([0, d3.max(bins, (b) => b.length)]).nice().range([h - mg.b, mg.t]);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${h - mg.b})`).call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${mg.l},0)`).call(d3.axisLeft(y).ticks(3).tickSizeOuter(0));
  svg.append("text").attr("x", w - mg.r).attr("y", h - 2).attr("text-anchor", "end").text("model pairs that swap");
  svg.append("g").selectAll("rect").data(bins).join("rect")
    .attr("x", (b) => x(b.x0) + 1).attr("width", (b) => Math.max(1, x(b.x1) - x(b.x0) - 2))
    .attr("y", (b) => y(b.length)).attr("height", (b) => y(0) - y(b.length)).attr("rx", 2)
    .attr("fill", "var(--clean)").attr("fill-opacity", 0.7);
  const um = d3.median(unmatched);
  svg.append("line").attr("x1", x(um)).attr("x2", x(um)).attr("y1", mg.t).attr("y2", h - mg.b)
    .attr("stroke", "var(--text-muted)").attr("stroke-dasharray", "3 3");
  svg.append("text").attr("x", x(um) - 3).attr("y", mg.t + 4).attr("text-anchor", "end").text("unmatched median");
  svg.append("line").attr("x1", x(D)).attr("x2", x(D)).attr("y1", mg.t - 4).attr("y2", h - mg.b)
    .attr("stroke", "var(--text-primary)").attr("stroke-width", 2);
  svg.append("text").attr("x", x(D) + (x(D) > w - 90 ? -4 : 4)).attr("y", mg.t + 4)
    .attr("text-anchor", x(D) > w - 90 ? "end" : "start").style("fill", "var(--text-primary)").text(`selection: ${D}`);
}

/* ---------- helpers ---------- */

function showTip(ev, html) {
  tooltip.innerHTML = html;
  tooltip.hidden = false;
  const pad = 14, r = tooltip.getBoundingClientRect();
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - pad;
  tooltip.style.left = x + "px"; tooltip.style.top = y + "px";
}
function hideTip() { tooltip.hidden = true; }
function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

/* ---------- boot ---------- */

load().then(() => {
  initControls();
  renderLegend();
  initMap();
  layoutMap();
  setSelection([], "");
  renderDetection();
  const status = document.getElementById("status");
  status.textContent = `${fmtInt(N)} questions · ${S.models.length} models · ${topIdx.length} top tier`;
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { drawMap(); });
}).catch((err) => {
  document.getElementById("status").textContent = "Could not load data: " + err.message;
  console.error(err);
});
