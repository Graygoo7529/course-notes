const DATA = JSON.parse(document.getElementById("report-payload").textContent);

const STANCE_COLORS = {
  dam_collapse_rumor: "#D75A4A",
  geopolitical_blame_other: "#D98A3C",
  earthquake_misread: "#9B6BB1",
  breaking_disaster_report: "#7F8FA0",
  science_glacier_collapse: "#238B8E",
  factcheck_debunk: "#2D6DA3",
  external_referenced: "#B8C4CC",
};

const SLOT_CN = {
  chinese_dam: "中国大坝溃坝词",
  hedging_reportedly: "reportedly免责词",
  tourists_291: "291外籍失联",
  indians_105: "105印度公民",
  us_inflated: "美籍夸大数字",
  delayed_warning: "预警延误/蓄意归责",
  quake_m44_m52: "M4.4/5.2地震词",
  cctv_67584ms: "67584ms同源监控视频",
};

const SPOTLIGHTS = {
  mirror: {
    mode: "cooccurrence",
    handles: ["@RealBababanaras", "@Ujwalthapa12"],
    title: "隐蔽镜像搬运锁定：@RealBababanaras (X09) ↔ @Ujwalthapa12 (X11)",
    summary:
      "X11 以“无转发/无引用原创帖”形式发布，在明线传播网中完全隐身；但暗线共现网通过三重独立物证一眼将其锁死：与 X09 相隔仅 39 分钟（0.66h）、6/6 特征槽位完全相同、视频附件精确同为 67584 ms，且正文连同截断片段几乎逐字照搬（TF-IDF 相似度 0.761，共现权重 0.896 全场第一）。",
    inspect: "@Ujwalthapa12",
  },
  hijack: {
    mode: "explicit",
    handles: ["@BlacksmithIndie", "@ravipandey2643", "@TheInsiderPaper"],
    title: "寄生式流量劫持：低粉丝号借 2,555 万与 653 万中立爆帖植入溃坝论",
    summary:
      "明线图中清晰暴露出“寄生式嫁接”机制：@BlacksmithIndie 自身浏览仅数百次，却精准瞄准高曝光纯灾情爆帖 @ravipandey2643（653 万浏览）与 @TheInsiderPaper（2,555 万浏览），通过引用与首评植入简短情绪词“Chinese dam break 💔”，将虚假归因强行搭车到中立灾害视频的巨大流量池中。",
    inspect: "@BlacksmithIndie",
  },
  clique: {
    mode: "cooccurrence",
    handles: ["@ferozwala", "@DailyExplorerX", "@NewsAlgebraIND", "@SanjeevCrime", "@RealBababanaras", "@TicTocTick"],
    title: "同源真实数字的分叉污染团块（291 外籍 / 105 印度人槽位）",
    summary:
      "这组账号在 1–4 小时内构成高密度暗线团块：它们几乎同步抓取了尼泊尔旅游局的“291 名外国游客（含 105 名印度人）”真实排查名单作为可信锚点，随后迅速分叉为三条截然不同的叙事——纯报灾情（@ferozwala / @DailyExplorerX / @NewsAlgebraIND）、地震致洪误读（@SanjeevCrime）与大坝溃坝谣言（@RealBababanaras）；8 小时后的 @TicTocTick 又承接了“大坝 + reportedly + 美籍夸大”槽位，将美籍失联数雪球式推高至 60 人。",
    inspect: "@RealBababanaras",
  },
  debunk: {
    mode: "explicit",
    handles: [
      "@RealBababanaras", "@Valeria__79", "@Praveen79066956",
      "@F1ght4dream", "@tiraspeaks", "@RushAnn1234", "@JeanMarcNoelB",
      "@Corrine4o", "@RadharamnDas", "@DHM_FloodEWS", "@NDRRMA_Nepal", "@moha_nepal",
    ],
    title: "跨语种外溢、单跳靶向辟谣与官方协作链",
    summary:
      "明线图同时呈现三种显式拓扑：① @Valeria__79（意大利语）与 @Praveen79066956 引用 @RealBababanaras 形成局部星型辐射与跨语种外溢；② @F1ght4dream → @tiraspeaks、@Corrine4o → @RadharamnDas 等民间核查呈现精准的“单跳点名驳斥”；③ @DHM_FloodEWS → @NDRRMA_Nepal → @moha_nepal 构成规范的官方应急背书链。",
    inspect: "@F1ght4dream",
  },
};

const SVG_NS = "http://www.w3.org/2000/svg";
let currentMode = "cooccurrence";
let activeSpotlight = null;
let selectedHandle = "@RealBababanaras";

function elSvg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

function fmtViews(v) {
  if (!v) return "0";
  if (v >= 1000000) return (v / 1000000).toFixed(2) + "M";
  if (v >= 1000) return (v / 1000).toFixed(1) + "k";
  return String(v);
}

function renderNetwork(mode, spotlightKey = null) {
  currentMode = mode;
  activeSpotlight = spotlightKey;
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  document.querySelectorAll(".spot-btn[data-spot]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.spot === spotlightKey);
  });

  const spotObj = spotlightKey ? SPOTLIGHTS[spotlightKey] : null;
  const spotSet = spotObj ? new Set(spotObj.handles) : null;

  const svg = document.getElementById("network-svg");
  const tooltip = document.getElementById("net-tooltip");
  svg.replaceChildren();

  const nodes = DATA.network.nodes;
  const nodeMap = Object.fromEntries(nodes.map((n) => [n.handle, n]));
  const explicitEdges = DATA.network.explicit_edges;
  const coEdges = DATA.network.cooccurrence_edges;

  const activeHandles = new Set();
  if (mode === "explicit") {
    explicitEdges.forEach((e) => { activeHandles.add(e.source); activeHandles.add(e.target); });
  } else if (mode === "cooccurrence") {
    coEdges.forEach((e) => { activeHandles.add(e.author_a); activeHandles.add(e.author_b); });
  } else {
    explicitEdges.forEach((e) => { activeHandles.add(e.source); activeHandles.add(e.target); });
    coEdges.forEach((e) => { activeHandles.add(e.author_a); activeHandles.add(e.author_b); });
  }

  const defs = elSvg("defs");
  const marker = elSvg("marker", {
    id: "arrow-exp", viewBox: "0 0 10 10", refX: "18", refY: "5",
    markerWidth: "6", markerHeight: "6", orient: "auto-start-reverse",
  });
  marker.appendChild(elSvg("path", { d: "M 0 1 L 10 5 L 0 9 z", fill: "#6e8290" }));
  defs.appendChild(marker);
  svg.appendChild(defs);

  const edgeGroup = elSvg("g");
  const nodeGroup = elSvg("g");
  svg.append(edgeGroup, nodeGroup);

  function showTip(evt, htmlText) {
    const rect = svg.parentElement.getBoundingClientRect();
    tooltip.innerHTML = htmlText;
    tooltip.style.left = Math.min(evt.clientX - rect.left + 14, rect.width - 315) + "px";
    tooltip.style.top = Math.max(evt.clientY - rect.top - 12, 8) + "px";
    tooltip.style.opacity = "1";
  }
  function hideTip() { tooltip.style.opacity = "0"; }

  if (mode === "cooccurrence" || mode === "both") {
    coEdges.forEach((e) => {
      const a = nodeMap[e.author_a];
      const b = nodeMap[e.author_b];
      if (!a || !b) return;
      const isVideo = Number(e.same_cctv_67584ms) === 1;
      const stroke = isVideo ? "#D75A4A" : "#D98A3C";
      const sw = isVideo ? 4.2 : Math.max(1.4, Number(e.weight) * 4.2);
      const inSpot = !spotSet || (spotSet.has(e.author_a) && spotSet.has(e.author_b));
      const cls = "net-edge" + (spotSet ? (inSpot ? " highlighted" : " dimmed") : "");
      const line = elSvg("line", {
        x1: a.x, y1: a.y, x2: b.x, y2: b.y,
        stroke, "stroke-width": sw.toFixed(2),
        "stroke-opacity": isVideo ? "0.94" : "0.55",
        "stroke-dasharray": isVideo ? "none" : "5,3",
        class: cls,
      });
      const slotNames = e.shared_slots.split("|").map((s) => SLOT_CN[s] || s).join(" · ");
      line.addEventListener("mousemove", (evt) => {
        showTip(
          evt,
          `<b>暗线共现边</b>：${e.author_a} ↔ ${e.author_b}<br>` +
          `帖子：${e.record_a} ↔ ${e.record_b}（相隔 ${e.delta_hours}h）<br>` +
          `共享 ${e.shared_count} 个槽位：${slotNames}<br>` +
          `权重 w=${e.weight} · TF-IDF 相似度=${e.tfidf_sim}` +
          (isVideo ? `<br><span style="color:#ff9e92">★ 命中 67584ms 同源监控视频指纹</span>` : "")
        );
      });
      line.addEventListener("mouseleave", hideTip);
      edgeGroup.appendChild(line);
    });
  }

  if (mode === "explicit" || mode === "both") {
    explicitEdges.forEach((e) => {
      const s = nodeMap[e.source];
      const t = nodeMap[e.target];
      if (!s || !t) return;
      const isMention = e.edge_type === "mention";
      const mx = (s.x + t.x) / 2 + (t.y - s.y) * 0.12;
      const my = (s.y + t.y) / 2 - (t.x - s.x) * 0.12;
      const inSpot = !spotSet || (spotSet.has(e.source) && spotSet.has(e.target));
      const cls = "net-edge" + (spotSet ? (inSpot ? " highlighted" : " dimmed") : "");
      const path = elSvg("path", {
        d: `M ${s.x} ${s.y} Q ${mx} ${my} ${t.x} ${t.y}`,
        fill: "none",
        stroke: isMention ? "#238B8E" : (e.source_stance === "factcheck_debunk" ? "#2D6DA3" : "#6c7c89"),
        "stroke-width": isMention ? "1.8" : "2.5",
        "stroke-opacity": mode === "both" ? "0.62" : "0.86",
        "stroke-dasharray": isMention ? "3,3" : "none",
        "marker-end": "url(#arrow-exp)",
        class: cls,
      });
      const typeCn = { quote: "引用转发 (Quote)", reply: "直接回复 (Reply)", mention: "正文 @提及" }[e.edge_type] || e.edge_type;
      path.addEventListener("mousemove", (evt) => {
        showTip(
          evt,
          `<b>明线互动边</b>：${e.source} → ${e.target}<br>` +
          `类型：${typeCn} · 帖号 ${e.record_id}（+${e.hours_since_t0}h）`
        );
      });
      path.addEventListener("mouseleave", hideTip);
      edgeGroup.appendChild(path);
    });
  }

  nodes.forEach((n) => {
    if (!activeHandles.has(n.handle)) return;
    const inSpot = !spotSet || spotSet.has(n.handle);
    let cls = "net-node";
    if (n.handle === selectedHandle) cls += " selected";
    if (spotSet) cls += inSpot ? " highlighted" : " dimmed";
    const g = elSvg("g", { class: cls });
    const color = STANCE_COLORS[n.primary_stance] || "#7F8FA0";
    const r = n.in_dataset ? Math.max(8, Math.min(22, 6 + Math.log10(Number(n.total_views) + 100) * 2.1)) : 7.5;

    if (n.handle === "@RealBababanaras" || n.handle === "@NDRRMA_Nepal") {
      g.appendChild(elSvg("circle", { cx: n.x, cy: n.y, r: r + 6, fill: color, opacity: "0.18" }));
    }
    const dot = elSvg("circle", {
      cx: n.x, cy: n.y, r, fill: color, stroke: "#ffffff", "stroke-width": "2", class: "main-dot",
    });
    const label = elSvg("text", { x: n.x, y: n.y - r - 5, "text-anchor": "middle", class: "net-label" });
    label.textContent = n.handle;
    g.append(dot, label);

    g.addEventListener("mousemove", (evt) => {
      showTip(
        evt,
        `<b>${n.handle}</b>（${n.primary_stance_cn}）<br>` +
        `收录帖数：${n.post_count} · 总浏览：${fmtViews(n.total_views)}<br>` +
        `明线度：入 ${n.explicit_in_degree} / 出 ${n.explicit_out_degree} · 暗线共现度：${n.cooccurrence_degree}<br>` +
        `<span style="color:#9ad8da">点击查看完整原帖与核查详情</span>`
      );
    });
    g.addEventListener("mouseleave", hideTip);
    g.addEventListener("click", () => {
      selectedHandle = n.handle;
      renderInspector(n);
      svg.querySelectorAll(".net-node").forEach((el) => el.classList.remove("selected"));
      g.classList.add("selected");
    });
    nodeGroup.appendChild(g);
  });

  const caption = document.getElementById("net-mode-desc");
  if (spotObj) {
    caption.innerHTML = `<b>${escapeHtml(spotObj.title)}：</b>${escapeHtml(spotObj.summary)}`;
    const targetNode = nodeMap[spotObj.inspect];
    if (targetNode) {
      selectedHandle = targetNode.handle;
      renderInspector(targetNode);
    }
  } else if (mode === "cooccurrence") {
    caption.textContent =
      "暗线视图（17 条共现边，已排除辟谣引述帖）：粗实心红线锁定无转发标注的 @RealBababanaras (X09) ↔ @Ujwalthapa12 (X11)；橙虚线团块揭示“291/105 真实名单”被分叉污染的同步结构。";
  } else if (mode === "explicit") {
    caption.textContent =
      "明线视图（14 条跨账号有向边，已剔除 X02 自引用）：清晰呈现跨语种星型辐射、@BlacksmithIndie 寄生劫持千万级爆帖、单跳靶向辟谣及尼泊尔官方协作链。";
  } else {
    caption.textContent =
      "全视角叠加：同时呈现显式引用/回复/提及（带箭头曲线）与隐式槽位/视频指纹共现（虚线与粗红实线），直观对照“明线断链、暗线成团”的隐蔽传播机制。";
  }
}

function renderInspector(node) {
  const box = document.getElementById("node-inspector");
  const posts = DATA.x_posts.filter((p) => p.author_handle === node.handle);
  if (!posts.length) {
    box.innerHTML = `<b>${escapeHtml(node.handle)}</b> · 外部被引用/提及节点（明线被引用/提及 ${node.explicit_in_degree} 次，未收录其原帖全文）。`;
    return;
  }
  const slotHtml = (node.slots ? node.slots.split("|") : [])
    .map((s) => `<span class="slot-pill ${s === "cctv_67584ms" ? "video" : ""}">${SLOT_CN[s] || s}</span>`)
    .join("");
  const postsHtml = posts
    .map((p) => (
      `<div style="margin-top:10px">` +
      `<span class="badge" style="background:${STANCE_COLORS[p.narrative_stance]}">${p.stance_cn}</span>` +
      `<b>${p.record_id}</b> · +${p.hours_since_t0}h (${p.created_at_utc}) · 浏览 ${fmtViews(p.views)} · ` +
      `<a href="${p.post_url}" target="_blank" rel="noopener">原帖链接</a>` +
      `<blockquote>${escapeHtml(p.text_verbatim)}</blockquote>` +
      `</div>`
    ))
    .join("");
  box.innerHTML =
    `<div><b>${escapeHtml(node.handle)}</b> · ${node.primary_stance_cn} · 收录 ${posts.length} 帖 · 累计浏览 ${fmtViews(node.total_views)}</div>` +
    (slotHtml ? `<div style="margin-top:6px">命中特征槽位：${slotHtml}</div>` : "") +
    postsHtml;
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

// Searchable Evidence Explorer (X posts + GDELT news tabs)
let activeTab = "x";
let showLimit = 8;

function renderExplorer() {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === activeTab));
  const stanceSel = document.getElementById("stance-filter");
  const phaseSel = document.getElementById("phase-filter");
  const langSel = document.getElementById("lang-filter");
  stanceSel.hidden = activeTab !== "x";
  phaseSel.hidden = activeTab !== "x";
  langSel.hidden = activeTab !== "x";

  const q = document.getElementById("search-input").value.trim().toLowerCase();
  const container = document.getElementById("records");
  container.replaceChildren();

  if (activeTab === "x") {
    const fStance = stanceSel.value;
    const fPhase = phaseSel.value;
    const fLang = langSel.value;
    const filtered = DATA.x_posts.filter((r) => {
      const hay = `${r.record_id} ${r.author_handle} ${r.text_verbatim} ${r.slots}`.toLowerCase();
      return (!q || hay.includes(q)) &&
        (!fStance || r.narrative_stance === fStance) &&
        (!fPhase || r.phase === fPhase) &&
        (!fLang || r.language === fLang);
    });
    document.getElementById("search-count").textContent = `X 定向核心样本共匹配 ${filtered.length} / ${DATA.x_posts.length} 条 · 当前显示 ${Math.min(showLimit, filtered.length)} 条`;
    filtered.slice(0, showLimit).forEach((r) => {
      const art = document.createElement("article");
      art.className = "record";
      const slots = r.slots ? r.slots.split("|").map((s) => `<span class="slot-pill ${s === "cctv_67584ms" ? "video" : ""}">${SLOT_CN[s] || s}</span>`).join("") : "";
      const corr = r.stance_correction_reason ? `<div class="muted" style="font-size:12px;margin-top:4px">★ 立场人工纠偏（原标 ${r.claimed_cause_raw}）：${escapeHtml(r.stance_correction_reason)}</div>` : "";
      art.innerHTML =
        `<span class="badge" style="background:${STANCE_COLORS[r.narrative_stance]}">${r.stance_cn}</span>` +
        `<span class="meta"><b>${r.record_id}</b> · ${escapeHtml(r.author_handle)} · +${r.hours_since_t0}h (${r.created_at_utc}) · ${r.phase} · 语种 ${r.language.toUpperCase()} · 浏览 ${fmtViews(r.views)} · <a href="${r.post_url}" target="_blank" rel="noopener">原链</a></span>` +
        `<p>${escapeHtml(r.text_verbatim)}</p>` +
        (slots ? `<div>${slots}</div>` : "") + corr;
      container.appendChild(art);
    });
    document.getElementById("more").hidden = showLimit >= filtered.length;
  } else {
    const filtered = DATA.gdelt_matches.filter((r) => {
      const hay = `${r.title} ${r.url} ${r.lang} ${r.file_stamp}`.toLowerCase();
      return !q || hay.includes(q);
    });
    document.getElementById("search-count").textContent = `GDELT 全球新闻共匹配 ${filtered.length} / ${DATA.gdelt_matches.length} 篇 · 当前显示 ${Math.min(showLimit, filtered.length)} 篇`;
    filtered.slice(0, showLimit).forEach((r) => {
      const art = document.createElement("article");
      art.className = "record";
      const tags = [];
      if (Number(r.T3_glacier_science)) tags.push(`<span class="badge" style="background:#238B8E">冰川/崩塌标题</span>`);
      if (Number(r.T1_dam_rumor)) tags.push(`<span class="badge" style="background:#D75A4A">含 dam 词（需核读）</span>`);
      if (Number(r.T2_seismic_misread)) tags.push(`<span class="badge" style="background:#9B6BB1">地震词</span>`);
      if (!tags.length) tags.push(`<span class="badge" style="background:#7F8FA0">综合灾情报道</span>`);
      art.innerHTML =
        tags.join("") +
        `<span class="meta">切片 ${r.file_stamp} · 语种 ${escapeHtml(r.lang)} · <a href="${r.url}" target="_blank" rel="noopener">原文链接</a></span>` +
        `<p>${escapeHtml(r.title)}</p>`;
      container.appendChild(art);
    });
    document.getElementById("more").hidden = showLimit >= filtered.length;
  }
}

document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => renderNetwork(btn.dataset.mode, null));
});
document.querySelectorAll(".spot-btn[data-spot]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const key = btn.dataset.spot;
    renderNetwork(SPOTLIGHTS[key].mode, key);
  });
});
const resetBtn = document.getElementById("spot-reset");
if (resetBtn) {
  resetBtn.addEventListener("click", () => renderNetwork(currentMode, null));
}
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => { activeTab = btn.dataset.tab; showLimit = 8; renderExplorer(); });
});
["search-input", "stance-filter", "phase-filter", "lang-filter"].forEach((id) => {
  document.getElementById(id).addEventListener("input", () => { showLimit = 8; renderExplorer(); });
});
document.getElementById("more").addEventListener("click", () => { showLimit += 20; renderExplorer(); });

renderNetwork("cooccurrence", null);
const initNode = DATA.network.nodes.find((n) => n.handle === "@RealBababanaras") || DATA.network.nodes[0];
if (initNode) renderInspector(initNode);
renderExplorer();
