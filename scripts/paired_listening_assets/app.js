"use strict";
const catalog = JSON.parse(document.getElementById("catalog").textContent);
const key = "stem-pair-drafts:" + catalog.bundle_sha256;
const labels = {
  mix: "原曲",
  htdemucs_vocals: "HTDemucs · 人声",
  htdemucs_accompaniment: "HTDemucs · 伴奏",
  kim_melband_vocals: "Mel-Band · 人声",
  kim_melband_accompaniment: "Mel-Band · 伴奏",
};
const fields = ["alignment", "htdemucs", "kim_melband", "rights", "notes"];
let drafts = {}, selected = catalog.records[0]?.song_id, windowIndex = 0;
let storageError = false;
try { drafts = JSON.parse(localStorage.getItem(key) || "{}"); } catch { storageError = true; }
if (!drafts || typeof drafts !== "object" || Array.isArray(drafts)) drafts = {};
const byId = new Map(catalog.records.map(r => [r.song_id, r]));
const el = id => document.getElementById(id);
const time = sample => {
  const seconds = sample / catalog.sample_rate;
  return Math.floor(seconds / 60) + ":" + Math.floor(seconds % 60).toString().padStart(2, "0");
};
function hasDraft(id) { return fields.some(f => drafts[id]?.[f] && drafts[id][f] !== "pending"); }
function listSongs() {
  el("songs").replaceChildren();
  const query = el("search").value.toLowerCase();
  let matched = 0;
  for (const row of catalog.records) {
    if (!(row.title + row.song_id).toLowerCase().includes(query)) continue;
    matched++;
    const button = document.createElement("button");
    button.className = "song" + (row.song_id === selected ? " active" : "");
    button.textContent = (hasDraft(row.song_id) ? "◌ " : "") + row.song_id + " · " + row.title;
    button.setAttribute("aria-pressed", String(row.song_id === selected));
    button.onclick = () => { selected = row.song_id; windowIndex = 0; render(); };
    el("songs").append(button);
  }
  el("empty").hidden = matched > 0;
  el("progress").textContent = `${catalog.records.filter(r => hasDraft(r.song_id)).length}/24 已写草稿`;
}
function players(row) {
  const activeWindow = row.windows[windowIndex];
  el("players").replaceChildren();
  // Lead with accompaniments; vocals are diagnostic, not the product output.
  const order = ["mix", "htdemucs_accompaniment", "kim_melband_accompaniment", "htdemucs_vocals", "kim_melband_vocals"];
  for (const name of order) {
    const box = document.createElement("div"), heading = document.createElement("h3"), audio = document.createElement("audio");
    heading.textContent = labels[name];
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = `${row.song_id}/w${windowIndex + 1}/${name}.wav`;
    audio.setAttribute("aria-label", labels[name] + "播放器");
    audio.onplay = () => document.querySelectorAll("audio").forEach(other => { if (other !== audio) other.pause(); });
    audio.onerror = () => { el("saved").textContent = "音频读取失败。请重新打开本机试听服务，或检查播放副本是否完整。"; };
    box.append(heading, audio);
    el("players").append(box);
  }
  el("position").textContent = `原曲 ${time(activeWindow.start_sample)} – ${time(activeWindow.start_sample + activeWindow.samples)} · ${(activeWindow.samples / catalog.sample_rate).toFixed(1)} 秒`;
}
function render() {
  document.querySelectorAll("audio").forEach(a => a.pause());
  const row = byId.get(selected);
  if (!row) { el("detail").hidden = true; return; }
  el("title").textContent = row.title;
  el("song-id").textContent = row.song_id;
  el("gain").textContent = `本首五路、三个位置共用播放增益 ${row.gain.toFixed(6)}；原始 FLOAT32 标签未改。`;
  el("windows").replaceChildren();
  row.windows.forEach((w, index) => {
    const button = document.createElement("button");
    button.textContent = ["前段 10%", "中段 50%", "后段 90%"][index];
    button.className = index === windowIndex ? "active" : "";
    button.setAttribute("aria-pressed", String(index === windowIndex));
    button.onclick = () => { windowIndex = index; render(); };
    el("windows").append(button);
  });
  el("hints").replaceChildren();
  for (const [teacher, hints] of Object.entries(row.hints)) {
    const p = document.createElement("p");
    p.textContent = `${teacher}: ${hints.length ? hints.join("；") : "无额外技术提示，仍需试听"}`;
    el("hints").append(p);
  }
  for (const field of fields) el(field).value = drafts[selected]?.[field] || (field === "notes" ? "" : "pending");
  el("saved").textContent = storageError ? "浏览器存储不可用；请及时导出当前草稿。" : "只保存草稿，不回写标签、不批准训练。";
  players(row); listSongs();
}
function save() {
  const draft = {};
  for (const field of fields) draft[field] = el(field).value;
  draft.updated_at = new Date().toISOString();
  drafts[selected] = draft;
  try { localStorage.setItem(key, JSON.stringify(drafts)); el("saved").textContent = "草稿已保存到当前浏览器。训练资格仍未批准。"; }
  catch { storageError = true; el("saved").textContent = "浏览器存储不可用；请导出草稿备份。"; }
  listSongs();
}
for (const field of fields) el(field).addEventListener(field === "notes" ? "input" : "change", save);
el("search").oninput = listSongs;
el("export").onclick = () => {
  const report = {schema: 1, purpose: "Human listening draft only; requires separate validation and signed approval",
    bundle_sha256: catalog.bundle_sha256, exported_at: new Date().toISOString(), training_authorized: false,
    original_label_entries_modified: false, records: catalog.records.map(row => ({song_id: row.song_id,
      listening_windows: row.windows, review_draft: drafts[row.song_id] || null}))};
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: "application/json"}));
  const link = document.createElement("a"); link.href = url; link.download = "paired_listening_draft.json"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
render();
