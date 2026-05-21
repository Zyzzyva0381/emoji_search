const fieldLabels = {
  image_composition: "图片构成",
  character_name: "角色",
  expression: "表情",
  action: "动作",
  subjective_emotion: "主观情绪",
  text_in_image: "图中文字",
  notes: "补充",
};

let fields = [];
let images = [];
let currentItems = [];
let previewItem = null;

const $ = (id) => document.getElementById(id);

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.classList.add("show");
  window.clearTimeout(node._timer);
  node._timer = window.setTimeout(() => node.classList.remove("show"), 2600);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  return response.json();
}

function setBusy(button, busy) {
  button.disabled = busy;
}

function selectedFields() {
  return [...document.querySelectorAll(".field-check:checked")].map((input) => input.value);
}

function renderFields() {
  const row = $("fieldRow");
  row.innerHTML = fields.map((field) => `
    <label class="field-chip">
      <input class="field-check" type="checkbox" value="${field}">
      <span>${fieldLabels[field] || field}</span>
    </label>
  `).join("");
}

function scoreText(item) {
  if (typeof item.score !== "number") return item.indexed ? "已索引" : "未索引";
  return `${item.best_field ? fieldLabels[item.best_field] || item.best_field : "score"} · ${item.score.toFixed(3)}`;
}

function itemName(item) {
  return item.path.split("/").pop();
}

function renderGrid(items, title = "全部图片") {
  currentItems = items;
  $("resultTitle").textContent = title;
  $("resultMeta").textContent = `${items.length} 张`;
  $("imageGrid").innerHTML = items.map((item, index) => `
    <article class="card">
      <img class="thumb" src="${item.url}" alt="${item.path}" data-index="${index}">
      <div class="card-body">
        <div class="card-title">${itemName(item)}</div>
        <div class="card-meta">${scoreText(item)}</div>
        <div class="card-actions">
          <button data-action="copy" data-index="${index}" type="button">复制</button>
          <button data-action="${item.indexed === false ? "index" : "delete-index"}" data-index="${index}" type="button">${item.indexed === false ? "索引" : "删索引"}</button>
          <button data-action="preview" data-index="${index}" type="button">预览</button>
          <button class="danger" data-action="delete-image" data-index="${index}" type="button">删除</button>
        </div>
      </div>
    </article>
  `).join("");
}

async function loadStatus() {
  const status = await api("/api/status");
  $("totalImages").textContent = status.total_images;
  $("captionedImages").textContent = status.captioned_images;
  $("indexedImages").textContent = status.vector_indexed_images;
  $("modelState").textContent = status.loaded_models.length
    ? `模型已加载：${status.loaded_models.join(", ")}`
    : "模型未加载";
}

async function loadImages() {
  images = await api("/api/images");
  renderGrid(images);
}

async function init() {
  const fieldResponse = await api("/api/fields");
  fields = fieldResponse.fields;
  renderFields();
  await Promise.all([loadStatus(), loadImages()]);
}

async function search() {
  const query = $("queryInput").value.trim();
  if (!query) {
    renderGrid(images);
    return;
  }
  const results = await api("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      fields: selectedFields(),
      top_k: Number($("topKInput").value || 24),
      score_mode: $("scoreModeInput").value,
    }),
  });
  renderGrid(results, "搜索结果");
}

async function copyImage(item) {
  const response = await fetch(item.url);
  const blob = await response.blob();
  if (navigator.clipboard && window.ClipboardItem) {
    await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
    toast("图片已复制");
    return;
  }
  await navigator.clipboard.writeText(new URL(item.url, window.location.href).href);
  toast("浏览器不支持复制图片，已复制链接");
}

function openPreview(item) {
  previewItem = item;
  $("previewPath").textContent = item.path;
  $("previewScore").textContent = scoreText(item);
  $("previewImage").src = item.url;
  const fieldData = item.fields || {};
  $("previewFields").innerHTML = fields.map((field) => `
    <div>
      <dt>${fieldLabels[field] || field}</dt>
      <dd>${fieldData[field] || "NONE"}</dd>
    </div>
  `).join("");
  $("previewDialog").showModal();
}

async function deleteImage(item) {
  if (!confirm(`删除图片 ${itemName(item)}？`)) return;
  await api(`/api/images/${encodeURIComponent(itemName(item))}`, { method: "DELETE" });
  toast("图片已删除");
  await Promise.all([loadStatus(), loadImages()]);
}

async function deleteIndex(item) {
  await api(`/api/index/${encodeURIComponent(itemName(item))}`, { method: "DELETE" });
  toast("索引已删除");
  await Promise.all([loadStatus(), loadImages()]);
}

async function indexImage(item) {
  await api(`/api/index/${encodeURIComponent(itemName(item))}`, { method: "POST" });
  toast("索引已更新");
  await Promise.all([loadStatus(), loadImages()]);
}

async function uploadImages() {
  const input = $("fileInput");
  if (!input.files.length) {
    toast("请选择图片");
    return;
  }
  const button = $("uploadBtn");
  setBusy(button, true);
  try {
    for (const file of input.files) {
      const form = new FormData();
      form.append("file", file);
      const autoIndex = $("autoIndexInput").checked ? "true" : "false";
      await api(`/api/images?auto_index=${autoIndex}`, {
        method: "POST",
        body: form,
      });
    }
    input.value = "";
    toast("上传完成");
    await Promise.all([loadStatus(), loadImages()]);
  } finally {
    setBusy(button, false);
  }
}

async function withButton(button, action, doneMessage) {
  setBusy(button, true);
  try {
    const result = await action();
    if (doneMessage) toast(typeof doneMessage === "function" ? doneMessage(result) : doneMessage);
    await Promise.all([loadStatus(), loadImages()]);
  } finally {
    setBusy(button, false);
  }
}

$("imageGrid").addEventListener("click", async (event) => {
  const target = event.target;
  const index = target.dataset.index;
  if (index === undefined) return;
  const item = currentItems[Number(index)];
  try {
    if (target.matches(".thumb") || target.dataset.action === "preview") openPreview(item);
    if (target.dataset.action === "copy") await copyImage(item);
    if (target.dataset.action === "delete-image") await deleteImage(item);
    if (target.dataset.action === "delete-index") await deleteIndex(item);
    if (target.dataset.action === "index") await indexImage(item);
  } catch (error) {
    toast(error.message);
  }
});

$("searchBtn").addEventListener("click", () => search().catch((error) => toast(error.message)));
$("queryInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") search().catch((error) => toast(error.message));
});
$("showAllBtn").addEventListener("click", () => renderGrid(images));
$("refreshBtn").addEventListener("click", () => Promise.all([loadStatus(), loadImages()]).catch((error) => toast(error.message)));
$("uploadBtn").addEventListener("click", () => uploadImages().catch((error) => toast(error.message)));
$("syncIndexBtn").addEventListener("click", (event) => withButton(event.currentTarget, () => api("/api/index/sync", { method: "POST" }), (result) => `补 ${result.added_vectors}，删 ${result.removed_stale_vectors}`));
$("loadModelBtn").addEventListener("click", (event) => withButton(
  event.currentTarget,
  () => api("/api/model/load", { method: "POST" }),
  (result) => `已加载 ${result.model}，${result.elapsed_seconds}s`
));
$("unloadModelBtn").addEventListener("click", (event) => withButton(event.currentTarget, () => api("/api/model/unload", { method: "POST" }), "模型已卸载"));
$("deleteAllIndexBtn").addEventListener("click", (event) => {
  if (!confirm("删除全部向量索引？caption 描述会保留。")) return;
  withButton(event.currentTarget, () => api("/api/index", { method: "DELETE" }), "全部向量索引已删除").catch((error) => toast(error.message));
});
$("closePreviewBtn").addEventListener("click", () => $("previewDialog").close());
$("copyPreviewBtn").addEventListener("click", () => previewItem && copyImage(previewItem).catch((error) => toast(error.message)));
$("deletePreviewBtn").addEventListener("click", async () => {
  if (!previewItem) return;
  $("previewDialog").close();
  try {
    await deleteImage(previewItem);
  } catch (error) {
    toast(error.message);
  }
});

init().catch((error) => toast(error.message));
