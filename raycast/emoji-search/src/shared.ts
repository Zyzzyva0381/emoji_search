import { Clipboard, Toast, getPreferenceValues, showToast } from "@raycast/api";
import crypto from "node:crypto";
import { execFile } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export type Preferences = {
  apiBaseUrl: string;
  projectRoot?: string;
  topK?: string;
};

export type EmojiFields = {
  image_composition?: string;
  character_name?: string;
  expression?: string;
  action?: string;
  subjective_emotion?: string;
  text_in_image?: string;
  usage_context?: string;
  wechat_keyword?: string;
  manual_tags?: string;
  notes?: string;
};

export type SearchResult = {
  path: string;
  url: string;
  score: number;
  best_field: string;
  field_scores: Record<string, number>;
  fields: EmojiFields;
};

export function preferences(): Preferences {
  return getPreferenceValues<Preferences>();
}

export function apiBaseUrl(): string {
  return preferences().apiBaseUrl.replace(/\/+$/, "");
}

export function configuredProjectRoot(): string | undefined {
  const explicitRoot = preferences().projectRoot?.trim();
  if (explicitRoot) {
    return explicitRoot;
  }
  return undefined;
}

export function projectRoot(): string {
  return configuredProjectRoot() || fallbackProjectRoot() || "";
}

export function absoluteImagePath(relativePath: string): string {
  const root = projectRoot();
  return root ? path.resolve(root, relativePath) : relativePath;
}

export function catalogPath(): string {
  const root = projectRoot();
  if (!root) {
    throw new Error("请在 Raycast 扩展偏好里设置 Project Root");
  }
  return path.resolve(root, "emoji_catalog.csv");
}

export function resultTitle(result: SearchResult): string {
  const fields = result.fields;
  return (
    clean(fields.wechat_keyword) ||
    clean(fields.text_in_image) ||
    clean(fields.character_name) ||
    path.basename(result.path)
  );
}

export function resultSubtitle(result: SearchResult): string {
  const fields = result.fields;
  return [fields.usage_context, fields.manual_tags, fields.expression]
    .map(clean)
    .filter(Boolean)
    .join("  |  ");
}

export function resultMarkdown(result: SearchResult): string {
  const fields = result.fields;
  const lines = [
    `![preview](${apiBaseUrl()}${result.url})`,
    "",
    `**路径**: \`${result.path}\``,
    `**分数**: ${result.score.toFixed(4)}  **最佳字段**: ${result.best_field}`,
    "",
    fieldLine("微信含义词", fields.wechat_keyword),
    fieldLine("人工标签", fields.manual_tags),
    fieldLine("角色", fields.character_name),
    fieldLine("表情", fields.expression),
    fieldLine("动作", fields.action),
    fieldLine("情绪", fields.subjective_emotion),
    fieldLine("图中文字", fields.text_in_image),
    fieldLine("场景", fields.usage_context),
    fieldLine("构成", fields.image_composition),
    fieldLine("补充", fields.notes),
  ].filter(Boolean);
  return lines.join("\n");
}

export async function copyImage(result: SearchResult): Promise<void> {
  const filePath = await fileForImageAction(result);
  await Clipboard.copy({ file: filePath });
  await showToast({ style: Toast.Style.Success, title: "已复制图片" });
}

export async function pasteImage(result: SearchResult): Promise<void> {
  const filePath = await fileForImageAction(result);
  await Clipboard.paste({ file: filePath });
}

export async function openImage(result: SearchResult): Promise<void> {
  await openFile(await fileForImageAction(result));
}

export async function revealImage(result: SearchResult): Promise<void> {
  await revealInFinder(await fileForImageAction(result));
}

export async function revealInFinder(filePath: string): Promise<void> {
  await execFileAsync("/usr/bin/open", ["-R", filePath]);
}

export async function openFile(filePath: string): Promise<void> {
  await execFileAsync("/usr/bin/open", [filePath]);
}

export async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(
      `${response.status} ${response.statusText}: ${await response.text()}`,
    );
  }
  return (await response.json()) as T;
}

function fieldLine(label: string, value?: string): string {
  const text = clean(value);
  return text ? `**${label}**: ${text}` : "";
}

async function fileForImageAction(result: SearchResult): Promise<string> {
  const localPath = await existingLocalImagePath(result.path);
  return localPath || downloadPreviewToCache(result);
}

async function existingLocalImagePath(
  relativePath: string,
): Promise<string | undefined> {
  const roots = [configuredProjectRoot(), fallbackProjectRoot()].filter(
    Boolean,
  ) as string[];
  for (const root of roots) {
    try {
      return await fs.realpath(path.resolve(root, relativePath));
    } catch {
      // Try the next possible root.
    }
  }
  return undefined;
}

async function downloadPreviewToCache(result: SearchResult): Promise<string> {
  const cacheDir = path.join(os.tmpdir(), "emoji-search-raycast");
  await fs.mkdir(cacheDir, { recursive: true });
  const ext = path.extname(result.path) || ".png";
  const hash = crypto.createHash("sha1").update(result.path).digest("hex");
  const outputPath = path.join(cacheDir, `${hash}${ext}`);

  try {
    await fs.access(outputPath);
    return outputPath;
  } catch {
    // Cache miss; download from the local FastAPI image endpoint.
  }

  const response = await fetch(`${apiBaseUrl()}${result.url}`);
  if (!response.ok) {
    throw new Error(
      `无法下载图片预览 ${response.status} ${response.statusText}`,
    );
  }
  await fs.writeFile(outputPath, Buffer.from(await response.arrayBuffer()));
  return outputPath;
}

function fallbackProjectRoot(): string | undefined {
  const root = path.resolve(process.cwd(), "../..");
  return root === "/" ? undefined : root;
}

function clean(value?: string): string {
  const text = (value || "").trim();
  if (!text || ["NONE", "none", "无", "未知", "null", "N/A"].includes(text)) {
    return "";
  }
  return text;
}
