import { Clipboard, Toast, getPreferenceValues, showToast } from "@raycast/api";
import { execFile } from "node:child_process";
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

export function projectRoot(): string {
  const explicitRoot = preferences().projectRoot?.trim();
  if (explicitRoot) {
    return explicitRoot;
  }
  return path.resolve(process.cwd(), "../..");
}

export function absoluteImagePath(relativePath: string): string {
  return path.resolve(projectRoot(), relativePath);
}

export function catalogPath(): string {
  return path.resolve(projectRoot(), "emoji_catalog.csv");
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

export async function copyImage(filePath: string): Promise<void> {
  await Clipboard.copy({ file: filePath });
  await showToast({ style: Toast.Style.Success, title: "已复制图片" });
}

export async function pasteImage(filePath: string): Promise<void> {
  await Clipboard.paste({ file: filePath });
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

function clean(value?: string): string {
  const text = (value || "").trim();
  if (!text || ["NONE", "none", "无", "未知", "null", "N/A"].includes(text)) {
    return "";
  }
  return text;
}
