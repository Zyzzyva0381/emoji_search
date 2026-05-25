import {
  Action,
  ActionPanel,
  Color,
  Detail,
  Icon,
  Keyboard,
  List,
  Toast,
  showToast,
} from "@raycast/api";
import { useEffect, useMemo, useState } from "react";
import {
  SearchResult,
  absoluteImagePath,
  apiBaseUrl,
  copyImage,
  openFile,
  pasteImage,
  postJson,
  preferences,
  projectRoot,
  revealInFinder,
  resultMarkdown,
  resultSubtitle,
  resultTitle,
} from "./shared";

export default function SearchEmoji() {
  const [searchText, setSearchText] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | undefined>();
  const topK = useMemo(() => {
    const value = Number.parseInt((preferences().topK || "").trim(), 10);
    return Number.isFinite(value) && value > 0 ? value : 20;
  }, []);

  useEffect(() => {
    const query = searchText.trim();
    if (!query) {
      setResults([]);
      setError(undefined);
      setIsLoading(false);
      return;
    }

    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setIsLoading(true);
      setError(undefined);
      try {
        const response = await fetch(`${apiBaseUrl()}/api/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            top_k: topK,
            score_mode: "max",
            fields: [],
          }),
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(
            `${response.status} ${response.statusText}: ${await response.text()}`,
          );
        }
        setResults((await response.json()) as SearchResult[]);
      } catch (err) {
        if (!controller.signal.aborted) {
          const message = err instanceof Error ? err.message : String(err);
          setError(message);
          setResults([]);
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoading(false);
        }
      }
    }, 180);

    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [searchText, topK]);

  if (error) {
    return (
      <Detail
        markdown={`# Emoji Search 连接失败\n\n\`${error}\`\n\n先启动本地服务：\n\n\`\`\`bash\ncd ${projectRoot()}\nuv run uvicorn backend:app --reload\n\`\`\``}
        actions={
          <ActionPanel>
            <Action.OpenInBrowser
              title="Open Local App"
              url={`${apiBaseUrl()}/app`}
            />
            <Action
              title="Retry"
              icon={Icon.ArrowClockwise}
              onAction={() => setError(undefined)}
            />
          </ActionPanel>
        }
      />
    );
  }

  return (
    <List
      isLoading={isLoading}
      onSearchTextChange={setSearchText}
      searchBarPlaceholder="搜索表情：无语、摆烂、等暑假、南大点赞..."
      throttle
      isShowingDetail
    >
      {!searchText.trim() ? (
        <List.EmptyView
          icon={Icon.MagnifyingGlass}
          title="输入关键词搜索本地表情库"
          description="结果来自本机 FastAPI 服务，不上传图片。"
        />
      ) : results.length === 0 && !isLoading ? (
        <List.EmptyView icon={Icon.FaceSad} title="没有匹配结果" />
      ) : (
        results.map((result) => <EmojiItem key={result.path} result={result} />)
      )}
    </List>
  );
}

function EmojiItem({ result }: { result: SearchResult }) {
  const filePath = absoluteImagePath(result.path);
  const fields = result.fields;

  return (
    <List.Item
      title={resultTitle(result)}
      subtitle={resultSubtitle(result)}
      icon={{ source: `${apiBaseUrl()}${result.url}` }}
      detail={
        <List.Item.Detail
          markdown={resultMarkdown(result)}
          metadata={<Metadata result={result} />}
        />
      }
      accessories={[
        { text: fields.character_name },
        { tag: { value: result.best_field, color: Color.Blue } },
        { text: result.score.toFixed(2) },
      ]}
      actions={
        <ActionPanel>
          <ActionPanel.Section>
            <Action
              title="Copy Image"
              icon={Icon.CopyClipboard}
              onAction={() => copyImage(filePath)}
            />
            <Action
              title="Paste Image"
              icon={Icon.Clipboard}
              shortcut={{ modifiers: ["cmd"], key: "enter" }}
              onAction={() => pasteImage(filePath)}
            />
          </ActionPanel.Section>
          <ActionPanel.Section>
            <Action.CopyToClipboard
              title="Copy Path"
              content={filePath}
              shortcut={{ modifiers: ["cmd", "shift"], key: "c" }}
            />
            <Action.CopyToClipboard
              title="Copy Tags"
              content={fields.manual_tags || ""}
            />
            <Action.OpenInBrowser
              title="Open Local Preview"
              url={`${apiBaseUrl()}${result.url}`}
            />
            <Action
              title="Open Image"
              icon={Icon.Image}
              onAction={() => openFile(filePath)}
            />
            <Action
              title="Reveal in Finder"
              icon={Icon.Finder}
              shortcut={Keyboard.Shortcut.Common.Open}
              onAction={() => revealInFinder(filePath)}
            />
            <Action
              title="Sync Index"
              icon={Icon.ArrowClockwise}
              onAction={() => syncIndex()}
            />
          </ActionPanel.Section>
        </ActionPanel>
      }
    />
  );
}

function Metadata({ result }: { result: SearchResult }) {
  const fields = result.fields;
  return (
    <List.Item.Detail.Metadata>
      <List.Item.Detail.Metadata.Label
        title="Score"
        text={result.score.toFixed(4)}
      />
      <List.Item.Detail.Metadata.Label
        title="Best Field"
        text={result.best_field}
      />
      <List.Item.Detail.Metadata.Separator />
      <List.Item.Detail.Metadata.Label
        title="微信含义词"
        text={fields.wechat_keyword || ""}
      />
      <List.Item.Detail.Metadata.Label
        title="人工标签"
        text={fields.manual_tags || ""}
      />
      <List.Item.Detail.Metadata.Label
        title="角色"
        text={fields.character_name || ""}
      />
      <List.Item.Detail.Metadata.Label
        title="表情"
        text={fields.expression || ""}
      />
      <List.Item.Detail.Metadata.Label
        title="动作"
        text={fields.action || ""}
      />
      <List.Item.Detail.Metadata.Label
        title="场景"
        text={fields.usage_context || ""}
      />
    </List.Item.Detail.Metadata>
  );
}

async function syncIndex() {
  await showToast({ style: Toast.Style.Animated, title: "同步索引中..." });
  const result = await postJson<{
    added_vectors: number;
    removed_stale_vectors: number;
  }>(`${apiBaseUrl()}/api/index/sync`, {});
  await showToast({
    style: Toast.Style.Success,
    title: "索引已同步",
    message: `新增 ${result.added_vectors}，移除 ${result.removed_stale_vectors}`,
  });
}
