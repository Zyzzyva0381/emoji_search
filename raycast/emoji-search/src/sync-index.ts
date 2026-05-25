import { Toast, showToast } from "@raycast/api";
import { apiBaseUrl, postJson } from "./shared";

export default async function SyncIndex() {
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
