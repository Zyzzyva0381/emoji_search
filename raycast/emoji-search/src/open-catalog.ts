import { Toast, showToast } from "@raycast/api";
import { catalogPath, openFile } from "./shared";

export default async function OpenCatalog() {
  try {
    await openFile(catalogPath());
    await showToast({
      style: Toast.Style.Success,
      title: "已打开 emoji_catalog.csv",
    });
  } catch (err) {
    await showToast({
      style: Toast.Style.Failure,
      title: "无法打开 emoji_catalog.csv",
      message: err instanceof Error ? err.message : String(err),
    });
  }
}
