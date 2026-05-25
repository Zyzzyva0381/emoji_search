import { Toast, showToast } from "@raycast/api";
import { catalogPath, openFile } from "./shared";

export default async function OpenCatalog() {
  await openFile(catalogPath());
  await showToast({
    style: Toast.Style.Success,
    title: "已打开 emoji_catalog.csv",
  });
}
