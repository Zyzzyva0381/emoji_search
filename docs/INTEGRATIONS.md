# Integrations

## Recommendation

Prefer Raycast for the first desktop integration.

Why:

- Raycast extensions are TypeScript/React commands with a first-class list UI.
- The API can copy text or files to the clipboard and can paste content into the frontmost app.
- Built-in actions cover copy/open flows, so the extension can stay small.

Alfred remains viable, especially if the target user already owns Powerpack, but the image search experience would be more workflow-like: Script Filter JSON, file actions, and separate copy/paste outputs.

## Raycast Extension Shape

Command: `Search Emoji`

Flow:

1. User types a natural-language query.
2. Extension calls local FastAPI: `POST /api/search`.
3. Results render as a list/grid-like list with thumbnail, score, `wechat_keyword`, `manual_tags`, and `usage_context`.
4. Primary action copies the image file to clipboard.
5. Secondary action pastes into the frontmost app.
6. Other actions copy path, open preview, or copy tags.

## Alfred Workflow Shape

Input: Script Filter

Flow:

1. Script Filter calls `POST /api/search`.
2. JSON result items use `title`, `subtitle`, `arg`, `quicklookurl`, and `action.file`.
3. A Copy to Clipboard output copies the selected file/path.

This is simpler if the user only needs file paths. It is less ergonomic for rich thumbnail search and custom actions.
