#用VSCode套件 Live Server 看輸出的html

import sys
from pathlib import Path
import html


def build_body_html(text: str) -> str:
    """
    把純文字轉成比較有結構的 HTML：
    - 「發文機關：...」這種 → <p><span class="label">發文機關：</span>內容</p>
    - 「說明：」「一、...」「二、...」 → <h2 class="section-title">
    - 「- xxx」開頭 → 條列式樣式
    - 其他 → 一般段落
    """
    lines_html: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")

        # 全空行 → 拉開一點距離
        if line.strip() == "":
            lines_html.append('<p class="spacer"></p>')
            continue

        stripped = line.strip()
        esc = html.escape(stripped)

        # 1️⃣ 像「說明：」「附註：」這種尾巴是冒號、而且不長的行 → 當成小標題
        if stripped.endswith("：") and len(stripped) <= 20:
            lines_html.append(f'<h2 class="section-title">{esc}</h2>')
            continue

        # 2️⃣ 像「一、說明」「二、查詢對象」這種 → 當成段落標題
        if len(stripped) >= 2 and stripped[0] in "一二三四五六七八九十" and stripped[1] == "、":
            lines_html.append(f'<h2 class="section-title">{esc}</h2>')
            continue

        # 3️⃣ - 開頭的行 → 條列（自己加個圓點）
        if stripped.startswith(("- ", "－")):
            # 去掉前面的 dash
            content = stripped.lstrip("-－").strip()
            content_esc = html.escape(content)
            lines_html.append(f'<p class="bullet">• {content_esc}</p>')
            continue

        # 4️⃣ 一般「標籤：內容」格式 → 粗體標籤＋一般內容
        if "：" in stripped:
            label, value = stripped.split("：", 1)
            label_esc = html.escape(label)
            value_esc = html.escape(value.strip())
            lines_html.append(
                f'<p><span class="label">{label_esc}：</span>{value_esc}</p>'
            )
            continue

        # 5️⃣ 其餘 → 普通段落
        lines_html.append(f"<p>{esc}</p>")

    return "\n".join(lines_html)


def txt_to_html(input_path: str, output_path: str | None = None, title: str | None = None) -> Path:
    in_path = Path(input_path)

    if not in_path.exists():
        raise FileNotFoundError(f"找不到檔案：{in_path}")

    # 預設輸出成同路徑同檔名的 .html
    out_path = Path(output_path) if output_path else in_path.with_suffix(".html")

    # 預設標題用檔名
    if title is None:
        title = in_path.stem

    # 讀取文字（如果是 cp950/Big5 再改 encoding）
    text = in_path.read_text(encoding="utf-8")

    body_html = build_body_html(text)

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <title>{html.escape(title)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 24px;
        }}
        .page {{
            max-width: 960px;
            margin: 0 auto;
        }}
        .card {{
            background-color: #ffffff;
            border-radius: 10px;
            padding: 24px 28px 28px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
            line-height: 1.7;
            border: 1px solid #e5e5e5;
        }}
        h1 {{
            font-size: 24px;
            margin-top: 0;
            margin-bottom: 12px;
            color: #222;
        }}
        .subtitle {{
            font-size: 13px;
            color: #777;
            margin-bottom: 4px;
        }}
        .section-title {{
            font-size: 16px;
            margin-top: 18px;
            margin-bottom: 8px;
            color: #1565c0;
            border-left: 4px solid #1565c0;
            padding-left: 8px;
        }}
        p {{
            margin: 4px 0;
            font-size: 14px;
            color: #333;
        }}
        .spacer {{
            margin: 8px 0;
        }}
        .label {{
            font-weight: 600;
            color: #555;
            display: inline-block;
            min-width: 6em;
        }}
        .bullet {{
            margin-left: 1.2em;
            text-indent: -0.8em;
        }}
        .footer-note {{
            margin-top: 18px;
            padding-top: 10px;
            border-top: 1px dashed #ddd;
            font-size: 12px;
            color: #888;
        }}
    </style>
</head>
<body>
<div class="page">
  <div class="card">
    <h1>{html.escape(title)}</h1>
    <div class="subtitle">自動轉換 .txt 檔，僅供閱讀使用。</div>
    <hr style="border:none;border-top:1px solid #eee;margin:10px 0 14px;">
    <div class="content">
{body_html}
    </div>
    <div class="footer-note">
      ※ 實際權利義務仍以原始公文及正式檔案為準。
    </div>
  </div>
</div>
</body>
</html>
"""

    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


def main():
    if len(sys.argv) < 2:
        print("使用方式：")
        print("  python txt_to_html.py input.txt [output.html]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None

    out_path = txt_to_html(input_file, output_file)
    print(f"轉換完成，已輸出：{out_path}")


if __name__ == "__main__":
    main()
