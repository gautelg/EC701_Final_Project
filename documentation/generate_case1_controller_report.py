from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "case1_controller_report.md"
PDF_PATH = ROOT / "case1_controller_report.pdf"


def clean_markdown_line(line: str) -> str:
    stripped = line.rstrip()
    if stripped.startswith("# "):
        return stripped[2:].upper()
    if stripped.startswith("## "):
        return stripped[3:]
    if stripped.startswith("### "):
        return stripped[4:]
    return stripped


def markdown_to_wrapped_lines(text: str, width: int = 92):
    lines = []
    in_code = False

    for raw in text.splitlines():
        line = raw.rstrip()

        if line.strip().startswith("```"):
            in_code = not in_code
            lines.append("")
            continue

        if in_code:
            lines.append("    " + line)
            continue

        if not line.strip():
            lines.append("")
            continue

        cleaned = clean_markdown_line(line)

        if cleaned.startswith("- "):
            body = cleaned[2:]
            wrapped = textwrap.wrap(
                body,
                width=width - 4,
                initial_indent="  - ",
                subsequent_indent="    ",
                break_long_words=False,
                break_on_hyphens=False,
            )
        elif len(cleaned) > 2 and cleaned[0].isdigit() and cleaned[1:3] == ". ":
            prefix = cleaned[:3]
            body = cleaned[3:]
            wrapped = textwrap.wrap(
                body,
                width=width - len(prefix) - 2,
                initial_indent=prefix,
                subsequent_indent="   ",
                break_long_words=False,
                break_on_hyphens=False,
            )
        else:
            wrapped = textwrap.wrap(
                cleaned,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )

        lines.extend(wrapped or [""])

    return lines


def write_pdf(lines, pdf_path: Path):
    page_width, page_height = 8.5, 11.0
    left = 0.72
    top = 10.25
    line_height = 0.155
    lines_per_page = 58

    with PdfPages(pdf_path) as pdf:
        page_num = 1
        for start in range(0, len(lines), lines_per_page):
            fig = plt.figure(figsize=(page_width, page_height))
            fig.patch.set_facecolor("white")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")

            y = top
            for line in lines[start : start + lines_per_page]:
                fontweight = "normal"
                fontsize = 9.2

                if line and not line.startswith((" ", "-", "1.", "2.", "3.", "4.", "5.", "6.")):
                    if line.upper() == line and len(line) < 80:
                        fontweight = "bold"
                        fontsize = 13
                    elif line[0].isdigit() and ". " in line[:5]:
                        fontweight = "bold"
                        fontsize = 10.5

                if line.startswith("    "):
                    fontsize = 8.4
                    family = "monospace"
                else:
                    family = "DejaVu Sans"

                ax.text(
                    left / page_width,
                    y / page_height,
                    line,
                    transform=fig.transFigure,
                    ha="left",
                    va="top",
                    fontsize=fontsize,
                    fontfamily=family,
                    fontweight=fontweight,
                    color="#111111",
                )
                y -= line_height

            ax.text(
                0.5,
                0.045,
                f"Case 1 Controller Report | page {page_num}",
                transform=fig.transFigure,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
            pdf.savefig(fig)
            plt.close(fig)
            page_num += 1


def main():
    text = MD_PATH.read_text(encoding="utf-8")
    lines = markdown_to_wrapped_lines(text)
    write_pdf(lines, PDF_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
