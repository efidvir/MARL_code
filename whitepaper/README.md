# UNITY-6G White Paper — The AI-Native Agentic Framework (LaTeX)

LaTeX white-paper edition of `T23_D24_Complete_v3_1.docx` (Task 2.3 contribution to D2.4).

## Open in Overleaf

1. Zip this folder (or use `unity6g-whitepaper-overleaf.zip`) and in Overleaf choose
   **New Project → Upload Project**.
2. Leave the compiler on **pdfLaTeX** (default). XeLaTeX and LuaLaTeX also work.
3. Main document: `main.tex`. A full build is about 220 pages.

## Layout

| Path | Content |
| --- | --- |
| `main.tex` | Cover page, front matter, section includes |
| `unity6g-whitepaper.sty` | All styling: fonts, colours, headings, requirement cards, tables, code |
| `frontmatter/abstract.tex` | Abstract, key points, reading guide (written for the white paper) |
| `sections/*.tex` | Chapters 1–12 and Annexes A–B, converted from the docx |
| `figures/` | The four figures of Chapter 3 |
| `tools/` | Converter: `build_from_docx.py` + pandoc Lua filter `whitepaper.lua` |

## Conventions in the converted text

* Requirements are `\begin{requirement}{ID}{PRIORITY}{Name} … \end{requirement}` with one
  `\reqfield{Label}{…}` per row; the priority badge colour follows MUST / SHOULD / MAY.
* Table and figure numbers are kept exactly as in the source (`Table 3-4`, `Figure 3-1`) so
  that cross-references in the running text stay correct: `\begin{wptable}{3-4}` sets the
  number, and the caption sits inside the `longtable`.
* In the requirement-to-KPI matrices, `\kpimark` marks a link (shaded cell in the docx).

## Regenerating from a newer docx

Requires pandoc ≥ 3.1:

```
python3 tools/build_from_docx.py path/to/T23_D24.docx
```

This rewrites `sections/` and `figures/`; `main.tex`, the style and the front matter are untouched.
Hand edits to `sections/` are overwritten, so make them in the docx or re-apply them.
