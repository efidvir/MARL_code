-- Pandoc Lua filter: turn the UNITY-6G T2.3 docx into whitepaper-ready LaTeX.
--  * strips manual heading numbers and converts ALL-CAPS headings to title case
--  * inserts \appendix before the first annex
--  * turns "ID / Req-... / Priority / ..." tables into \begin{requirement} boxes
--  * attaches "Table N-M." / "Figure N-M." paragraphs as real captions
--  * unwraps block quotes that Word list styles leave inside table cells

local stringify = pandoc.utils.stringify

local SMALL = {}
for w in ("a an and are as at by for from in is of on or than that the to with"):gmatch("%S+") do SMALL[w] = true end
local SPECIAL = { AI = "AI", JSON = "JSON", KPIS = "KPIs", ["UNITY-6G"] = "UNITY-6G",
  ["AI-NATIVE"] = "AI-Native" }

local function cap(w)
  return w:sub(1, 1):upper() .. w:sub(2):lower()
end

local function titlecase(s)
  local out, i = {}, 0
  for w in s:gmatch("%S+") do
    i = i + 1
    local r
    if SPECIAL[w] then
      r = SPECIAL[w]
    elseif i > 1 and SMALL[w:lower()] and not (out[#out] or ""):match(":$") then
      r = w:lower()
    else
      -- hyphenated words: capitalize each part (Agent-Facing, Fail-Closed)
      local parts = {}
      for p in (w .. "-"):gmatch("([^-]*)-") do
        parts[#parts + 1] = SPECIAL[p] or cap(p)
      end
      r = table.concat(parts, "-")
    end
    out[#out + 1] = r
  end
  return table.concat(out, " ")
end

local function latex(blocks)
  local s = pandoc.write(pandoc.Pandoc(blocks), "latex")
  return (s:gsub("%s+$", ""))
end

local function inline_latex(inlines)
  return latex({ pandoc.Plain(inlines) })
end

local function esc(s)
  return (s:gsub("\\", "\\textbackslash{}"):gsub("([#%%&_{}%$])", "\\%1"))
end

-- unwrap BlockQuote anywhere inside a cell
local unquote = { BlockQuote = function(b) return b.content end }

-- Word cells sometimes carry "• item" paragraphs instead of a real list;
-- turn each run of them into a BulletList.
local function bulletize(blocks)
  local out, items = {}, nil
  local function flush()
    if items then out[#out + 1] = pandoc.BulletList(items); items = nil end
  end
  for _, b in ipairs(blocks) do
    local first = (b.t == "Para" or b.t == "Plain") and b.content[1]
    if first and first.t == "Str" and first.text:match("^•") then
      local inl = pandoc.List(b.content)
      local rest = first.text:gsub("^•%s*", "")
      if rest == "" then inl:remove(1) else inl[1] = pandoc.Str(rest) end
      while #inl > 0 and (inl[1].t == "Space" or inl[1].t == "SoftBreak") do inl:remove(1) end
      items = items or {}
      items[#items + 1] = { pandoc.Plain(inl) }
    else
      flush()
      out[#out + 1] = b
    end
  end
  flush()
  return out
end

local function cell_blocks(c)
  local d = pandoc.Pandoc(c.contents):walk(unquote)
  return bulletize(d.blocks)
end

---------------------------------------------------------------- headings
local in_annex = false
function Header(h)
  local text = stringify(h.content)
  local annex = text:match("^ANNEX%s+%u%s+(.*)$")
  local stripped = annex or text:match("^[%dA-Z]+%.[%d.]*%s+(.*)$") or text:match("^%d+%s+(.*)$")
  if not stripped then return h end
  -- only reformat if the heading was ALL CAPS
  if stripped == stripped:upper() then stripped = titlecase(stripped) end
  h.content = { pandoc.Str(stripped) }
  if annex and not in_annex then
    in_annex = true
    return { pandoc.RawBlock("latex", "\\appendix"), h }
  end
  return h
end

---------------------------------------------------------------- requirements
local function is_requirement(t)
  local h = t.head.rows[1]
  if not h or #h.cells ~= 4 then return false end
  return stringify(h.cells[1].contents) == "ID" and stringify(h.cells[2].contents):match("^Req%-") ~= nil
end

local function requirement(t)
  local h = t.head.rows[1].cells
  local id = stringify(h[2].contents)
  local prio = stringify(h[4].contents)
  local name, fields = "", {}
  for _, body in ipairs(t.bodies) do
    for _, row in ipairs(body.body) do
      local label = stringify(row.cells[1].contents):gsub("%s+", " ")
      local blocks = {}
      for k = 2, #row.cells do
        for _, b in ipairs(cell_blocks(row.cells[k])) do blocks[#blocks + 1] = b end
      end
      if label == "Name" then
        name = latex(blocks)
      else
        label = label:gsub("Description/ ?Rationale", "Description / rationale")
        fields[#fields + 1] = "\\reqfield{" .. esc(label) .. "}{%\n" .. latex(blocks) .. "}"
      end
    end
  end
  local out = "\\begin{requirement}{" .. esc(id) .. "}{" .. esc(prio) .. "}{" .. name .. "}\n"
    .. table.concat(fields, "\n") .. "\n\\end{requirement}"
  return pandoc.RawBlock("latex", out)
end

---------------------------------------------------------------- captions
local function caption_of(block, kind)
  if not block or (block.t ~= "Para" and block.t ~= "Plain") then return nil end
  local s = stringify(block.content)
  local num, rest = s:match("^" .. kind .. " ([%u%d]+%-%d+)%.%s+(.*)$")
  if not num or #s > 220 then return nil end
  -- drop the leading "Table N-M." inlines, keep formatting of the rest
  local inl, seen, out = block.content, 0, {}
  local i = 1
  while i <= #inl and seen < 2 do
    if inl[i].t == "Str" then seen = seen + 1 end
    i = i + 1
  end
  while i <= #inl and (inl[i].t == "Space" or inl[i].t == "SoftBreak") do i = i + 1 end
  for j = i, #inl do out[#out + 1] = inl[j] end
  return num, out
end

local function figure_latex(img_block, num, cap)
  local img = nil
  pandoc.Pandoc({ img_block }):walk({ Image = function(im) img = im end })
  local src = img.src
  local c = inline_latex(cap)
  return pandoc.RawBlock("latex", table.concat({
    "\\begin{figure}[htbp]",
    "\\centering",
    "\\includegraphics[width=\\linewidth,height=0.45\\textheight,keepaspectratio]{" .. src .. "}",
    "\\captionnum{" .. num .. "}",
    "\\caption{" .. c .. "}",
    "\\label{fig:" .. num .. "}",
    "\\end{figure}" }, "\n"))
end

local function has_image(b)
  local found = false
  if b.t == "Para" or b.t == "Plain" then
    for _, x in ipairs(b.content) do if x.t == "Image" then found = true end end
  elseif b.t == "Figure" then found = true end
  return found
end

local function fix_table(t)
  -- unwrap quotes and normalize "•" paragraphs in every cell
  t = t:walk(unquote)
  local function fix_rows(rows)
    for _, row in ipairs(rows) do
      for _, c in ipairs(row.cells) do c.contents = bulletize(c.contents) end
    end
  end
  fix_rows(t.head.rows)
  for _, body in ipairs(t.bodies) do fix_rows(body.body) end
  return t
end

function Blocks(blocks)
  local out = {}
  local i = 1
  while i <= #blocks do
    local b = blocks[i]
    if b.t == "Table" then
      if is_requirement(b) then
        out[#out + 1] = requirement(b)
      else
        local num, cap = caption_of(blocks[i + 1], "Table")
        b = fix_table(b)
        local size = (#b.colspecs >= 7) and "\\footnotesize" or ""
        if num then
          b.caption = { long = { pandoc.Plain(cap) } }
          i = i + 1
        end
        out[#out + 1] = pandoc.RawBlock("latex", "\\begin{wptable}{" .. (num or "") .. "}" .. size)
        out[#out + 1] = b
        out[#out + 1] = pandoc.RawBlock("latex", "\\end{wptable}")
      end
    elseif has_image(b) then
      local num, cap = caption_of(blocks[i + 1], "Figure")
      if num then
        out[#out + 1] = figure_latex(b, num, cap)
        i = i + 1
      else
        out[#out + 1] = b
      end
    else
      out[#out + 1] = b
    end
    i = i + 1
  end
  return out
end

-- let long identifiers (field.paths, snake_case) break after "." and "_"
local function breakable(s)
  if s.text == "KPIMARK" then return pandoc.RawInline("latex", "\\kpimark{}") end
  if not s.text:match("_") and not (#s.text >= 14 and s.text:match("%l%.%l")) then
    return nil
  end
  local out, buf = {}, ""
  for ch in s.text:gmatch(utf8.charpattern) do
    buf = buf .. ch
    if ch == "_" or ch == "." then
      out[#out + 1] = pandoc.Str(buf); buf = ""
      out[#out + 1] = pandoc.RawInline("latex", "\\allowbreak{}")
    end
  end
  if buf ~= "" then out[#out + 1] = pandoc.Str(buf) end
  return out
end

return {
  { Header = Header },
  { Str = breakable },
  { Blocks = Blocks },
}
