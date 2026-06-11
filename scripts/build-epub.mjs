/* global process, Buffer */
import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { execFileSync } from 'node:child_process'
import { createRequire } from 'node:module'
import archiver from 'archiver'
import {
  docsDir,
  distDir,
  packageJson,
  bookVersion,
  bookAuthor,
  bookTitle,
  bookTagline,
  bookBuildDate,
  bookGithubUrl,
  bookLicense,
  logoPath as pdfLogoPath,
  escapeHtml,
  stripFrontmatter,
  stripScriptSetup,
  flattenSidebar,
  pagePathForLink,
  loadVitePressConfig,
  loadSidebar,
  frontMatter
} from './book-shared.mjs'

const renderBookAssetScript = path.join(
  path.dirname(new URL(import.meta.url).pathname),
  'render-book-asset.mjs'
)

const require = createRequire(import.meta.url)
const markdownIt = require('markdown-it')
const markdownItFootnote = require('markdown-it-footnote')
const markdownItContainer = require('markdown-it-container')
const katex = require('katex')

const logoPath = path.resolve(docsDir, '__pdf__', pdfLogoPath)
const epubFileName =
  process.env.EPUB_FILE_NAME ||
  `${packageJson.name}-open-textbook-v${bookVersion}.epub`
const epubOutputPath = path.join(distDir, epubFileName)
const epubLang = 'zh-CN'
const epubBookId = 'hands-on-modern-rl-epub'

// --- Markdown-it setup (mirrors VitePress config) ---

function isValidMathDelimiter(state, pos) {
  const max = state.posMax
  const prevChar = pos > 0 ? state.src.charCodeAt(pos - 1) : -1
  const nextChar = pos + 1 <= max ? state.src.charCodeAt(pos + 1) : -1

  return {
    canOpen: nextChar !== 0x20 && nextChar !== 0x09,
    canClose:
      prevChar !== 0x20 &&
      prevChar !== 0x09 &&
      (nextChar < 0x30 || nextChar > 0x39)
  }
}

function mathInline(state, silent) {
  if (state.src[state.pos] !== '$') return false

  let delimiter = isValidMathDelimiter(state, state.pos)
  if (!delimiter.canOpen) {
    if (!silent) state.pending += '$'
    state.pos += 1
    return true
  }

  const start = state.pos + 1
  const max = state.posMax
  let match = start
  while ((match = state.src.indexOf('$', match)) !== -1) {
    if (match >= max) {
      match = -1
      break
    }
    let pos = match - 1
    while (state.src[pos] === '\\') pos -= 1
    if ((match - pos) % 2 === 1) break
    match += 1
  }

  if (match === -1) {
    if (!silent) state.pending += '$'
    state.pos = start
    return true
  }

  if (match - start === 0) {
    if (!silent) state.pending += '$$'
    state.pos = start + 1
    return true
  }

  delimiter = isValidMathDelimiter(state, match)
  if (!delimiter.canClose) {
    if (!silent) state.pending += '$'
    state.pos = start
    return true
  }

  if (!silent) {
    const token = state.push('math_inline', 'math', 0)
    token.markup = '$'
    token.content = state.src.slice(start, match)
  }

  state.pos = match + 1
  return true
}

function mathBlock(state, start, end, silent) {
  let pos = state.bMarks[start] + state.tShift[start]
  const max = state.eMarks[start]

  if (pos + 2 > max) return false
  if (state.src.slice(pos, pos + 2) !== '$$') return false

  pos += 2
  let firstLine = state.src.slice(pos, max)
  let lastLine = ''
  let found = false
  let next = start

  if (silent) return true
  if (firstLine.trim().slice(-2) === '$$') {
    firstLine = firstLine.trim().slice(0, -2)
    found = true
  }

  while (!found) {
    next++
    if (next >= end) break

    pos = state.bMarks[next] + state.tShift[next]
    const lineMax = state.eMarks[next]
    if (pos < lineMax && state.tShift[next] < state.blkIndent) break

    if (state.src.slice(pos, lineMax).trim().slice(-2) === '$$') {
      const lastPos = state.src.slice(0, lineMax).lastIndexOf('$$')
      lastLine = state.src.slice(pos, lastPos)
      found = true
    }
  }

  state.line = next + 1
  const token = state.push('math_block', 'math', 0)
  token.block = true
  token.content =
    (firstLine && firstLine.trim() ? `${firstLine}\n` : '') +
    state.getLines(start + 1, next, state.tShift[start], true) +
    (lastLine && lastLine.trim() ? lastLine : '')
  token.map = [start, state.line]
  token.markup = '$$'
  return true
}

function renderKatex(content, displayMode) {
  const html = katex.renderToString(content, {
    displayMode,
    output: 'mathml',
    throwOnError: false,
    strict: false,
    trust: true
  })
  return html
    .replace(/^<span class="katex">/, '')
    .replace(/<\/span>$/, '')
    .replace(/<semantics>/, '')
    .replace(/<annotation encoding="application\/x-tex">[\s\S]*?<\/annotation><\/semantics>/, '')
}

let mermaidAssetDir = null
let mermaidAssetCount = 0

function getMermaidAssetDir() {
  if (!mermaidAssetDir) {
    mermaidAssetDir = fs.mkdtempSync(path.join(os.tmpdir(), 'epub-mermaid-'))
  }
  return mermaidAssetDir
}

function cleanupMermaidAssets() {
  if (mermaidAssetDir) {
    fs.rmSync(mermaidAssetDir, { recursive: true, force: true })
  }
}

// --- Image format detection & conversion ---

let convertedAssetDir = null

function getConvertedAssetDir() {
  if (!convertedAssetDir) {
    convertedAssetDir = fs.mkdtempSync(path.join(os.tmpdir(), 'epub-converted-'))
  }
  return convertedAssetDir
}

function cleanupConvertedAssets() {
  if (convertedAssetDir) {
    fs.rmSync(convertedAssetDir, { recursive: true, force: true })
  }
}

function detectActualFormat(filePath) {
  try {
    const fd = fs.openSync(filePath, 'r')
    const buf = Buffer.alloc(12)
    fs.readSync(fd, buf, 0, 12, 0)
    fs.closeSync(fd)

    if (buf[0] === 0x52 && buf[1] === 0x49 && buf[2] === 0x46 && buf[3] === 0x46 &&
        buf[8] === 0x57 && buf[9] === 0x45 && buf[10] === 0x42 && buf[11] === 0x50) {
      return 'webp'
    }
    if (buf[0] === 0x47 && buf[1] === 0x49 && buf[2] === 0x46) {
      return 'gif'
    }
  } catch {
    // unreadable
  }
  return null
}

const convertCache = new Map()

function convertImageToPng(imagePath) {
  const cached = convertCache.get(imagePath)
  if (cached) return cached

  const dir = getConvertedAssetDir()
  const basename = path.basename(imagePath, path.extname(imagePath))
  const pngPath = path.join(dir, `${basename}-${Date.now()}.png`)

  try {
    execFileSync('sips', ['-s', 'format', 'png', imagePath, '--out', pngPath], {
      stdio: ['ignore', 'pipe', 'pipe'],
      timeout: 15_000
    })
    if (fs.existsSync(pngPath) && fs.statSync(pngPath).size > 0) {
      convertCache.set(imagePath, pngPath)
      return pngPath
    }
  } catch {
    // sips not available or conversion failed
  }
  return null
}

function needsConversion(filePath, ext) {
  if (ext === '.webp') return true
  if (ext === '.gif') return true
  const actual = detectActualFormat(filePath)
  if (actual === 'webp') return true
  return false
}

// --- Cover image rendering ---

function findBrowserExecutable() {
  const candidates = [
    process.env.PUPPETEER_EXECUTABLE_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser'
  ].filter(Boolean)
  return candidates.find((c) => fs.existsSync(c)) || null
}

async function renderCoverToPng() {
  const browser = findBrowserExecutable()
  if (!browser) {
    console.warn('Warning: No browser found for cover rendering, using logo fallback')
    return null
  }

  let puppeteer
  try {
    const require = createRequire(import.meta.url)
    puppeteer = require('puppeteer-core')
  } catch {
    console.warn('Warning: puppeteer-core not available, using logo fallback')
    return null
  }

  const { cover } = frontMatter
  const coverHtml = `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  width: 1600px; height: 2560px;
  font-family: "Noto Serif CJK SC", "Source Han Serif SC", "Songti SC", Georgia, serif;
  background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
  display: flex; flex-direction: column; justify-content: space-between;
  padding: 160px 120px 120px;
}
.top { flex: 1; display: flex; flex-direction: column; justify-content: center; }
.logo { max-width: 900px; max-height: 280px; margin-bottom: 80px; object-fit: contain; }
.kicker { color: #0f4c81; font-size: 32px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 24px; }
h1 { color: #102033; font-size: 120px; line-height: 1.08; margin-bottom: 24px; }
.title-zh { color: #344054; font-size: 64px; font-weight: 700; margin-bottom: 48px; }
.description { color: #344054; font-size: 36px; line-height: 1.7; max-width: 1200px; padding-top: 32px; border-top: 4px solid #f59e0b; }
.bottom { padding-top: 60px; border-top: 2px solid #cbd5e1; }
.meta { display: grid; grid-template-columns: 1fr 1fr; gap: 24px 60px; margin-bottom: 40px; }
.meta p { font-size: 28px; color: #212121; }
.meta span { display: block; color: #667085; font-size: 22px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px; }
.publisher { color: #667085; font-size: 28px; letter-spacing: 0.04em; text-transform: uppercase; }
</style>
</head>
<body>
<div class="top">
  <img class="logo" src="${logoPath}" />
  <p class="kicker">${escapeHtml(cover.kicker)}</p>
  <h1>${escapeHtml(cover.titleEn)}</h1>
  <p class="title-zh">${escapeHtml(cover.titleZh)}</p>
  <p class="description">${escapeHtml(cover.description)}</p>
</div>
<div class="bottom">
  <div class="meta">
    <p><span>Version</span>${escapeHtml(bookVersion)}</p>
    <p><span>Authors</span>${escapeHtml(bookAuthor)}</p>
    <p><span>Repository</span>${escapeHtml(bookGithubUrl)}</p>
    <p><span>Build</span>${escapeHtml(bookBuildDate)}</p>
  </div>
  <p class="publisher">${escapeHtml(cover.publisher)}</p>
</div>
</body>
</html>`

  const dir = getConvertedAssetDir()
  const htmlPath = path.join(dir, 'cover.html')
  const pngPath = path.join(dir, 'cover-full.png')
  fs.writeFileSync(htmlPath, coverHtml)

  try {
    const instance = await puppeteer.launch({
      executablePath: browser,
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    })
    const page = await instance.newPage()
    await page.setViewport({ width: 1600, height: 2560, deviceScaleFactor: 1 })
    await page.goto(`file://${htmlPath}`, { waitUntil: 'networkidle0', timeout: 30000 })
    await page.screenshot({ path: pngPath, type: 'png', fullPage: false })
    await instance.close()

    if (fs.existsSync(pngPath) && fs.statSync(pngPath).size > 0) {
      console.log('Generated cover image (1600x2560)')
      return pngPath
    }
  } catch (err) {
    console.warn(`Warning: Cover rendering failed: ${err.message}`)
  }

  return null
}

function renderMermaidToPng(source) {
  if (!fs.existsSync(renderBookAssetScript)) return null

  mermaidAssetCount += 1
  const stem = `mermaid-${String(mermaidAssetCount).padStart(4, '0')}`
  const dir = getMermaidAssetDir()
  const mmdPath = path.join(dir, `${stem}.mmd`)
  const pngPath = path.join(dir, `${stem}.png`)

  fs.writeFileSync(mmdPath, source)

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    fs.rmSync(pngPath, { force: true })
    try {
      execFileSync(process.execPath, [renderBookAssetScript, 'mermaid', mmdPath, pngPath], {
        cwd: path.resolve(path.dirname(renderBookAssetScript), '..'),
        stdio: ['ignore', 'pipe', 'pipe'],
        timeout: 30_000
      })
    } catch {
      continue
    }
    if (fs.existsSync(pngPath) && fs.statSync(pngPath).size > 0) {
      return pngPath
    }
  }

  return null
}

// --- Heading ID generation (matches VitePress slugify) ---

const rControl = /[\u0000-\u001f]/g // eslint-disable-line no-control-regex
const rSpecial = /[\s~`!@#$%^&*()\-_+=[\]{}|\\;:"'"“”‘’<>,.?/]+/g
const rCombining = /[̀-ͯ]/g

function slugify(str) {
  return str
    .normalize('NFKD')
    .replace(rCombining, '')
    .replace(rControl, '')
    .replace(rSpecial, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/^(\d)/, '_$1')
    .toLowerCase()
}

function headingIdPlugin(md) {
  const slugCounts = new Map()

  md.core.ruler.push('heading_ids', (state) => {
    slugCounts.clear()
    for (let idx = 0; idx < state.tokens.length - 1; idx++) {
      const token = state.tokens[idx]
      if (token.type !== 'heading_open') continue

      const inline = state.tokens[idx + 1]
      if (!inline || inline.type !== 'inline') continue

      // Check for {#custom-id} syntax
      let customId = null
      const children = inline.children || []
      const lastText = [...children].reverse().find((t) => t.type === 'text')
      if (lastText) {
        const match = lastText.content.match(
          /\s*\{#([A-Za-z0-9][A-Za-z0-9_.:-]*)\}$/
        )
        if (match) {
          customId = match[1]
          lastText.content = lastText.content.slice(0, match.index)
          inline.content = inline.content.replace(match[0], '')
        }
      }

      const title = inline.content.replace(/`([^`]+)`/g, '$1').replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').trim()
      const baseSlug = customId || slugify(title)
      const count = slugCounts.get(baseSlug) || 0
      slugCounts.set(baseSlug, count + 1)
      const slug = count === 0 ? baseSlug : `${baseSlug}-${count}`

      token.attrSet('id', slug)
    }
  })
}

function createMarkdownRenderer(imageMap) {
  const md = markdownIt({ html: true, xhtmlOut: true, linkify: true, typographer: false })

  md.inline.ruler.before('text', 'math_inline', mathInline)
  md.block.ruler.after('blockquote', 'math_block', mathBlock, {
    alt: ['paragraph', 'reference', 'blockquote', 'list']
  })
  md.renderer.rules.math_inline = (tokens, idx) =>
    renderKatex(tokens[idx].content, false)
  md.renderer.rules.math_block = (tokens, idx) =>
    `<p class="math-block">${renderKatex(tokens[idx].content, true)}</p>\n`

  md.use(markdownItFootnote)
  headingIdPlugin(md)

  md.use(markdownItContainer, 'output', {
    render(tokens, idx) {
      if (tokens[idx].nesting === 1) {
        const title = tokens[idx].info.trim().slice(6).trim() || '运行结果'
        return `<div class="output-block"><p class="output-title">${escapeXml(title)}</p>\n`
      }
      return '</div>\n'
    }
  })

  const defaultFence =
    md.renderer.rules.fence ||
    function (tokens, idx, options, env, self) {
      return self.renderToken(tokens, idx, options)
    }

  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    if (token.info.trim().startsWith('mermaid')) {
      const pngPath = renderMermaidToPng(token.content)
      if (pngPath) {
        const imageId = `mermaid-${String(imageMap.size).padStart(4, '0')}.png`
        imageMap.set(pngPath, imageId)
        return `<div class="mermaid-diagram"><img src="../images/${imageId}" alt="Mermaid diagram" /></div>\n`
      }
      const escaped = escapeXml(token.content)
      return `<div class="mermaid-placeholder"><p class="mermaid-label">[图表]</p><pre>${escaped}</pre></div>\n`
    }
    return defaultFence(tokens, idx, options, env, self)
  }

  return md
}

// --- Utility functions ---

function htmlToXhtml(html) {
  const voidElements = 'area|base|br|col|embed|hr|img|input|link|meta|param|source|track|wbr'
  const pattern = new RegExp(`<(${voidElements})(\\b[^>]*?)\\s*/?>`, 'gi')
  return html
    .replace(pattern, (match, tag, attrs) => {
      const cleanAttrs = attrs.replace(/\s*\/$/, '')
      return `<${tag}${cleanAttrs} />`
    })
    .replace(/&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)/g, '&amp;')
}

function escapeXml(value) {
  return escapeHtml(value).replace(/'/g, '&apos;')
}

// --- Internal link rewriting ---

function normalizeLinkKey(link) {
  return link
    .replace(/^\//, '')
    .replace(/#.*$/, '')
    .replace(/\?.*$/, '')
    .replace(/\/$/, '')
    .replace(/\/index$/, '')
    .replace(/\.md$/, '')
}

function buildLinkMap(allPages) {
  const map = new Map()
  for (const page of allPages) {
    if (!page._chapterId || !page.link) continue
    const key = normalizeLinkKey(page.link)
    map.set(key, page._chapterId)
  }
  return map
}

function rewriteInternalLinks(html, linkMap, sourcePageLink) {
  const sourceDir = sourcePageLink
    ? sourcePageLink.replace(/^\//, '').replace(/\/[^/]*$/, '')
    : ''

  return html.replace(/<a\s([^>]*?)href=["']([^"']+)["']([^>]*)>/gi, (match, pre, href, post) => {
    if (href.startsWith('http://') || href.startsWith('https://') || href.startsWith('mailto:') || href.startsWith('#')) {
      return match
    }

    const hashIdx = href.indexOf('#')
    const fragment = hashIdx >= 0 ? href.slice(hashIdx) : ''
    const pathPart = hashIdx >= 0 ? href.slice(0, hashIdx) : href

    let key
    if (pathPart.startsWith('/')) {
      key = normalizeLinkKey(pathPart)
    } else if (pathPart.startsWith('./') || pathPart.startsWith('../') || !pathPart.includes('://')) {
      const resolved = path.posix.join(sourceDir, pathPart)
      key = normalizeLinkKey(resolved)
    } else {
      key = normalizeLinkKey(pathPart)
    }

    const chapterId = linkMap.get(key)

    if (chapterId) {
      return `<a ${pre}href="${chapterId}.xhtml${fragment}"${post}>`
    }

    const docsPath = key.startsWith('code/') ? key : `docs/${key}`
    const fallbackUrl = `${bookGithubUrl}/blob/main/${docsPath}`
    return `<a ${pre}href="${escapeXml(fallbackUrl)}"${post}>`
  })
}

function mediaTypeForExt(ext) {
  const types = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.webp': 'image/webp'
  }
  return types[ext.toLowerCase()] || 'application/octet-stream'
}

// --- Sidebar & page loading ---

function collectTocSections(sidebar) {
  const sections = []
  for (const section of sidebar || []) {
    const entry = { title: section.text || '未命名', items: [] }
    flattenSidebar(section.items || [], entry.items)
    if (section.link) {
      entry.items.unshift({ title: section.text, link: section.link })
    }
    sections.push(entry)
  }
  return sections
}

// --- Image collection ---

function collectAndRewriteImages(html, sourcePagePath, imageMap) {
  const imgPattern = /<img\s[^>]*src=["']([^"']+)["'][^>]*>/gi
  return html.replace(imgPattern, (match, src) => {
    if (src.startsWith('data:')) return match

    if (src.startsWith('http://') || src.startsWith('https://')) {
      const altMatch = match.match(/alt=["']([^"']*)["']/)
      const alt = altMatch ? altMatch[1] : '外部图片'
      return `<span class="external-image-placeholder">[在线图片: ${escapeXml(alt)}]</span>`
    }

    const sourceDir = path.dirname(sourcePagePath)
    let absolutePath
    if (src.startsWith('/')) {
      absolutePath = path.join(docsDir, src)
    } else {
      absolutePath = path.resolve(sourceDir, decodeURIComponent(src))
    }

    if (!fs.existsSync(absolutePath)) return match

    let ext = path.extname(absolutePath).toLowerCase()

    if (needsConversion(absolutePath, ext)) {
      const converted = convertImageToPng(absolutePath)
      if (converted) {
        absolutePath = converted
        ext = '.png'
      } else if (ext === '.webp') {
        return match
      }
    }

    const imageId =
      imageMap.get(absolutePath) ||
      `img-${String(imageMap.size).padStart(4, '0')}${ext}`

    if (!imageMap.has(absolutePath)) {
      imageMap.set(absolutePath, imageId)
    }

    return match.replace(src, `../images/${imageId}`)
  })
}

function collectMarkdownImages(markdown, sourcePagePath, imageMap) {
  const imgPattern = /!\[([^\]]*)\]\(([^)\s]+)(?:\s+["'][^"']*["'])?\)/g
  const sourceDir = path.dirname(sourcePagePath)

  for (const match of markdown.matchAll(imgPattern)) {
    const src = match[2]
    if (src.startsWith('data:') || src.startsWith('http://') || src.startsWith('https://')) {
      continue
    }

    let absolutePath
    if (src.startsWith('/')) {
      absolutePath = path.join(docsDir, src)
    } else {
      absolutePath = path.resolve(sourceDir, decodeURIComponent(src))
    }

    if (!fs.existsSync(absolutePath)) continue

    let ext = path.extname(absolutePath).toLowerCase()

    if (needsConversion(absolutePath, ext)) {
      const converted = convertImageToPng(absolutePath)
      if (converted) {
        absolutePath = converted
        ext = '.png'
      } else if (ext === '.webp') {
        continue
      }
    }

    if (!imageMap.has(absolutePath)) {
      const imageId = `img-${String(imageMap.size).padStart(4, '0')}${ext}`
      imageMap.set(absolutePath, imageId)
    }
  }
}

// --- ePub structure generation ---

function generateMimetype() {
  return 'application/epub+zip'
}

function generateContainerXml() {
  return `<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>`
}

function generateContentOpf(chapters, imageMap, pageCount, hasCoverImage) {
  const now = new Date().toISOString().replace(/\.\d+Z$/, 'Z')
  const manifestItems = [
    '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
    '    <item id="style" href="style.css" media-type="text/css"/>',
    '    <item id="cover" href="chapters/cover.xhtml" media-type="application/xhtml+xml"/>',
    '    <item id="intro" href="chapters/intro.xhtml" media-type="application/xhtml+xml"/>',
    '    <item id="pub-note" href="chapters/publication-note.xhtml" media-type="application/xhtml+xml"/>'
  ]

  if (hasCoverImage) {
    manifestItems.push(
      '    <item id="cover-image" href="images/cover-image.png" media-type="image/png" properties="cover-image"/>',
      '    <item id="cover-logo" href="images/cover-logo.png" media-type="image/png"/>'
    )
  } else {
    manifestItems.push(
      '    <item id="cover-logo" href="images/cover-logo.png" media-type="image/png" properties="cover-image"/>'
    )
  }
  const spineItems = [
    '    <itemref idref="cover"/>',
    '    <itemref idref="intro"/>',
    '    <itemref idref="pub-note"/>'
  ]

  for (let i = 0; i < chapters.length; i++) {
    const id = `ch${String(i).padStart(3, '0')}`
    manifestItems.push(
      `    <item id="${id}" href="chapters/${id}.xhtml" media-type="application/xhtml+xml" properties="mathml"/>`
    )
    spineItems.push(`    <itemref idref="${id}"/>`)
  }

  for (const [, imageId] of imageMap) {
    const ext = path.extname(imageId).toLowerCase()
    const mediaType = mediaTypeForExt(ext)
    const safeId = imageId.replace(/[^a-zA-Z0-9-]/g, '-')
    manifestItems.push(
      `    <item id="${safeId}" href="images/${imageId}" media-type="${mediaType}"/>`
    )
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="book-id">${escapeXml(epubBookId)}</dc:identifier>
    <dc:title>${escapeXml(bookTitle)}</dc:title>
    <dc:creator>${escapeXml(bookAuthor)}</dc:creator>
    <dc:language>${epubLang}</dc:language>
    <dc:date>${now}</dc:date>
    <meta property="dcterms:modified">${now}</meta>
    <dc:description>${escapeXml(bookTagline)}</dc:description>
    <dc:rights>${escapeXml(bookLicense)}</dc:rights>
    <dc:publisher>WalkingLabs Open Course Series</dc:publisher>
    <meta name="version" content="${escapeXml(bookVersion)}"/>
    <meta name="build-date" content="${escapeXml(bookBuildDate)}"/>
    <meta name="page-count" content="${pageCount}"/>
  </metadata>
  <manifest>
${manifestItems.join('\n')}
  </manifest>
  <spine>
${spineItems.join('\n')}
  </spine>
</package>`
}

function generateNavXhtml(tocSections) {
  const navItems = [
    '      <li><span>前置页</span>',
    '        <ol>',
    '          <li><a href="chapters/cover.xhtml">封面</a></li>',
    '          <li><a href="chapters/intro.xhtml">本书简介</a></li>',
    '          <li><a href="chapters/publication-note.xhtml">版本说明</a></li>',
    '        </ol>',
    '      </li>'
  ]
  for (const section of tocSections) {
    navItems.push(`      <li><span>${escapeXml(section.title)}</span>`)
    if (section.items.length) {
      navItems.push('        <ol>')
      for (const item of section.items) {
        const id = item._chapterId
        if (id) {
          navItems.push(
            `          <li><a href="chapters/${id}.xhtml">${escapeXml(item.title)}</a></li>`
          )
        }
      }
      navItems.push('        </ol>')
    }
    navItems.push('      </li>')
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="${epubLang}" xml:lang="${epubLang}">
<head>
  <meta charset="UTF-8"/>
  <title>目录</title>
  <link rel="stylesheet" href="style.css"/>
</head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>目录</h1>
    <ol>
${navItems.join('\n')}
    </ol>
  </nav>
</body>
</html>`
}

function generateStyleCss() {
  return `@charset "UTF-8";

body {
  font-family: "Noto Serif CJK SC", "Source Han Serif SC", "Songti SC", Georgia, serif;
  font-size: 1em;
  line-height: 1.7;
  color: #1a1a1a;
  margin: 1em;
}

h1 { font-size: 1.8em; margin: 1.2em 0 0.6em; line-height: 1.2; }
h2 { font-size: 1.4em; margin: 1em 0 0.5em; line-height: 1.3; }
h3 { font-size: 1.2em; margin: 0.8em 0 0.4em; }
h4 { font-size: 1.05em; margin: 0.6em 0 0.3em; }

p { margin: 0.6em 0; }

pre {
  background: #f5f5f5;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  padding: 0.8em;
  font-size: 0.85em;
  line-height: 1.4;
  overflow-x: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
}

code {
  font-family: "JetBrains Mono", "Fira Code", "Source Code Pro", monospace;
  font-size: 0.88em;
  background: #f0f0f0;
  padding: 0.1em 0.3em;
  border-radius: 3px;
}

pre code {
  background: none;
  padding: 0;
  border-radius: 0;
}

blockquote {
  margin: 0.8em 0;
  padding: 0.5em 1em;
  border-left: 3px solid #ccc;
  color: #555;
}

img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.8em auto;
}

table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.8em 0;
  font-size: 0.9em;
}

th, td {
  border: 1px solid #ddd;
  padding: 0.4em 0.6em;
  text-align: left;
}

th {
  background: #f5f5f5;
  font-weight: bold;
}

a { color: #1a237e; text-decoration: none; }

.output-block {
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 4px;
  padding: 0.6em 0.8em;
  margin: 0.8em 0;
}

.output-title {
  font-weight: bold;
  font-size: 0.9em;
  color: #495057;
  margin: 0 0 0.4em;
}

.mermaid-diagram {
  text-align: center;
  margin: 0.8em 0;
}

.mermaid-diagram img {
  max-width: 100%;
  height: auto;
}

.mermaid-placeholder {
  background: #f0f4ff;
  border: 1px dashed #7986cb;
  border-radius: 4px;
  padding: 0.6em 0.8em;
  margin: 0.8em 0;
}

.mermaid-label {
  font-style: italic;
  color: #5c6bc0;
  margin: 0 0 0.4em;
}

.mermaid-placeholder pre {
  font-size: 0.75em;
  color: #666;
  background: transparent;
  border: none;
  padding: 0;
}

.math-block {
  text-align: center;
  margin: 1em 0;
  overflow-x: auto;
}

section.footnotes {
  margin-top: 2em;
  padding-top: 1em;
  border-top: 1px solid #ddd;
  font-size: 0.85em;
}

sup { font-size: 0.75em; }

.external-image-placeholder {
  display: block;
  background: #f0f4ff;
  border: 1px dashed #7986cb;
  border-radius: 4px;
  padding: 0.4em 0.8em;
  color: #5c6bc0;
  font-style: italic;
  text-align: center;
  margin: 0.8em 0;
}

/* Cover page */
.cover-page {
  text-align: center;
  padding: 2em 1em;
}

.cover-top {
  margin-bottom: 3em;
}

.cover-logo {
  max-width: 80%;
  max-height: 15em;
  margin: 0 auto 2em;
}

.cover-kicker {
  color: #0f4c81;
  font-size: 0.85em;
  font-weight: bold;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin: 0 0 0.5em;
}

.cover-page h1 {
  font-size: 2.4em;
  line-height: 1.1;
  margin: 0;
  color: #102033;
}

.cover-title-zh {
  font-size: 1.4em;
  font-weight: bold;
  color: #344054;
  margin: 0.5em 0 0;
}

.cover-description {
  max-width: 90%;
  margin: 1.2em auto 0;
  padding-top: 0.8em;
  border-top: 2px solid #f59e0b;
  color: #344054;
  font-size: 0.95em;
  line-height: 1.7;
  text-align: left;
}

.cover-bottom {
  margin-top: 2em;
}

.cover-meta {
  width: 80%;
  margin: 0 auto;
  border: none;
  font-size: 0.9em;
}

.cover-meta td {
  border: none;
  padding: 0.3em 0.5em;
  vertical-align: top;
}

.meta-label {
  color: #667085;
  font-size: 0.8em;
  font-weight: bold;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  text-align: right;
  width: 30%;
}

.meta-value {
  color: #212121;
  text-align: left;
}

.cover-publisher {
  margin-top: 1.5em;
  color: #667085;
  font-size: 0.85em;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
`
}

function generateCoverXhtml() {
  const { cover } = frontMatter
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="${epubLang}" xml:lang="${epubLang}">
<head>
  <meta charset="UTF-8"/>
  <title>封面</title>
  <link rel="stylesheet" href="../style.css"/>
</head>
<body class="cover-page">
  <div class="cover-top">
    <img src="../images/cover-logo.png" alt="Hands-On Modern RL Logo" class="cover-logo" />
    <p class="cover-kicker">${escapeXml(cover.kicker)}</p>
    <h1>${escapeXml(cover.titleEn)}</h1>
    <p class="cover-title-zh">${escapeXml(cover.titleZh)}</p>
    <p class="cover-description">${escapeXml(cover.description)}</p>
  </div>
  <div class="cover-bottom">
    <table class="cover-meta">
      <tr><td class="meta-label">Version</td><td class="meta-value">${escapeXml(bookVersion)}</td></tr>
      <tr><td class="meta-label">Authors</td><td class="meta-value">${escapeXml(bookAuthor)}</td></tr>
      <tr><td class="meta-label">Repository</td><td class="meta-value">${escapeXml(bookGithubUrl)}</td></tr>
      <tr><td class="meta-label">Build</td><td class="meta-value">${escapeXml(bookBuildDate)}</td></tr>
    </table>
    <p class="cover-publisher">${escapeXml(cover.publisher)}</p>
  </div>
</body>
</html>`
}

function generateIntroXhtml(pageCount) {
  const { intro } = frontMatter
  const paragraphs = intro.paragraphs.map((p) => `  <p>${p}</p>`).join('\n')
  const audienceItems = intro.audience.items
    .map((item) => `    <li>${escapeXml(item)}</li>`)
    .join('\n')
  const howToRead = intro.howToRead.paragraphs
    .map((p) => `  <p>${escapeXml(p)}</p>`)
    .join('\n')

  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="${epubLang}" xml:lang="${epubLang}">
<head>
  <meta charset="UTF-8"/>
  <title>${escapeXml(intro.title)}</title>
  <link rel="stylesheet" href="../style.css"/>
</head>
<body>
  <h1>${escapeXml(intro.title)}</h1>
${paragraphs}
  <h2>${escapeXml(intro.audience.title)}</h2>
  <ul>
${audienceItems}
  </ul>
  <h2>${escapeXml(intro.howToRead.title)}</h2>
${howToRead}
  <p>本书共汇编 ${pageCount} 个课程页面，目标是让它更像一本可持续更新的课本。</p>
</body>
</html>`
}

function generatePublicationNoteXhtml() {
  const { publicationNote } = frontMatter
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="${epubLang}" xml:lang="${epubLang}">
<head>
  <meta charset="UTF-8"/>
  <title>${escapeXml(publicationNote.title)}</title>
  <link rel="stylesheet" href="../style.css"/>
</head>
<body>
  <h1>${escapeXml(publicationNote.title)}</h1>
  <p><strong>版本：</strong>${escapeXml(bookVersion)}，构建日期：${escapeXml(bookBuildDate)}。</p>
  <p><strong>作者：</strong>${escapeXml(bookAuthor)}。</p>
  <p><strong>项目地址：</strong><a href="${escapeXml(bookGithubUrl)}">${escapeXml(bookGithubUrl)}</a>。</p>
  <p>${escapeXml(publicationNote.iterationNote)}</p>
  <p>${escapeXml(publicationNote.disclaimer)}</p>
  <h2>${escapeXml(publicationNote.licenseSection.title)}</h2>
  <p>${escapeXml(publicationNote.licenseSection.text)}</p>
</body>
</html>`
}

function wrapChapterXhtml(title, bodyHtml) {
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="${epubLang}" xml:lang="${epubLang}">
<head>
  <meta charset="UTF-8"/>
  <title>${escapeXml(title)}</title>
  <link rel="stylesheet" href="../style.css"/>
</head>
<body>
${bodyHtml}
</body>
</html>`
}

// --- Main build ---

async function main() {
  const config = await loadVitePressConfig()
  const sidebar = loadSidebar(config)

  const tocSections = collectTocSections(sidebar)
  const allPages = []
  for (const section of tocSections) {
    for (const item of section.items) {
      allPages.push(item)
    }
  }

  console.log(`Found ${allPages.length} pages in sidebar`)

  const imageMap = new Map()
  const md = createMarkdownRenderer(imageMap)
  const chapters = []
  const missingPages = []

  for (let i = 0; i < allPages.length; i++) {
    const page = allPages[i]
    const sourcePagePath = pagePathForLink(page.link)

    if (!sourcePagePath) {
      missingPages.push(page.link)
      continue
    }

    const chapterId = `ch${String(chapters.length).padStart(3, '0')}`
    page._chapterId = chapterId

    let source = fs.readFileSync(sourcePagePath, 'utf8')
    source = stripScriptSetup(stripFrontmatter(source)).trim()

    collectMarkdownImages(source, sourcePagePath, imageMap)

    let html = md.render(source)
    html = collectAndRewriteImages(html, sourcePagePath, imageMap)
    html = htmlToXhtml(html)

    chapters.push({
      id: chapterId,
      title: page.title,
      html
    })
  }

  if (missingPages.length) {
    console.warn(
      `Warning: ${missingPages.length} pages not found: ${missingPages.slice(0, 5).join(', ')}${missingPages.length > 5 ? '...' : ''}`
    )
  }

  // Rewrite internal links in all chapters
  const linkMap = buildLinkMap(allPages)
  for (let i = 0; i < chapters.length; i++) {
    const page = allPages.find((p) => p._chapterId === chapters[i].id)
    chapters[i].html = rewriteInternalLinks(chapters[i].html, linkMap, page?.link || '')
  }

  console.log(
    `Rendered ${chapters.length} chapters, collected ${imageMap.size} images, mapped ${linkMap.size} internal links`
  )

  // Generate cover image
  const coverImagePath = await renderCoverToPng()

  // Build ePub zip
  fs.mkdirSync(distDir, { recursive: true })
  const output = fs.createWriteStream(epubOutputPath)
  const archive = archiver('zip', { zlib: { level: 9 } })

  const archiveFinished = new Promise((resolve, reject) => {
    output.on('close', resolve)
    archive.on('error', reject)
  })

  archive.pipe(output)

  // mimetype must be first entry, uncompressed
  archive.append(generateMimetype(), {
    name: 'mimetype',
    store: true
  })

  archive.append(generateContainerXml(), {
    name: 'META-INF/container.xml'
  })

  const pageCount = chapters.length

  archive.append(generateContentOpf(chapters, imageMap, pageCount, !!coverImagePath), {
    name: 'OEBPS/content.opf'
  })

  archive.append(generateNavXhtml(tocSections), {
    name: 'OEBPS/nav.xhtml'
  })

  archive.append(generateStyleCss(), {
    name: 'OEBPS/style.css'
  })

  // Front matter pages
  archive.append(generateCoverXhtml(), {
    name: 'OEBPS/chapters/cover.xhtml'
  })

  archive.append(generateIntroXhtml(pageCount), {
    name: 'OEBPS/chapters/intro.xhtml'
  })

  archive.append(generatePublicationNoteXhtml(), {
    name: 'OEBPS/chapters/publication-note.xhtml'
  })

  // Cover image
  if (coverImagePath) {
    archive.file(coverImagePath, { name: 'OEBPS/images/cover-image.png' })
  }
  archive.file(logoPath, { name: 'OEBPS/images/cover-logo.png' })

  for (const chapter of chapters) {
    const xhtml = wrapChapterXhtml(chapter.title, chapter.html)
    archive.append(xhtml, {
      name: `OEBPS/chapters/${chapter.id}.xhtml`
    })
  }

  let skippedImages = 0
  for (const [absolutePath, imageId] of imageMap) {
    if (!fs.existsSync(absolutePath)) {
      skippedImages++
      continue
    }
    const stat = fs.statSync(absolutePath)
    if (stat.size < 100) {
      skippedImages++
      continue
    }
    archive.file(absolutePath, { name: `OEBPS/images/${imageId}` })
  }

  if (skippedImages > 0) {
    console.warn(`Warning: Skipped ${skippedImages} invalid/empty images`)
  }

  await archive.finalize()
  await archiveFinished

  cleanupMermaidAssets()
  cleanupConvertedAssets()

  const sizeMb = fs.statSync(epubOutputPath).size / 1024 / 1024
  console.log(`Wrote ${epubOutputPath} (${sizeMb.toFixed(1)} MB)`)
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
