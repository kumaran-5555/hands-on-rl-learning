/* global process */
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'
import {
  rootDir,
  docsDir,
  distDir,
  packageJson,
  bookVersion,
  bookAuthor,
  bookBuildDate,
  bookGithubUrl,
  logoPath as pdfLogoPath,
  escapeHtml,
  stripFrontmatter,
  stripScriptSetup,
  splitSuffix,
  pagePathForLink,
  loadVitePressConfig,
  loadSidebar,
  frontMatter
} from './book-shared.mjs'

const generatedDir = path.join(docsDir, '__pdf__')
const generatedPagePath = path.join(generatedDir, 'index.md')
const publicSitemapPath = path.join(docsDir, 'public', 'sitemap.xml')
const pdfFileName = process.env.PDF_FILE_NAME || `${packageJson.name}.pdf`
const pdfOutputPath = path.join(distDir, pdfFileName)
const pdfPrintTimeoutMs = Number(process.env.PDF_PRINT_TIMEOUT_MS || 1200000)
const resourceWaitTimeoutMs = Number(
  process.env.PDF_RESOURCE_TIMEOUT_MS || 60000
)
const pdfVersion = bookVersion
const pdfGenerateOutline = process.env.PDF_OUTLINE !== '0'
const pdfAuthor = bookAuthor
const pdfBuildDate = bookBuildDate
const skipBuild = process.argv.includes('--skip-build')
const keepSource = process.argv.includes('--keep-source')

function toPosix(value) {
  return value.split(path.sep).join('/')
}

function escapeAttribute(value) {
  return escapeHtml(value).replace(/'/g, '&#39;')
}

const pdfGithubUrl = bookGithubUrl
const pdfHeaderFooterStyle = [
  'width:100%',
  'box-sizing:border-box',
  'padding:0 14mm',
  'font-family:Arial, sans-serif',
  'font-size:6px',
  'line-height:1.35',
  'color:#667085',
  'text-align:center',
  'white-space:nowrap',
  'overflow:hidden'
].join(';')
const pdfFooterStyle = [
  'width:100%',
  'box-sizing:border-box',
  'padding:0 14mm',
  'font-family:Arial, sans-serif',
  'font-size:6px',
  'line-height:1.35',
  'color:#667085',
  'display:grid',
  'grid-template-columns:minmax(0,1fr) minmax(0,3fr) minmax(54px,1fr)',
  'column-gap:4mm',
  'align-items:center'
].join(';')
const pdfFooterMetaStyle = [
  'min-width:0',
  'text-align:center',
  'white-space:nowrap',
  'overflow:hidden',
  'text-overflow:ellipsis'
].join(';')
const pdfFooterPageStyle = ['text-align:right', 'white-space:nowrap'].join(';')

function pdfHeaderTemplate() {
  const text = `Authors: ${escapeHtml(pdfAuthor)} | Repo: ${escapeHtml(pdfGithubUrl)}`

  return [`<div style="${pdfHeaderFooterStyle}">`, text, '</div>'].join('')
}

function pdfFooterTemplate() {
  const text = `Authors: ${escapeHtml(pdfAuthor)} | Repo: ${escapeHtml(pdfGithubUrl)}`

  return [
    `<div style="${pdfFooterStyle}">`,
    '<span></span>',
    `<span style="${pdfFooterMetaStyle}">${text}</span>`,
    `<span style="${pdfFooterPageStyle}">Page <span class="pageNumber"></span> / <span class="totalPages"></span></span>`,
    '</div>'
  ].join('')
}

function localizeFootnotes(markdown) {
  const lines = markdown.split('\n')
  const output = []
  const footnotes = []

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    const match = line.match(/^\[\^([^\]]+)\]:\s*(.*)$/)

    if (!match) {
      output.push(line)
      continue
    }

    const note = {
      id: match[1],
      content: [match[2]]
    }

    while (index + 1 < lines.length) {
      const nextLine = lines[index + 1]

      if (!/^(?: {2,}|\t)/.test(nextLine)) break
      note.content.push(nextLine.trim())
      index += 1
    }

    footnotes.push(note)
  }

  if (!footnotes.length) {
    return output.join('\n')
  }

  const noteNumbers = new Map(
    footnotes.map((note, index) => [note.id, String(index + 1)])
  )
  const withRefs = output.map((line) =>
    line.replace(/\[\^([^\]]+)\]/g, (match, id) => {
      const label = noteNumbers.get(id) || id
      return `<sup class="pdf-footnote-ref">[${escapeHtml(label)}]</sup>`
    })
  )
  const referenceList = [
    '<ol class="pdf-reference-list">',
    ...footnotes.map((note) => {
      const label = noteNumbers.get(note.id) || note.id
      const content = note.content.join(' ').trim()
      return `<li><span class="pdf-reference-label">[${escapeHtml(label)}]</span> ${renderReferenceContent(content)}</li>`
    }),
    '</ol>'
  ]

  let insertIndex = -1
  for (let index = withRefs.length - 1; index >= 0; index -= 1) {
    if (
      /^#{2,4}\s+(参考文献|参考资料|延伸阅读与参考资料)/.test(withRefs[index])
    ) {
      insertIndex = index + 1
      break
    }
  }

  if (insertIndex === -1) {
    withRefs.push('', '## 参考文献')
    insertIndex = withRefs.length
  }

  withRefs.splice(insertIndex, 0, '', ...referenceList, '')
  return withRefs.join('\n')
}

function renderReferenceContent(content) {
  const linkPattern = /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g
  let output = ''
  let lastIndex = 0

  for (const match of content.matchAll(linkPattern)) {
    output += renderReferenceText(content.slice(lastIndex, match.index))
    output += `<a href="${escapeAttribute(match[2])}">${renderReferenceText(match[1])}</a>`
    lastIndex = match.index + match[0].length
  }

  output += renderReferenceText(content.slice(lastIndex))
  return output
}

function renderReferenceText(content) {
  return escapeHtml(content)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^\w])_([^_\n]+)_($|[^\w])/g, '$1<em>$2</em>$3')
}

function transformOutsideInlineCode(line, transform) {
  const inlineCodePattern = /(`+)([^`]*?)\1/g
  let output = ''
  let lastIndex = 0

  for (const match of line.matchAll(inlineCodePattern)) {
    output += transform(line.slice(lastIndex, match.index))
    output += match[0]
    lastIndex = match.index + match[0].length
  }

  output += transform(line.slice(lastIndex))
  return output
}

function transformOutsideFencedCode(markdown, transform) {
  const lines = markdown.split('\n')
  let fence = null

  return lines
    .map((line) => {
      const fenceMatch = line.match(/^(\s*)(`{3,}|~{3,})/)

      if (fenceMatch) {
        const marker = fenceMatch[2][0]
        const length = fenceMatch[2].length

        if (!fence) {
          fence = { marker, length }
          return line
        }

        if (fence.marker === marker && length >= fence.length) {
          fence = null
          return line
        }
      }

      if (fence) return line
      return transformOutsideInlineCode(line, transform)
    })
    .join('\n')
}

function escapePdfPseudoTags(markdown) {
  const pseudoTagPattern =
    /<\/?(?:search|query|doc|think|answer|doc_result|exec_result|test_result)\b[^<>]*>/gi
  const malformedThinkPattern = /<\/?think(?=[^\x00-\x7F]|\b)/gi

  return transformOutsideFencedCode(markdown, (text) =>
    text
      .replace(pseudoTagPattern, (tag) => escapeHtml(tag))
      .replace(malformedThinkPattern, (tag) => escapeHtml(tag))
  )
}

function snapshotFile(filePath) {
  if (!fs.existsSync(filePath)) return null

  return fs.readFileSync(filePath, 'utf8')
}

function restoreFile(filePath, snapshot) {
  if (snapshot === null) {
    fs.rmSync(filePath, { force: true })
    return
  }

  fs.writeFileSync(filePath, snapshot)
}

function isExternalTarget(value) {
  return (
    !value ||
    value.startsWith('#') ||
    value.startsWith('data:') ||
    /^[a-z][a-z0-9+.-]*:/i.test(value) ||
    /^\/\//.test(value)
  )
}

function normalizeMarkdownRoute(link) {
  const { clean, suffix } = splitSuffix(link)
  if (!clean.endsWith('.md')) return link

  const normalized = clean
    .replace(/\\/g, '/')
    .replace(/^index\.md$/, '')
    .replace(/\/index\.md$/, '/')
    .replace(/\.md$/, '')

  return `${normalized}${suffix}`
}

function slugifyIdPrefix(value) {
  return value
    .replace(/\.[^.]+$/, '')
    .replace(/[^A-Za-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase()
}

function namespaceExplicitHeadingIds(markdown, prefix) {
  const idMap = new Map()
  const headingAttrsPattern =
    /^(#{1,6}\s+.+?)\s+\{((?:[#.][A-Za-z0-9][A-Za-z0-9_.:-]*)(?:\s+[#.][A-Za-z0-9][A-Za-z0-9_.:-]*)*)\}$/gm
  const namespaced = markdown.replace(
    headingAttrsPattern,
    (match, heading, attrs) => {
      const nextAttrs = attrs
        .trim()
        .split(/\s+/)
        .map((attr) => {
          if (!attr.startsWith('#')) return attr

          const previousId = attr.slice(1)
          const nextId = `${prefix}-${previousId}`
          idMap.set(previousId, nextId)
          return `#${nextId}`
        })
        .join(' ')

      return `${heading} {${nextAttrs}}`
    }
  )

  return { markdown: namespaced, idMap }
}

function rewriteLocalAnchorTarget(target, idMap) {
  if (!target.startsWith('#') || !idMap?.size) return target

  const id = target.slice(1)
  return idMap.has(id) ? `#${idMap.get(id)}` : target
}

function rewriteRelativeTarget(sourcePagePath, target, mode, idMap) {
  if (target.startsWith('#')) {
    return rewriteLocalAnchorTarget(target, idMap)
  }

  if (isExternalTarget(target) || target.startsWith('/')) {
    return target
  }

  const { clean, suffix } = splitSuffix(target)
  const sourceDir = path.dirname(sourcePagePath)
  const absoluteTarget = path.resolve(sourceDir, clean)

  if (!absoluteTarget.startsWith(`${docsDir}${path.sep}`)) {
    return target
  }

  if (mode === 'asset') {
    const relativeToGenerated = toPosix(
      path.relative(generatedDir, absoluteTarget)
    )
    return `${relativeToGenerated}${suffix}`
  }

  const docsRelative = toPosix(path.relative(docsDir, absoluteTarget))
  return normalizeMarkdownRoute(`${docsRelative}${suffix}`)
}

function rewriteMarkdownLinks(sourcePagePath, markdown, idMap) {
  let output = markdown.replace(
    /!\[([^\]]*)\]\(([^)\s]+)(\s+["'][^"']*["'])?\)/g,
    (match, alt, target, title = '') => {
      const rewritten = rewriteRelativeTarget(sourcePagePath, target, 'asset')
      return `![${alt}](${rewritten}${title})`
    }
  )

  output = output.replace(
    /(?<!!)\[([^\]]+)\]\(([^)\s]+)(\s+["'][^"']*["'])?\)/g,
    (match, label, target, title = '') => {
      const rewritten = rewriteRelativeTarget(
        sourcePagePath,
        target,
        'link',
        idMap
      )
      return `[${label}](${rewritten}${title})`
    }
  )

  output = output.replace(
    /\b(src|href)=(["'])([^"']+)\2/g,
    (match, attr, quote, target) => {
      const mode = attr === 'src' ? 'asset' : 'link'
      const rewritten = rewriteRelativeTarget(
        sourcePagePath,
        target,
        mode,
        idMap
      )
      return `${attr}=${quote}${rewritten}${quote}`
    }
  )

  return output
}

function pageAnchorForLink(link) {
  return `pdf-page-${slugifyIdPrefix(link.replace(/^\//, ''))}`
}

function chapterAnchorForItem(item) {
  return `pdf-chapter-${slugifyIdPrefix(item.link || item.text || 'chapter')}`
}

function chapterMetaForItem(item) {
  const text = item.text || ''
  const chapter = text.match(/^(\d+)\.(?!\d)\s*(.+)$/)
  if (chapter) {
    return {
      label: `第 ${chapter[1]} 章`,
      title: chapter[2].trim(),
      kind: 'chapter'
    }
  }

  const appendix = text.match(/^([A-Z])\.\s+(.+)$/)
  if (appendix) {
    return {
      label: `附录 ${appendix[1]}`,
      title: appendix[2].trim(),
      kind: 'appendix'
    }
  }

  return null
}

function collectPdfStructure(sidebar) {
  const chunks = []
  const tocSections = []
  const seenPages = new Set()

  function addPage(item, tocItems, chapterChunk = null) {
    if (!item.link || seenPages.has(item.link)) return

    seenPages.add(item.link)
    const anchor = pageAnchorForLink(item.link)
    const page = {
      type: 'page',
      title: item.text || item.link,
      link: item.link,
      anchor
    }

    chunks.push(page)
    tocItems.push({
      title: page.title,
      href: `#${anchor}`,
      children: []
    })

    if (chapterChunk) {
      chapterChunk.pages.push(page)
    }
  }

  function visitItem(item, tocItems, chapterChunk = null) {
    const chapterMeta = chapterMetaForItem(item)

    if (chapterMeta) {
      const chapter = {
        type: 'chapter',
        title: chapterMeta.title,
        label: chapterMeta.label,
        kind: chapterMeta.kind,
        anchor: chapterAnchorForItem(item),
        pages: []
      }
      const tocEntry = {
        title: `${chapter.label} ${chapter.title}`,
        href: `#${chapter.anchor}`,
        children: []
      }

      chunks.push(chapter)
      tocItems.push(tocEntry)
      addPage(item, tocEntry.children, chapter)

      for (const child of item.items || []) {
        visitItem(child, tocEntry.children, chapter)
      }

      return
    }

    addPage(item, tocItems, chapterChunk)

    for (const child of item.items || []) {
      visitItem(child, tocItems, chapterChunk)
    }
  }

  for (const section of sidebar || []) {
    const tocSection = {
      title: section.text || '未命名部分',
      children: []
    }

    tocSections.push(tocSection)

    if (section.link) {
      visitItem(section, tocSection.children)
    }

    for (const item of section.items || []) {
      visitItem(item, tocSection.children)
    }
  }

  return { chunks, tocSections }
}

function renderTocItems(items, level = 0) {
  if (!items?.length) return ''

  return [
    `<ol class="pdf-toc-list pdf-toc-level-${level}">`,
    ...items.map((item) =>
      [
        '<li>',
        `<a href="${escapeAttribute(item.href)}">${escapeHtml(item.title)}</a>`,
        renderTocItems(item.children, level + 1),
        '</li>'
      ].join('')
    ),
    '</ol>'
  ].join('')
}

function renderToc(tocSections) {
  const sections = [
    {
      title: '前置页',
      children: [
        { title: '本书简介', href: '#pdf-book-intro', children: [] },
        { title: '版本说明', href: '#pdf-publication-note', children: [] }
      ]
    },
    ...tocSections
  ]

  return [
    '<section id="pdf-toc" class="pdf-export-toc">',
    '<p class="pdf-section-eyebrow">Table of Contents</p>',
    '<h1>目录</h1>',
    ...sections.map((section) =>
      [
        '<div class="pdf-toc-section">',
        `<h2>${escapeHtml(section.title)}</h2>`,
        renderTocItems(section.children),
        '</div>'
      ].join('')
    ),
    '</section>'
  ].join('\n')
}

function renderBookCover(pageCount) {
  const { cover } = frontMatter

  return [
    '<section id="pdf-book-cover" class="pdf-export-cover">',
    '  <div class="pdf-cover-top">',
    `    <img class="pdf-cover-logo" src="${escapeAttribute(pdfLogoPath)}" alt="Hands-On Modern RL Logo">`,
    `    <p class="pdf-cover-kicker">${escapeHtml(cover.kicker)}</p>`,
    `    <h1>${escapeHtml(cover.titleEn)}</h1>`,
    `    <p class="pdf-cover-title-zh">${escapeHtml(cover.titleZh)}</p>`,
    `    <p class="pdf-cover-description">${escapeHtml(cover.description)}</p>`,
    '  </div>',
    '  <div class="pdf-cover-bottom">',
    '    <div class="pdf-cover-meta-grid">',
    `      <p><span>Version</span><strong>${escapeHtml(pdfVersion)}</strong></p>`,
    `      <p><span>Authors</span><strong>${escapeHtml(pdfAuthor)}</strong></p>`,
    `      <p><span>Repository</span><strong>${escapeHtml(pdfGithubUrl)}</strong></p>`,
    `      <p><span>Build</span><strong>${escapeHtml(pdfBuildDate)} · ${pageCount} course pages</strong></p>`,
    '    </div>',
    `    <p class="pdf-cover-publisher">${escapeHtml(cover.publisher)}</p>`,
    '  </div>',
    '</section>'
  ].join('\n')
}

function renderBookIntro(pageCount) {
  const { intro } = frontMatter

  return [
    '<section id=”pdf-book-intro” class=”pdf-book-frontmatter”>',
    '<p class=”pdf-section-eyebrow”>About This Book</p>',
    `<h1>${escapeHtml(intro.title)}</h1>`,
    ...intro.paragraphs.map((p) => `<p>${p}</p>`),
    `<h2>${escapeHtml(intro.audience.title)}</h2>`,
    '<ul>',
    ...intro.audience.items.map((item) => `<li>${escapeHtml(item)}</li>`),
    '</ul>',
    `<h2>${escapeHtml(intro.howToRead.title)}</h2>`,
    ...intro.howToRead.paragraphs.map((p) => `<p>${escapeHtml(p)}</p>`),
    `<p>本 PDF 共汇编 ${pageCount} 个课程页面，并保留章节开篇页、目录、页眉页脚、参考文献和侧边书签，目标是让它更像一本可持续更新的课本。</p>`,
    '</section>'
  ].join('\n')
}

function renderPublicationNote() {
  const { publicationNote } = frontMatter

  return [
    '<section id="pdf-publication-note" class="pdf-book-frontmatter">',
    '<p class="pdf-section-eyebrow">Edition Notes</p>',
    `<h1>${escapeHtml(publicationNote.title)}</h1>`,
    `<p><strong>版本：</strong>${escapeHtml(pdfVersion)}，构建日期：${escapeHtml(pdfBuildDate)}。</p>`,
    `<p><strong>作者：</strong>${escapeHtml(pdfAuthor)}。</p>`,
    `<p><strong>项目地址：</strong><a href="${escapeAttribute(pdfGithubUrl)}">${escapeHtml(pdfGithubUrl)}</a>。</p>`,
    `<p>${escapeHtml(publicationNote.iterationNote)}</p>`,
    `<p>${escapeHtml(publicationNote.disclaimer)}</p>`,
    `<h2>${escapeHtml(publicationNote.licenseSection.title)}</h2>`,
    `<p>${escapeHtml(publicationNote.licenseSection.text)}</p>`,
    '</section>'
  ].join('\n')
}

function renderChapterOpener(chapter) {
  const pageList = chapter.pages
    .slice(0, 10)
    .map((page) => `<li>${escapeHtml(page.title)}</li>`)
    .join('')

  return [
    `<section id="${escapeAttribute(chapter.anchor)}" class="pdf-chapter-opener">`,
    '<div>',
    `<p class="pdf-section-eyebrow">${escapeHtml(chapter.kind === 'appendix' ? 'Appendix' : 'Chapter')}</p>`,
    `<p class="pdf-chapter-label">${escapeHtml(chapter.label)}</p>`,
    `<h1>${escapeHtml(chapter.title)}</h1>`,
    `<p class="pdf-chapter-summary">本章包含 ${chapter.pages.length} 个小节，建议先扫目录，再进入正文和代码。</p>`,
    pageList ? `<ul>${pageList}</ul>` : '',
    '</div>',
    '</section>'
  ].join('\n')
}

async function generatePdfSource() {
  const config = await loadVitePressConfig()
  const sidebar = loadSidebar(config)
  const { chunks, tocSections } = collectPdfStructure(sidebar)
  const pageCount = chunks.filter((chunk) => chunk.type === 'page').length
  const missingPages = []
  const sections = []

  for (const chunk of chunks) {
    if (chunk.type === 'chapter') {
      sections.push(renderChapterOpener(chunk))
      continue
    }

    const sourcePagePath = pagePathForLink(chunk.link)

    if (!sourcePagePath) {
      missingPages.push(chunk.link)
      continue
    }

    const sourceRelative = toPosix(path.relative(docsDir, sourcePagePath))
    const source = fs.readFileSync(sourcePagePath, 'utf8')
    const idPrefix = slugifyIdPrefix(sourceRelative)
    const localized = localizeFootnotes(
      stripScriptSetup(stripFrontmatter(source)).trim()
    )
    const escaped = escapePdfPseudoTags(localized)
    const namespaced = namespaceExplicitHeadingIds(escaped, idPrefix)
    const body = rewriteMarkdownLinks(
      sourcePagePath,
      namespaced.markdown,
      namespaced.idMap
    )

    sections.push(
      [
        '<div class="pdf-export-page"></div>',
        `<div id="${escapeAttribute(chunk.anchor)}" class="pdf-page-anchor"></div>`,
        '',
        `<!-- source: ${sourceRelative} -->`,
        '',
        body
      ].join('\n')
    )
  }

  if (missingPages.length) {
    throw new Error(`Missing PDF source pages: ${missingPages.join(', ')}`)
  }

  fs.mkdirSync(generatedDir, { recursive: true })
  fs.writeFileSync(
    generatedPagePath,
    [
      '---',
      'title: Hands-on Modern RL PDF',
      'description: 现代强化学习实战 PDF 导出',
      'sidebar: false',
      'aside: false',
      'editLink: false',
      'lastUpdated: false',
      'prev: false',
      'next: false',
      'outline: false',
      '---',
      '',
      '<style>',
      ':root { --ct-doc-font-size: 12px; --ct-doc-line-height: 1.52; }',
      'html, html.dark { color-scheme: light; --vp-c-bg: #fff; --vp-c-bg-soft: #f6f6f7; --vp-c-text-1: #212121; --vp-c-text-2: #424242; --vp-c-text-3: #667085; --vp-c-divider: rgba(0, 0, 0, 0.12); }',
      '.vp-doc { font-size: 12px; line-height: 1.52; }',
      'html.dark .vp-doc, html.dark .vp-doc p, html.dark .vp-doc li, html.dark .vp-doc blockquote { color: #212121 !important; }',
      'html.dark .vp-doc h1, html.dark .vp-doc h2, html.dark .vp-doc h3, html.dark .vp-doc h4 { color: #212121 !important; }',
      'html.dark .vp-doc table th, html.dark .vp-doc table td { color: #212121 !important; border-bottom-color: rgba(0, 0, 0, 0.12) !important; }',
      '.vp-doc p, .vp-doc li { line-height: 1.52; margin-top: 6px; margin-bottom: 6px; }',
      '.vp-doc h1 { font-size: 26px; line-height: 1.18; margin: 20px 0 10px; }',
      '.vp-doc h2 { font-size: 20px; line-height: 1.24; margin: 18px 0 8px; }',
      '.vp-doc h3 { font-size: 16px; line-height: 1.28; margin: 14px 0 6px; }',
      '.vp-doc h4 { font-size: 14px; line-height: 1.3; margin: 12px 0 4px; }',
      '.vp-doc pre { font-size: 9.5px; line-height: 1.34; padding: 10px 12px; }',
      '.vp-doc code { font-size: 0.88em; }',
      '.vp-doc table { font-size: 10.5px; }',
      '.vp-doc blockquote { margin: 10px 0; padding: 8px 12px; }',
      '.vp-doc a { color: #1a237e !important; text-decoration: none; }',
      '.pdf-footnote-ref { color: #1a237e; font-weight: 700; }',
      '.pdf-reference-list { margin: 8px 0 0; padding-left: 20px; color: #212121; font-size: 10.5px; line-height: 1.42; }',
      '.pdf-reference-list li { margin: 4px 0; color: #212121; }',
      '.pdf-reference-label { color: #1a237e; font-weight: 700; }',
      '.pdf-export-cover { min-height: 82vh; display: flex; flex-direction: column; justify-content: space-between; padding: 8mm 0 2mm; }',
      '.pdf-cover-logo { width: 132mm; max-width: 100%; max-height: 54mm; margin: 0 0 18mm; object-fit: contain; object-position: left center; }',
      '.pdf-cover-kicker { margin: 0 0 8px; color: #0f4c81; font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }',
      '.pdf-export-cover h1 { margin: 0; color: #102033; font-size: 46px; line-height: 1.06; letter-spacing: 0; }',
      '.pdf-cover-title-zh { margin: 8px 0 0; color: #344054; font-size: 24px; font-weight: 700; line-height: 1.28; }',
      '.pdf-cover-description { max-width: 620px; margin: 18px 0 0; padding-top: 12px; border-top: 2px solid #f59e0b; color: #344054; font-size: 13px; line-height: 1.68; }',
      '.pdf-cover-bottom { margin-top: 20mm; }',
      '.pdf-cover-meta-grid { display: grid; grid-template-columns: 1fr 1.35fr; gap: 8px 18px; padding: 12px 0; border-top: 1px solid var(--vp-c-divider); border-bottom: 1px solid var(--vp-c-divider); }',
      '.pdf-cover-meta-grid p { margin: 0; min-width: 0; }',
      '.pdf-cover-meta-grid span { display: block; color: var(--vp-c-text-3); font-size: 9px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }',
      '.pdf-cover-meta-grid strong { display: block; margin-top: 3px; color: #212121; font-size: 11px; line-height: 1.42; overflow-wrap: anywhere; }',
      '.pdf-cover-publisher { margin: 10px 0 0; color: var(--vp-c-text-3); font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase; }',
      '.pdf-book-frontmatter { min-height: 82vh; break-before: page; page-break-before: always; break-after: page; page-break-after: always; }',
      '.pdf-book-frontmatter h1 { margin: 0 0 18px; font-size: 34px; line-height: 1.16; }',
      '.pdf-book-frontmatter h2 { margin: 22px 0 8px; font-size: 18px; }',
      '.pdf-book-frontmatter p, .pdf-book-frontmatter li { max-width: 690px; font-size: 13px; line-height: 1.72; }',
      '.pdf-book-frontmatter ul { margin: 8px 0 0; padding-left: 20px; }',
      '.pdf-export-toc { break-before: page; page-break-before: always; break-after: page; page-break-after: always; }',
      '.pdf-export-toc h1 { margin: 0 0 18px; font-size: 32px; }',
      '.pdf-toc-section { margin: 0 0 12px; }',
      '.pdf-toc-section h2 { margin: 0 0 4px; font-size: 15px; color: var(--vp-c-text-1); }',
      '.pdf-toc-list { margin: 0; padding-left: 20px; font-size: 11px; line-height: 1.4; }',
      '.pdf-toc-list li { margin: 2px 0; }',
      '.pdf-toc-level-1 { font-size: 10px; color: var(--vp-c-text-2); }',
      '.pdf-toc-level-2 { display: none; }',
      '.pdf-toc-list a { color: inherit; text-decoration: none; }',
      '.pdf-section-eyebrow { margin: 0 0 8px; color: var(--vp-c-brand-1); font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }',
      '.pdf-chapter-opener { min-height: 82vh; display: flex; align-items: center; break-before: page; page-break-before: always; break-after: page; page-break-after: always; }',
      '.pdf-chapter-opener > div { max-width: 620px; }',
      '.pdf-chapter-label { margin: 0 0 10px; color: var(--vp-c-text-2); font-size: 20px; font-weight: 700; }',
      '.pdf-chapter-opener h1 { margin: 0; font-size: 40px; line-height: 1.14; }',
      '.pdf-chapter-summary { margin: 18px 0 10px; color: var(--vp-c-text-2); font-size: 13px; }',
      '.pdf-chapter-opener ul { margin: 0; padding-left: 18px; color: var(--vp-c-text-2); font-size: 11px; line-height: 1.45; columns: 2; column-gap: 28px; }',
      '.pdf-export-page { break-before: page; page-break-before: always; height: 0; overflow: hidden; }',
      '.VPNav, .VPLocalNav, .VPSidebar, .VPDocAside, .VPDocFooter, .ct-reading-tools, .ct-sidebar-hover-area, .reading-progress { display: none !important; }',
      '.VPContent, .VPDoc { padding: 0 !important; }',
      '.VPDoc .container { display: block !important; max-width: none !important; margin: 0 !important; }',
      '.VPDoc .content { max-width: none !important; padding: 0 !important; }',
      '.VPDoc .content-container { max-width: none !important; }',
      '.vp-doc img { display: block; max-width: 100%; max-height: 110mm; margin-left: auto; margin-right: auto; object-fit: contain; }',
      '.vp-doc .mermaid-static img { max-height: 170mm; }',
      '@media print {',
      '  @page { size: A4; margin: 14mm 12mm 16mm; }',
      '  html, body { background: #fff !important; }',
      '  .pdf-export-cover { break-after: page; page-break-after: always; }',
      '  h1, h2, h3 { break-after: avoid; page-break-after: avoid; }',
      '  img, pre, table, blockquote, .custom-block { break-inside: avoid; page-break-inside: avoid; }',
      '  a { color: inherit; text-decoration: none; }',
      '}',
      '</style>',
      '',
      renderBookCover(pageCount),
      '',
      renderBookIntro(pageCount),
      '',
      renderPublicationNote(),
      '',
      renderToc(tocSections),
      '',
      sections.join('\n\n')
    ].join('\n')
  )

  console.log(`Generated ${toPosix(path.relative(rootDir, generatedPagePath))}`)
}

function runBuild() {
  if (skipBuild) return

  const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm'
  const result = spawnSync(npmCommand, ['run', 'build'], {
    cwd: rootDir,
    env: process.env,
    stdio: 'inherit'
  })

  if (result.status !== 0) {
    throw new Error(`npm run build failed with status ${result.status}`)
  }
}

function commandPath(command) {
  const result = spawnSync('command', ['-v', command], {
    shell: true,
    encoding: 'utf8'
  })

  if (result.status !== 0) return null
  return result.stdout.trim().split('\n')[0] || null
}

function findBrowserExecutable() {
  const envCandidates = [
    process.env.PUPPETEER_EXECUTABLE_PATH,
    process.env.CHROME_PATH
  ].filter(Boolean)
  const macCandidates = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge'
  ]
  const pathCandidates = [
    'chromium',
    'chromium-browser',
    'google-chrome',
    'google-chrome-stable',
    'chrome',
    'msedge'
  ]
    .map(commandPath)
    .filter(Boolean)

  return [...envCandidates, ...macCandidates, ...pathCandidates].find(
    (candidate) => fs.existsSync(candidate)
  )
}

function contentTypeFor(filePath) {
  const extension = path.extname(filePath).toLowerCase()
  const types = {
    '.css': 'text/css; charset=utf-8',
    '.gif': 'image/gif',
    '.html': 'text/html; charset=utf-8',
    '.ico': 'image/x-icon',
    '.js': 'text/javascript; charset=utf-8',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.json': 'application/json; charset=utf-8',
    '.pdf': 'application/pdf',
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
    '.webp': 'image/webp',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2'
  }

  return types[extension] || 'application/octet-stream'
}

function resolveStaticPath(urlPath, base) {
  let pathname = decodeURIComponent(urlPath)

  if (base !== '/' && pathname.startsWith(base)) {
    pathname = `/${pathname.slice(base.length)}`
  }

  let filePath = path.join(distDir, pathname)
  if (!filePath.startsWith(distDir)) return null

  if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
    filePath = path.join(filePath, 'index.html')
  } else if (!fs.existsSync(filePath) && !path.extname(filePath)) {
    const cleanUrlPath = `${filePath}.html`
    const indexPath = path.join(filePath, 'index.html')
    filePath = fs.existsSync(cleanUrlPath) ? cleanUrlPath : indexPath
  }

  if (!filePath.startsWith(distDir) || !fs.existsSync(filePath)) {
    return null
  }

  return filePath
}

function createStaticServer(base) {
  const server = http.createServer((request, response) => {
    const url = new URL(request.url || '/', 'http://localhost')
    const filePath = resolveStaticPath(url.pathname, base)

    if (!filePath) {
      response.writeHead(404)
      response.end('Not found')
      return
    }

    response.writeHead(200, {
      'content-type': contentTypeFor(filePath)
    })
    fs.createReadStream(filePath).pipe(response)
  })

  return new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      resolve({ server, origin: `http://127.0.0.1:${address.port}` })
    })
  })
}

async function waitForImages(page) {
  await page.evaluate(async () => {
    document.querySelectorAll('img[loading="lazy"]').forEach((image) => {
      image.loading = 'eager'
    })

    await Promise.all(
      Array.from(document.images).map((image) => {
        if (image.complete) return Promise.resolve()

        return new Promise((resolve) => {
          image.addEventListener('load', resolve, { once: true })
          image.addEventListener('error', resolve, { once: true })
        })
      })
    )
  })
}

async function waitBestEffort(promise, timeoutMs, label) {
  let timeoutId
  const timeout = new Promise((_, reject) => {
    timeoutId = setTimeout(
      () => reject(new Error(`${label} timed out after ${timeoutMs}ms`)),
      timeoutMs
    )
  })

  try {
    await Promise.race([promise, timeout])
  } catch (error) {
    console.warn(`${label}: ${error.message}`)
  } finally {
    clearTimeout(timeoutId)
  }
}

async function printPdf(base) {
  const executablePath = findBrowserExecutable()
  if (!executablePath) {
    throw new Error(
      'Chrome or Chromium is required. Set PUPPETEER_EXECUTABLE_PATH if it is not on PATH.'
    )
  }

  const puppeteer = await import('puppeteer-core')
  const { server, origin } = await createStaticServer(base)
  const browser = await puppeteer.default.launch({
    executablePath,
    protocolTimeout: pdfPrintTimeoutMs + 30000,
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  })

  try {
    const page = await browser.newPage()
    page.setDefaultTimeout(120000)
    const pdfUrl = `${origin}${base}__pdf__/`
    console.log(`Loading ${pdfUrl}`)
    const response = await page.goto(pdfUrl, {
      waitUntil: 'networkidle2',
      timeout: 120000
    })
    if (!response?.ok()) {
      throw new Error(
        `PDF source page failed to load: ${response?.status()} ${pdfUrl}`
      )
    }
    await waitBestEffort(
      page.evaluate(() => document.fonts?.ready),
      resourceWaitTimeoutMs,
      'Font loading'
    )
    console.log('Waiting for images')
    await waitBestEffort(
      waitForImages(page),
      resourceWaitTimeoutMs,
      'Image loading'
    )
    await page.emulateMediaType('print')
    console.log(`Printing PDF with ${pdfPrintTimeoutMs}ms timeout`)
    await page.pdf({
      path: pdfOutputPath,
      format: 'A4',
      printBackground: true,
      preferCSSPageSize: true,
      tagged: pdfGenerateOutline,
      outline: pdfGenerateOutline,
      displayHeaderFooter: true,
      headerTemplate: pdfHeaderTemplate(),
      footerTemplate: pdfFooterTemplate(),
      margin: {
        top: '14mm',
        right: '12mm',
        bottom: '16mm',
        left: '12mm'
      },
      timeout: pdfPrintTimeoutMs
    })

    const sizeMb = fs.statSync(pdfOutputPath).size / 1024 / 1024
    console.log(
      `Wrote ${toPosix(path.relative(rootDir, pdfOutputPath))} (${sizeMb.toFixed(
        1
      )} MB)`
    )
  } finally {
    await browser.close()
    await new Promise((resolve) => server.close(resolve))
  }
}

async function main() {
  const config = await loadVitePressConfig()
  const base = config.base || '/'
  const sitemapSnapshot = snapshotFile(publicSitemapPath)

  try {
    await generatePdfSource()
    runBuild()
    await printPdf(base)
  } finally {
    if (!keepSource) {
      fs.rmSync(generatedDir, { recursive: true, force: true })
    }

    restoreFile(publicSitemapPath, sitemapSnapshot)
  }
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
