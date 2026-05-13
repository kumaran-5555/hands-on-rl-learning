/* global process */
import { defineConfig } from 'vitepress'
import { createLogger } from 'vite'
import { MermaidMarkdown } from 'vitepress-plugin-mermaid'
import { createRequire } from 'module'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const require = createRequire(import.meta.url)
const markdownItFootnote = require('markdown-it-footnote')
const markdownItContainer = require('markdown-it-container')
const katex = require('katex')

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const packageJsonPath = path.resolve(__dirname, '../../package.json')
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))
const docsRoot = path.resolve(__dirname, '..')
const assetManifestPath = path.resolve(
  docsRoot,
  'public/optimized/asset-manifest.json'
)

function loadAssetManifest() {
  if (!fs.existsSync(assetManifestPath)) return { assets: {} }

  try {
    return JSON.parse(fs.readFileSync(assetManifestPath, 'utf8'))
  } catch {
    return { assets: {} }
  }
}

const assetManifest = loadAssetManifest()

const isVercel = process.env.VERCEL === '1' || !!process.env.VERCEL_URL

function parseRepository() {
  const repositoryUrl =
    process.env.GITHUB_REPOSITORY ||
    packageJson.repository?.url ||
    packageJson.repository ||
    ''

  if (repositoryUrl.includes('/')) {
    if (repositoryUrl.includes(':')) {
      const sshMatch = repositoryUrl.match(/github\.com:(.+?)\/(.+?)(\.git)?$/)
      if (sshMatch) {
        return { owner: sshMatch[1], repo: sshMatch[2] }
      }
    }

    const normalized = repositoryUrl
      .replace(/^https?:\/\/github\.com\//, '')
      .replace(/^git@github\.com:/, '')
      .replace(/\.git$/, '')

    if (normalized.includes('/')) {
      const [owner, repo] = normalized.split('/')
      return { owner, repo }
    }
  }

  return { owner: 'walkinglabs', repo: packageJson.name || 'course-template' }
}

const { owner, repo } = parseRepository()
const base = process.env.BASE || (isVercel ? '/' : `/${repo}/`)
const siteUrl = process.env.SITE_URL || `https://${owner}.github.io/${repo}`
const editLinkPattern = `https://github.com/${owner}/${repo}/edit/main/docs/:path`
const enableLocalSearch = process.env.LOCAL_SEARCH !== '0'
const mermaidConfig = {
  securityLevel: 'loose',
  startOnLoad: false
}

function mermaidConfigPlugin() {
  const virtualModuleId = 'virtual:mermaid-config'
  const resolvedVirtualModuleId = `\0${virtualModuleId}`

  return {
    name: 'local-mermaid-config',
    resolveId(id) {
      if (id === virtualModuleId) {
        return resolvedVirtualModuleId
      }
    },
    load(id) {
      if (id === resolvedVirtualModuleId) {
        return `export default ${JSON.stringify(mermaidConfig)}`
      }
    }
  }
}

function normalizeBrokenDocPathPlugin() {
  const canonicalSegments = ['appendix_math', 'linear-algebra-basics']

  return {
    name: 'normalize-broken-doc-path',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url) return next()

        const url = new URL(req.url, 'http://localhost')
        if (!url.pathname.includes('appendix')) return next()

        const decodedPathname = decodeURIComponent(url.pathname)
        const normalizedDecodedPathname = decodedPathname
          .replace(/\s+/g, '')
          .replace(/appendix_m+ath/gi, canonicalSegments[0])
          .replace(/linea+r-algebra-basics/gi, canonicalSegments[1])

        if (normalizedDecodedPathname !== decodedPathname) {
          const normalizedPathname = normalizedDecodedPathname
            .split('/')
            .map((segment) => encodeURIComponent(segment))
            .join('/')
          const redirectTarget = `${normalizedPathname}${url.search}`
          res.statusCode = 302
          res.setHeader('Location', redirectTarget)
          res.end()
          return
        }

        next()
      })
    }
  }
}

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function slugifySearchHeading(value) {
  return value
    .normalize('NFKD')
    .replace(/[\u0300-\u036F]/g, '')
    .replace(/[\x00-\x1f]/g, '')
    .replace(/[\s~`!@#$%^&*()\-_+=[\]{}|\\;:"'“”‘’<>,.?/]+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/^(\d)/, '_$1')
    .toLowerCase()
}

function stripMarkdown(value) {
  return value
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*_~>#-]/g, '')
    .trim()
}

function renderSearchMarkdown(src) {
  const html = []
  const slugCounts = new Map()
  let inFence = false

  for (const rawLine of src.replace(/^---[\s\S]*?---\n/, '').split('\n')) {
    const line = rawLine.trim()

    if (line.startsWith('```')) {
      inFence = !inFence
      continue
    }

    if (!line || inFence || line.startsWith(':::')) {
      continue
    }

    const heading = /^(#{1,6})\s+(.+)$/.exec(line)
    if (heading) {
      const level = heading[1].length
      const title = stripMarkdown(heading[2])
      const escapedTitle = escapeHtml(title)
      const baseSlug = slugifySearchHeading(title)
      const count = slugCounts.get(baseSlug) || 0
      slugCounts.set(baseSlug, count + 1)
      const slug = count === 0 ? baseSlug : `${baseSlug}-${count}`
      html.push(
        `<h${level}>${escapedTitle}<a class="header-anchor" href="#${slug}"></a></h${level}>`
      )
      continue
    }

    html.push(`<p>${escapeHtml(stripMarkdown(line))}</p>`)
  }

  return html.join('\n')
}

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
  return katex.renderToString(content, {
    displayMode,
    output: 'htmlAndMathml',
    throwOnError: false,
    strict: false,
    trust: true
  })
}

function rescueMathInInline(md) {
  md.core.ruler.push('math_inline_rescue', function (state) {
    for (let i = 0; i < state.tokens.length; i++) {
      const token = state.tokens[i]
      if (token.type !== 'inline' || !token.children) continue

      let needsRescue = false
      for (const child of token.children) {
        if (child.type === 'text' && /\$[^$]+\$/.test(child.content)) {
          needsRescue = true
          break
        }
      }
      if (!needsRescue) continue

      const newChildren = []
      for (const child of token.children) {
        if (child.type === 'text' && child.content.includes('$')) {
          const parts = child.content.split(/(\$[^$]+\$)/)
          for (const part of parts) {
            if (!part) continue
            if (part.startsWith('$') && part.endsWith('$') && part.length > 2) {
              const t = new state.Token('math_inline', 'math', 0)
              t.content = part.slice(1, -1)
              t.markup = '$'
              newChildren.push(t)
            } else {
              const t = new state.Token('text', '', 0)
              t.content = part
              newChildren.push(t)
            }
          }
        } else {
          newChildren.push(child)
        }
      }
      token.children = newChildren
    }
  })
}

function katexMarkdown(md) {
  md.inline.ruler.before('text', 'math_inline', mathInline)
  md.block.ruler.after('blockquote', 'math_block', mathBlock, {
    alt: ['paragraph', 'reference', 'blockquote', 'list']
  })
  md.renderer.rules.math_inline = (tokens, idx) =>
    renderKatex(tokens[idx].content, false)
  md.renderer.rules.math_block = (tokens, idx) =>
    `<p>${renderKatex(tokens[idx].content, true)}</p>\n`
  rescueMathInInline(md)
}

function safeHeadingAttrs(md) {
  md.core.ruler.before('linkify', 'safe_heading_attrs', (state) => {
    for (let idx = 0; idx < state.tokens.length - 1; idx += 1) {
      const headingOpen = state.tokens[idx]
      const inline = state.tokens[idx + 1]

      if (headingOpen.type !== 'heading_open' || inline.type !== 'inline') {
        continue
      }

      const children = inline.children || []
      const lastText = [...children]
        .reverse()
        .find((token) => token.type === 'text')
      if (!lastText) continue

      const match = lastText.content.match(
        /\s*\{((?:[#.][A-Za-z0-9][A-Za-z0-9_.:-]*)(?:\s+[#.][A-Za-z0-9][A-Za-z0-9_.:-]*)*)\}$/
      )
      if (!match) continue

      const attrs = match[1].trim().split(/\s+/)
      const classes = []

      for (const attr of attrs) {
        if (attr.startsWith('#')) {
          headingOpen.attrSet('id', attr.slice(1))
        } else if (attr.startsWith('.')) {
          classes.push(attr.slice(1))
        }
      }

      if (classes.length) {
        headingOpen.attrJoin('class', classes.join(' '))
      }

      lastText.content = lastText.content.slice(0, match.index)
      inline.content = inline.content.replace(match[0], '')
    }
  })
}

function isExternalAsset(value) {
  return (
    /^(?:[a-z][a-z0-9+.-]*:)?\/\//i.test(value) || value.startsWith('data:')
  )
}

function resolveMarkdownAsset(relativePagePath, src) {
  if (!src || isExternalAsset(src) || src.startsWith('/')) return null

  const hashIndex = src.indexOf('#')
  const queryIndex = src.indexOf('?')
  const suffixIndexCandidates = [hashIndex, queryIndex].filter(
    (idx) => idx >= 0
  )
  const suffixIndex = suffixIndexCandidates.length
    ? Math.min(...suffixIndexCandidates)
    : -1
  const cleanSrc = suffixIndex >= 0 ? src.slice(0, suffixIndex) : src
  const suffix = suffixIndex >= 0 ? src.slice(suffixIndex) : ''
  const pageDir = path.posix.dirname(relativePagePath || '')
  const sourcePath = path.posix
    .normalize(path.posix.join(pageDir, cleanSrc))
    .replace(/^\.\//, '')

  return {
    sourcePath,
    suffix
  }
}

function optimizedImagesPlugin(md) {
  const imageRule = md.renderer.rules.image

  md.renderer.rules.image = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    const src = token.attrGet('src')
    const resolved = resolveMarkdownAsset(env.relativePath, src)
    const asset = resolved && assetManifest.assets?.[resolved.sourcePath]

    if (asset?.status === 'optimized' && asset.optimized) {
      token.attrSet('src', `${asset.optimized}${resolved.suffix}`)
      token.attrSet('data-source-src', `/src/${resolved.sourcePath}`)
    }

    if (!token.attrGet('loading')) {
      token.attrSet('loading', 'lazy')
    }

    if (!token.attrGet('decoding')) {
      token.attrSet('decoding', 'async')
    }

    return imageRule(tokens, idx, options, env, self)
  }
}

function mermaidBlockIndex(tokens, idx) {
  let index = 0

  for (let i = 0; i <= idx; i += 1) {
    const info = tokens[i].info?.trim()
    if (info === 'mermaid' || info === 'mmd') {
      index += 1
    }
  }

  return index
}

function buildMermaidManifestMap() {
  return new Map(
    (assetManifest.mermaid || [])
      .filter((block) => block.status === 'optimized' && block.optimized)
      .map((block) => [`${block.page}:${block.index}`, block])
  )
}

const mermaidManifest = buildMermaidManifestMap()

function optimizedMermaidPlugin(md) {
  const fenceRule = md.renderer.rules.fence

  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    const info = token.info.trim()

    if (info === 'mermaid' || info === 'mmd') {
      const index = mermaidBlockIndex(tokens, idx)
      const block = mermaidManifest.get(`${env.relativePath}:${index}`)

      if (block?.optimized) {
        const src = md.utils.escapeHtml(block.optimized)
        const source = md.utils.escapeHtml(block.source)
        const page = md.utils.escapeHtml(block.page)

        return `<p class="mermaid-static"><img src="${src}" alt="Mermaid diagram" data-mermaid-viewer="true" data-source-src="/src/${source}" data-source-page="/src/${page}" data-source-index="${block.index}" loading="lazy" decoding="async"></p>\n`
      }
    }

    return fenceRule(tokens, idx, options, env, self)
  }
}

const zhNav = [
  { text: '前言与导论', link: '/preface/intro' },
  { text: '基础导论', link: '/chapter01_cartpole/intro' },
  { text: '核心理论与方法', link: '/chapter03_mdp/intro' },
  { text: '大模型强化学习', link: '/chapter08_rlhf/intro' },
  {
    text: '前沿与进阶专题',
    link: '/chapter11_vlm_rl/intro'
  }
]

const enNav = [
  { text: 'Preface', link: '/en/preface/intro' },
  { text: 'Fundamentals', link: '/en/chapter01_cartpole/intro' },
  { text: 'Core Theory', link: '/en/chapter03_mdp/intro' },
  { text: 'LLM RL', link: '/en/chapter08_rlhf/intro' },
  {
    text: 'Frontier Topics',
    link: '/en/chapter11_vlm_rl/intro'
  }
]

const zhSidebar = {
  '/': [
    {
      text: '前言',
      items: [
        { text: '课程导读', link: '/preface/intro' },
        { text: '强化学习简史', link: '/preface/brief-history' },
        { text: '环境安装指南', link: '/preface/env-setup' }
      ]
    },
    {
      text: '基础导论',
      items: [
        {
          text: '1. CartPole 倒立摆',
          link: '/chapter01_cartpole/intro',
          collapsed: false,
          items: [
            {
              text: '1.1 基本概念',
              link: '/chapter01_cartpole/principles'
            },
            {
              text: '1.2 训练指标',
              link: '/chapter01_cartpole/metrics'
            }
          ]
        },
        {
          text: '2. DPO 偏好微调',
          link: '/chapter02_dpo/intro',
          collapsed: false,
          items: [
            {
              text: '2.1 DPO 推导',
              link: '/chapter02_dpo/principles'
            },
            {
              text: '2.2 训练指标',
              link: '/chapter02_dpo/metrics'
            }
          ]
        },
        { text: '本篇小结', link: '/summaries/part1-summary' }
      ]
    },
    {
      text: '核心理论与方法',
      items: [
        {
          text: '3. MDP 与价值函数',
          link: '/chapter03_mdp/intro',
          collapsed: false,
          items: [
            {
              text: '3.1 两台老虎机问题',
              link: '/chapter03_mdp/bandit'
            },
            { text: '3.2 马尔可夫决策过程', link: '/chapter03_mdp/mdp' },
            {
              text: '3.3 价值函数与贝尔曼方程',
              link: '/chapter03_mdp/value-bellman'
            },
            {
              text: '3.4 DP、MC 与 TD',
              link: '/chapter03_mdp/dp-mc-td'
            },
            { text: '3.5 从 Q 到 Q-Learning', link: '/chapter03_mdp/value-q' },
            {
              text: '3.6 从价值到策略',
              link: '/chapter03_mdp/policy-objective'
            },
            {
              text: '3.7 数据从哪里来',
              link: '/chapter03_mdp/algorithm-taxonomy'
            },
            {
              text: '3.8 奖励函数设计',
              link: '/chapter03_mdp/reward-design'
            },
            {
              text: '3.9 本章总结',
              link: '/chapter03_mdp/panorama'
            }
          ]
        },
        {
          text: '4. 深度 Q 网络',
          link: '/chapter04_dqn/intro',
          collapsed: false,
          items: [
            {
              text: '4.1 DQN 的必要性',
              link: '/chapter04_dqn/from-q-to-dqn'
            },
            {
              text: '4.2 DQN 的结构',
              link: '/chapter04_dqn/dqn-components'
            },
            {
              text: '4.3 动手：LunarLander 实战',
              link: '/chapter04_dqn/lunar-lander'
            },
            {
              text: '4.4 DQN 改进家族',
              link: '/chapter04_dqn/dqn-family'
            },
            {
              text: '4.5 动手：视觉游戏项目',
              link: '/chapter04_dqn/visual-game-projects'
            }
          ]
        },
        {
          text: '5. Policy-Based 方法',
          link: '/chapter05_policy_gradient/intro',
          collapsed: false,
          items: [
            {
              text: '5.1 为什么需要策略梯度',
              link: '/chapter05_policy_gradient/pg-necessity'
            },
            {
              text: '5.2 策略梯度定理与 REINFORCE',
              link: '/chapter05_policy_gradient/reinforce'
            },
            {
              text: '5.3 动手：策略梯度实战 CartPole',
              link: '/chapter05_policy_gradient/cartpole'
            },
            {
              text: '5.4 策略梯度的方差与基线',
              link: '/chapter05_policy_gradient/pg-improvements'
            },
            {
              text: '5.5 动手：带基线的策略梯度',
              link: '/chapter05_policy_gradient/cartpole-baseline'
            }
          ]
        },
        {
          text: '6. Actor-Critic',
          link: '/chapter06_actor_critic/intro',
          collapsed: false,
          items: [
            {
              text: '6.1 优势函数',
              link: '/chapter06_actor_critic/advantage-function'
            },
            {
              text: '6.2 Critic 训练',
              link: '/chapter06_actor_critic/critic-training'
            },
            {
              text: '6.3 Actor-Critic 架构',
              link: '/chapter06_actor_critic/actor-critic'
            },
            {
              text: '6.4 动手：Pendulum 摆杆平衡',
              link: '/chapter06_actor_critic/pendulum'
            },
            {
              text: '6.5 动手：BipedalWalker 双足行走',
              link: '/chapter06_actor_critic/bipedalwalker'
            },
            {
              text: '6.6 Actor-Critic 的前沿大规模应用',
              link: '/chapter06_actor_critic/ac-frontier'
            }
          ]
        },
        {
          text: '7. PPO',
          link: '/chapter07_ppo/intro',
          collapsed: false,
          items: [
            {
              text: '7.1 动手：BipedalWalker 连续控制',
              link: '/chapter07_ppo/ppo-bipedal-walker'
            },
            {
              text: '7.2 PPO 推导',
              link: '/chapter07_ppo/ppo-math'
            },
            {
              text: '7.3 信任域与裁剪机制',
              link: '/chapter07_ppo/trust-region-clipping'
            },
            {
              text: '7.4 GAE 与奖励模型',
              link: '/chapter07_ppo/gae-reward-model'
            },
            {
              text: '7.5 PPO 游戏项目实践导论',
              link: '/chapter07_ppo/ppo-game-benchmark'
            },
            {
              text: '7.6 长程任务中的 RL',
              link: '/chapter07_ppo/rl-long-horizon-planning'
            }
          ]
        },
        { text: '本篇小结', link: '/summaries/part2-summary' }
      ]
    },
    {
      text: '大模型 RL',
      items: [
        {
          text: '8. RLHF 全流程',
          link: '/chapter08_rlhf/intro',
          collapsed: false,
          items: [
            {
              text: '8.1 Base 模型与对齐助手',
              link: '/chapter08_rlhf/base-model-to-assistant'
            },
            {
              text: '8.2 RLHF 流水线',
              link: '/chapter08_rlhf/standard-rlhf-pipeline'
            },
            {
              text: '8.3 SFT 指令微调',
              link: '/chapter08_rlhf/imitation-learning-pipeline'
            },
            {
              text: '8.4 奖励模型',
              link: '/chapter08_rlhf/reward-function-design'
            },
            {
              text: '8.5 PPO-RLHF 对齐',
              link: '/chapter08_rlhf/ppo-rlhf-loop'
            },
            {
              text: '8.6 评估与奖励黑客',
              link: '/chapter08_rlhf/evaluation'
            }
          ]
        },
        {
          text: '9. 后训练对齐',
          link: '/chapter09_alignment/intro',
          collapsed: false,
          items: [
            {
              text: '9.1 DPO 原理、数学与选型',
              link: '/chapter09_alignment/dpo-theory-and-family'
            },
            {
              text: '9.2 GRPO 训练与核心机制',
              link: '/chapter09_grpo_rlvr/grpo-practice-and-mechanism'
            },
            {
              text: '9.3 R1-Zero 范式',
              link: '/chapter09_grpo_rlvr/deepseek-dapo'
            },
            {
              text: '9.4 RLVR 可验证奖励',
              link: '/chapter09_grpo_rlvr/rlvr'
            },
            {
              text: '9.5 OPD 在线蒸馏',
              link: '/chapter09_grpo_rlvr/on-policy-distillation'
            },
            {
              text: '9.6 后训练工业实践',
              link: '/chapter09_alignment/industrial-post-training'
            }
          ]
        },
        {
          text: '10. Agentic RL',
          link: '/chapter10_agentic_rl/intro',
          collapsed: false,
          items: [
            {
              text: '10.1 多轮交互',
              link: '/chapter10_agentic_rl/multi-turn-rl'
            },
            {
              text: '10.2 工具调用',
              link: '/chapter10_agentic_rl/tool-use-and-trajectory'
            },
            {
              text: '10.3 评测与案例',
              link: '/chapter10_agentic_rl/industrial-evaluation'
            },
            {
              text: '10.4 动手：Agent 数据制造',
              link: '/chapter10_agentic_rl/agent-data-swe-smith'
            },
            {
              text: '10.5 动手：rLLM DeepCoder',
              link: '/chapter10_agentic_rl/rllm-deepcoder-lab'
            },
            {
              text: '10.6 动手：金融问答 Agent',
              link: '/chapter10_agentic_rl/rllm-finqa-lab'
            },
            {
              text: '10.7 Deep Research',
              link: '/chapter10_agentic_rl/deep-research-agent'
            },
            {
              text: '10.8 Agentic RL 训练系统',
              link: '/chapter10_agentic_rl/build-agentic-training-system'
            },
            {
              text: '10.9 延伸阅读',
              link: '/chapter10_agentic_rl/extended-readings'
            }
          ]
        },
        { text: '本篇小结', link: '/summaries/part3-summary' }
      ]
    },
    {
      text: '前沿',
      items: [
        {
          text: '11. VLM 强化学习',
          link: '/chapter11_vlm_rl/intro',
          collapsed: false,
          items: [
            {
              text: '11.1 VLM 强化训练',
              link: '/chapter11_vlm_rl/vlm-grpo-hands-on'
            },
            {
              text: '11.2 视觉奖励信号',
              link: '/chapter11_vlm_rl/vlm-challenges'
            },
            {
              text: '11.3 VLM RL 推理框架',
              link: '/chapter11_vlm_rl/vlm-frameworks'
            },
            {
              text: '11.4 视觉生成 RL',
              link: '/chapter11_vlm_rl/visual-generation-rl'
            },
            {
              text: '11.5 动手：GeoQA 几何推理',
              link: '/chapter11_vlm_rl/easyr1-geoqa'
            }
          ]
        },
        {
          text: '12. 未来趋势',
          link: '/chapter12_future_trends/intro',
          collapsed: false,
          items: [
            {
              text: '12.1 具身智能',
              link: '/chapter12_future_trends/embodied-intelligence/'
            },
            {
              text: '12.2 模型式强化学习',
              link: '/chapter12_future_trends/embodied-intelligence/model-based-rl/'
            },
            {
              text: '12.3 自我博弈',
              link: '/chapter12_future_trends/self-play-outlook/'
            },
            {
              text: '12.4 多智能体',
              link: '/chapter12_future_trends/llm-multi-agent-rl/'
            },
            {
              text: '12.5 离线强化学习',
              link: '/chapter12_future_trends/offline-rl/'
            },
            {
              text: '12.6 规模化趋势',
              link: '/chapter12_future_trends/rl-scaling-outlook'
            }
          ]
        },
        { text: '本篇小结', link: '/summaries/part4-summary' }
      ]
    },
    {
      text: '附录',
      items: [
        {
          text: 'A. 训练调试指南',
          link: '/appendix_common_pitfalls/intro'
        },
        {
          text: 'B. RL 工程实践',
          link: '/appendix_industrial_training/intro',
          collapsed: false,
          items: [
            {
              text: 'B.1 训练系统底座',
              link: '/appendix_industrial_training/rl-infrastructure'
            },
            {
              text: 'B.2 Agent 沙箱',
              link: '/appendix_industrial_training/agentic-rl-infra'
            },
            {
              text: 'B.3 评测基准',
              link: '/appendix_industrial_training/evaluation-badcase'
            },
            {
              text: 'B.4 训练指标词典',
              link: '/appendix_industrial_training/metrics-glossary'
            },
            {
              text: 'B.5 工业实战练习',
              link: '/appendix_industrial_training/industrial-exercises'
            }
          ]
        },
        {
          text: 'C. 手写代码速记',
          link: '/appendix_code_cheatsheet/intro',
          collapsed: false,
          items: [
            {
              text: 'C.1 SFT 与 KL',
              link: '/appendix_code_cheatsheet/sft-kl'
            },
            {
              text: 'C.2 PPO 与 GAE',
              link: '/appendix_code_cheatsheet/ppo-gae'
            },
            {
              text: 'C.3 DPO 家族',
              link: '/appendix_code_cheatsheet/dpo-family'
            },
            {
              text: 'C.4 GRPO 与奖励模型',
              link: '/appendix_code_cheatsheet/grpo-rlvr'
            },
            {
              text: 'C.5 Softmax 与交叉熵',
              link: '/appendix_code_cheatsheet/softmax-ce'
            },
            {
              text: 'C.6 采样方法',
              link: '/appendix_code_cheatsheet/top-k-top-p'
            },
            {
              text: 'C.7 注意力机制',
              link: '/appendix_code_cheatsheet/attention-mha'
            },
            {
              text: 'C.8 DAPO',
              link: '/appendix_code_cheatsheet/dapo'
            }
          ]
        },
        {
          text: 'D. 学习资料与复现项目推荐',
          link: '/appendix_game_projects/intro'
        },
        {
          text: 'E. 强化学习的数学基础',
          link: '/appendix_math/intro',
          collapsed: false,
          items: [
            {
              text: 'E.1 数学对象与线性代数',
              link: '/appendix_math/linear-algebra',
              collapsed: true,
              items: [
                {
                  text: '基础对象',
                  link: '/appendix_math/linear-algebra-basics'
                },
                {
                  text: '贝尔曼矩阵',
                  link: '/appendix_math/linear-algebra-bellman'
                },
                {
                  text: '函数近似',
                  link: '/appendix_math/linear-algebra-function-approx'
                },
                {
                  text: '收敛与信任域',
                  link: '/appendix_math/linear-algebra-advanced'
                },
                {
                  text: '公式与练习',
                  link: '/appendix_math/linear-algebra-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.2 概率、期望与随机估计',
              link: '/appendix_math/probability-statistics',
              collapsed: true,
              items: [
                { text: '概率基础', link: '/appendix_math/probability-basics' },
                {
                  text: '回报与价值',
                  link: '/appendix_math/probability-value'
                },
                {
                  text: '采样估计',
                  link: '/appendix_math/probability-sampling'
                },
                {
                  text: '轨迹与 GAE',
                  link: '/appendix_math/probability-trajectory-td'
                },
                {
                  text: '贝尔曼期望',
                  link: '/appendix_math/probability-bellman-advanced'
                },
                {
                  text: '公式与练习',
                  link: '/appendix_math/probability-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.3 微积分与优化',
              link: '/appendix_math/calculus-optimization',
              collapsed: true,
              items: [
                { text: '导数与梯度', link: '/appendix_math/calculus-basics' },
                {
                  text: '策略梯度',
                  link: '/appendix_math/calculus-policy-gradient'
                },
                { text: 'PPO 与 Adam', link: '/appendix_math/calculus-ppo' },
                {
                  text: '推导工具',
                  link: '/appendix_math/calculus-derivations'
                },
                {
                  text: '完整公式',
                  link: '/appendix_math/calculus-advanced-formulas'
                },
                {
                  text: '公式与练习',
                  link: '/appendix_math/calculus-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.4 信息论与分布距离',
              link: '/appendix_math/information-theory',
              collapsed: true,
              items: [
                { text: '熵与探索', link: '/appendix_math/information-basics' },
                {
                  text: '交叉熵与 KL',
                  link: '/appendix_math/information-cross-entropy-kl'
                },
                {
                  text: 'RLHF 与 DPO',
                  link: '/appendix_math/information-rlhf-dpo'
                },
                {
                  text: '互信息',
                  link: '/appendix_math/information-mutual-info'
                },
                {
                  text: '完整公式',
                  link: '/appendix_math/information-advanced-formulas'
                },
                {
                  text: '公式与练习',
                  link: '/appendix_math/information-formulas-exercises'
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

const enSidebar = {
  '/en/': [
    {
      text: 'Preface',
      items: [
        { text: 'Course Overview', link: '/en/preface/intro' },
        { text: 'A Brief History of RL', link: '/en/preface/brief-history' },
        { text: 'Environment Setup', link: '/en/preface/env-setup' }
      ]
    },
    {
      text: 'Fundamentals',
      items: [
        {
          text: '1. CartPole Balancing',
          link: '/en/chapter01_cartpole/intro',
          collapsed: false,
          items: [
            {
              text: '1.1 Core Concepts',
              link: '/en/chapter01_cartpole/principles'
            },
            {
              text: '1.2 Training Metrics',
              link: '/en/chapter01_cartpole/metrics'
            }
          ]
        },
        {
          text: '2. DPO Preference Tuning',
          link: '/en/chapter02_dpo/intro',
          collapsed: false,
          items: [
            {
              text: '2.1 DPO Derivation',
              link: '/en/chapter02_dpo/principles'
            },
            { text: '2.2 Training Metrics', link: '/en/chapter02_dpo/metrics' }
          ]
        },
        { text: 'Part I Summary', link: '/en/summaries/part1-summary' }
      ]
    },
    {
      text: 'Core Theory and Methods',
      items: [
        {
          text: '3. MDP and Value Functions',
          link: '/en/chapter03_mdp/intro',
          collapsed: false,
          items: [
            {
              text: '3.1 The Two-Armed Bandit',
              link: '/en/chapter03_mdp/bandit'
            },
            {
              text: '3.2 Markov Decision Processes',
              link: '/en/chapter03_mdp/mdp'
            },
            {
              text: '3.3 Value Functions & Bellman',
              link: '/en/chapter03_mdp/value-bellman'
            },
            { text: '3.4 DP, MC, and TD', link: '/en/chapter03_mdp/dp-mc-td' },
            {
              text: '3.5 From V to Q-Learning',
              link: '/en/chapter03_mdp/value-q'
            },
            {
              text: '3.6 From Value to Policy',
              link: '/en/chapter03_mdp/policy-objective'
            },
            {
              text: '3.7 Where Does Data Come From',
              link: '/en/chapter03_mdp/algorithm-taxonomy'
            },
            {
              text: '3.8 Reward Function Design',
              link: '/en/chapter03_mdp/reward-design'
            },
            { text: '3.9 Chapter Summary', link: '/en/chapter03_mdp/panorama' }
          ]
        },
        {
          text: '4. Deep Q-Networks',
          link: '/en/chapter04_dqn/intro',
          collapsed: false,
          items: [
            {
              text: '4.1 Why DQN Is Needed',
              link: '/en/chapter04_dqn/from-q-to-dqn'
            },
            {
              text: '4.2 DQN Architecture',
              link: '/en/chapter04_dqn/dqn-components'
            },
            {
              text: '4.3 Hands-On: LunarLander',
              link: '/en/chapter04_dqn/lunar-lander'
            },
            {
              text: '4.4 DQN Improvement Family',
              link: '/en/chapter04_dqn/dqn-family'
            },
            {
              text: '4.5 Hands-On: Visual Games',
              link: '/en/chapter04_dqn/visual-game-projects'
            }
          ]
        },
        {
          text: '5. Policy-Based Methods',
          link: '/en/chapter05_policy_gradient/intro',
          collapsed: false,
          items: [
            {
              text: '5.1 Why Policy Gradients',
              link: '/en/chapter05_policy_gradient/pg-necessity'
            },
            {
              text: '5.2 Policy Gradient & REINFORCE',
              link: '/en/chapter05_policy_gradient/reinforce'
            },
            {
              text: '5.3 Hands-On: PG CartPole',
              link: '/en/chapter05_policy_gradient/cartpole'
            },
            {
              text: '5.4 Variance and Baselines',
              link: '/en/chapter05_policy_gradient/pg-improvements'
            },
            {
              text: '5.5 Hands-On: PG with Baseline',
              link: '/en/chapter05_policy_gradient/cartpole-baseline'
            }
          ]
        },
        {
          text: '6. Actor-Critic',
          link: '/en/chapter06_actor_critic/intro',
          collapsed: false,
          items: [
            {
              text: '6.1 The Advantage Function',
              link: '/en/chapter06_actor_critic/advantage-function'
            },
            {
              text: '6.2 Training the Critic',
              link: '/en/chapter06_actor_critic/critic-training'
            },
            {
              text: '6.3 Actor-Critic Architecture',
              link: '/en/chapter06_actor_critic/actor-critic'
            },
            {
              text: '6.4 Hands-On: Pendulum',
              link: '/en/chapter06_actor_critic/pendulum'
            },
            {
              text: '6.5 Hands-On: BipedalWalker',
              link: '/en/chapter06_actor_critic/bipedalwalker'
            },
            {
              text: '6.6 Actor-Critic at Scale',
              link: '/en/chapter06_actor_critic/ac-frontier'
            }
          ]
        },
        {
          text: '7. PPO',
          link: '/en/chapter07_ppo/intro',
          collapsed: false,
          items: [
            {
              text: '7.1 Hands-On: BipedalWalker',
              link: '/en/chapter07_ppo/ppo-bipedal-walker'
            },
            { text: '7.2 PPO Derivation', link: '/en/chapter07_ppo/ppo-math' },
            {
              text: '7.3 Trust Region & Clipping',
              link: '/en/chapter07_ppo/trust-region-clipping'
            },
            {
              text: '7.4 GAE and Reward Models',
              link: '/en/chapter07_ppo/gae-reward-model'
            },
            {
              text: '7.5 PPO Game Benchmarks',
              link: '/en/chapter07_ppo/ppo-game-benchmark'
            },
            {
              text: '7.6 RL in Long-Horizon Tasks',
              link: '/en/chapter07_ppo/rl-long-horizon-planning'
            }
          ]
        },
        { text: 'Part II Summary', link: '/en/summaries/part2-summary' }
      ]
    },
    {
      text: 'LLM Reinforcement Learning',
      items: [
        {
          text: '8. The RLHF Pipeline',
          link: '/en/chapter08_rlhf/intro',
          collapsed: false,
          items: [
            {
              text: '8.1 Base Model to Assistant',
              link: '/en/chapter08_rlhf/base-model-to-assistant'
            },
            {
              text: '8.2 RLHF Pipeline',
              link: '/en/chapter08_rlhf/standard-rlhf-pipeline'
            },
            {
              text: '8.3 SFT Instruction Tuning',
              link: '/en/chapter08_rlhf/imitation-learning-pipeline'
            },
            {
              text: '8.4 Reward Models',
              link: '/en/chapter08_rlhf/reward-function-design'
            },
            {
              text: '8.5 PPO-RLHF Alignment',
              link: '/en/chapter08_rlhf/ppo-rlhf-loop'
            },
            {
              text: '8.6 Evaluation & Reward Hacking',
              link: '/en/chapter08_rlhf/evaluation'
            }
          ]
        },
        {
          text: '9. Post-Training Alignment',
          link: '/en/chapter09_alignment/intro',
          collapsed: false,
          items: [
            {
              text: '9.1 DPO Theory and Selection',
              link: '/en/chapter09_alignment/dpo-theory-and-family'
            },
            {
              text: '9.2 GRPO Training',
              link: '/en/chapter09_grpo_rlvr/grpo-practice-and-mechanism'
            },
            {
              text: '9.3 The R1-Zero Paradigm',
              link: '/en/chapter09_grpo_rlvr/deepseek-dapo'
            },
            {
              text: '9.4 RLVR: Verifiable Rewards',
              link: '/en/chapter09_grpo_rlvr/rlvr'
            },
            {
              text: '9.5 On-Policy Distillation',
              link: '/en/chapter09_grpo_rlvr/on-policy-distillation'
            },
            {
              text: '9.6 Industrial Post-Training',
              link: '/en/chapter09_alignment/industrial-post-training'
            }
          ]
        },
        {
          text: '10. Agentic RL',
          link: '/en/chapter10_agentic_rl/intro',
          collapsed: false,
          items: [
            {
              text: '10.1 Multi-Turn Interaction',
              link: '/en/chapter10_agentic_rl/multi-turn-rl'
            },
            {
              text: '10.2 Tool Use',
              link: '/en/chapter10_agentic_rl/tool-use-and-trajectory'
            },
            {
              text: '10.3 Benchmarks & Cases',
              link: '/en/chapter10_agentic_rl/industrial-evaluation'
            },
            {
              text: '10.4 Hands-On: Agent Data',
              link: '/en/chapter10_agentic_rl/agent-data-swe-smith'
            },
            {
              text: '10.5 Hands-On: DeepCoder',
              link: '/en/chapter10_agentic_rl/rllm-deepcoder-lab'
            },
            {
              text: '10.6 Hands-On: FinQA Agent',
              link: '/en/chapter10_agentic_rl/rllm-finqa-lab'
            },
            {
              text: '10.7 Deep Research',
              link: '/en/chapter10_agentic_rl/deep-research-agent'
            },
            {
              text: '10.8 Agentic Training Systems',
              link: '/en/chapter10_agentic_rl/build-agentic-training-system'
            },
            {
              text: '10.9 Extended Readings',
              link: '/en/chapter10_agentic_rl/extended-readings'
            }
          ]
        },
        { text: 'Part III Summary', link: '/en/summaries/part3-summary' }
      ]
    },
    {
      text: 'Frontier Topics',
      items: [
        {
          text: '11. VLM Reinforcement Learning',
          link: '/en/chapter11_vlm_rl/intro',
          collapsed: false,
          items: [
            {
              text: '11.1 VLM RL Training',
              link: '/en/chapter11_vlm_rl/vlm-grpo-hands-on'
            },
            {
              text: '11.2 Visual Reward Signals',
              link: '/en/chapter11_vlm_rl/vlm-challenges'
            },
            {
              text: '11.3 VLM RL Frameworks',
              link: '/en/chapter11_vlm_rl/vlm-frameworks'
            },
            {
              text: '11.4 Visual Generation RL',
              link: '/en/chapter11_vlm_rl/visual-generation-rl'
            },
            {
              text: '11.5 Hands-On: GeoQA',
              link: '/en/chapter11_vlm_rl/easyr1-geoqa'
            }
          ]
        },
        {
          text: '12. Future Trends',
          link: '/en/chapter12_future_trends/intro',
          collapsed: false,
          items: [
            {
              text: '12.1 Embodied Intelligence',
              link: '/en/chapter12_future_trends/embodied-intelligence/'
            },
            {
              text: '12.2 Model-Based RL',
              link: '/en/chapter12_future_trends/embodied-intelligence/model-based-rl/'
            },
            {
              text: '12.3 Self-Play',
              link: '/en/chapter12_future_trends/self-play-outlook/'
            },
            {
              text: '12.4 Multi-Agent RL',
              link: '/en/chapter12_future_trends/llm-multi-agent-rl/'
            },
            {
              text: '12.5 Offline RL',
              link: '/en/chapter12_future_trends/offline-rl/'
            },
            {
              text: '12.6 Scaling Trends',
              link: '/en/chapter12_future_trends/rl-scaling-outlook'
            }
          ]
        },
        { text: 'Part IV Summary', link: '/en/summaries/part4-summary' }
      ]
    },
    {
      text: 'Appendices',
      items: [
        {
          text: 'A. Training Debugging Guide',
          link: '/en/appendix_common_pitfalls/intro'
        },
        {
          text: 'B. RL Engineering Practice',
          link: '/en/appendix_industrial_training/intro',
          collapsed: false,
          items: [
            {
              text: 'B.1 Training Infrastructure',
              link: '/en/appendix_industrial_training/rl-infrastructure'
            },
            {
              text: 'B.2 Agent Sandbox',
              link: '/en/appendix_industrial_training/agentic-rl-infra'
            },
            {
              text: 'B.3 Evaluation Benchmarks',
              link: '/en/appendix_industrial_training/evaluation-badcase'
            },
            {
              text: 'B.4 Metrics Glossary',
              link: '/en/appendix_industrial_training/metrics-glossary'
            },
            {
              text: 'B.5 Industrial Exercises',
              link: '/en/appendix_industrial_training/industrial-exercises'
            }
          ]
        },
        {
          text: 'C. Code Cheatsheet',
          link: '/en/appendix_code_cheatsheet/intro',
          collapsed: false,
          items: [
            {
              text: 'C.1 SFT and KL',
              link: '/en/appendix_code_cheatsheet/sft-kl'
            },
            {
              text: 'C.2 PPO and GAE',
              link: '/en/appendix_code_cheatsheet/ppo-gae'
            },
            {
              text: 'C.3 DPO Family',
              link: '/en/appendix_code_cheatsheet/dpo-family'
            },
            {
              text: 'C.4 GRPO and Reward Models',
              link: '/en/appendix_code_cheatsheet/grpo-rlvr'
            },
            {
              text: 'C.5 Softmax & Cross-Entropy',
              link: '/en/appendix_code_cheatsheet/softmax-ce'
            },
            {
              text: 'C.6 Sampling Methods',
              link: '/en/appendix_code_cheatsheet/top-k-top-p'
            },
            {
              text: 'C.7 Attention Mechanism',
              link: '/en/appendix_code_cheatsheet/attention-mha'
            },
            { text: 'C.8 DAPO', link: '/en/appendix_code_cheatsheet/dapo' }
          ]
        },
        {
          text: 'D. Learning Resources',
          link: '/en/appendix_game_projects/intro'
        },
        {
          text: 'E. Math Foundations for RL',
          link: '/en/appendix_math/intro',
          collapsed: false,
          items: [
            {
              text: 'E.1 Linear Algebra',
              link: '/en/appendix_math/linear-algebra',
              collapsed: true,
              items: [
                {
                  text: 'Basic Objects',
                  link: '/en/appendix_math/linear-algebra-basics'
                },
                {
                  text: 'Bellman Matrix',
                  link: '/en/appendix_math/linear-algebra-bellman'
                },
                {
                  text: 'Function Approximation',
                  link: '/en/appendix_math/linear-algebra-function-approx'
                },
                {
                  text: 'Convergence & Trust Regions',
                  link: '/en/appendix_math/linear-algebra-advanced'
                },
                {
                  text: 'Formulas & Exercises',
                  link: '/en/appendix_math/linear-algebra-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.2 Probability & Estimation',
              link: '/en/appendix_math/probability-statistics',
              collapsed: true,
              items: [
                {
                  text: 'Probability Basics',
                  link: '/en/appendix_math/probability-basics'
                },
                {
                  text: 'Returns and Value',
                  link: '/en/appendix_math/probability-value'
                },
                {
                  text: 'Sampling & Estimation',
                  link: '/en/appendix_math/probability-sampling'
                },
                {
                  text: 'Trajectories and GAE',
                  link: '/en/appendix_math/probability-trajectory-td'
                },
                {
                  text: 'Bellman Expectations',
                  link: '/en/appendix_math/probability-bellman-advanced'
                },
                {
                  text: 'Formulas & Exercises',
                  link: '/en/appendix_math/probability-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.3 Calculus & Optimization',
              link: '/en/appendix_math/calculus-optimization',
              collapsed: true,
              items: [
                {
                  text: 'Derivatives & Gradients',
                  link: '/en/appendix_math/calculus-basics'
                },
                {
                  text: 'Policy Gradient',
                  link: '/en/appendix_math/calculus-policy-gradient'
                },
                {
                  text: 'PPO and Adam',
                  link: '/en/appendix_math/calculus-ppo'
                },
                {
                  text: 'Derivation Tools',
                  link: '/en/appendix_math/calculus-derivations'
                },
                {
                  text: 'Complete Formulas',
                  link: '/en/appendix_math/calculus-advanced-formulas'
                },
                {
                  text: 'Formulas & Exercises',
                  link: '/en/appendix_math/calculus-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.4 Information Theory',
              link: '/en/appendix_math/information-theory',
              collapsed: true,
              items: [
                {
                  text: 'Entropy & Exploration',
                  link: '/en/appendix_math/information-basics'
                },
                {
                  text: 'Cross-Entropy & KL',
                  link: '/en/appendix_math/information-cross-entropy-kl'
                },
                {
                  text: 'RLHF and DPO',
                  link: '/en/appendix_math/information-rlhf-dpo'
                },
                {
                  text: 'Mutual Information',
                  link: '/en/appendix_math/information-mutual-info'
                },
                {
                  text: 'Complete Formulas',
                  link: '/en/appendix_math/information-advanced-formulas'
                },
                {
                  text: 'Formulas & Exercises',
                  link: '/en/appendix_math/information-formulas-exercises'
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

const logger = createLogger()
const originalWarn = logger.warn
logger.warn = (msg, options) => {
  if (msg.includes('Failed to resolve "/@siteData"')) return
  originalWarn(msg, options)
}

export default defineConfig({
  lang: 'zh-CN',
  title: 'Hands-on Modern RL',
  description:
    '现代强化学习实战指南：涵盖经典控制、LLM 后训练、RLVR 与多模态智能体',
  base,
  cleanUrls: true,
  lastUpdated: true,
  markdown: {
    image: {
      lazyLoading: true
    },
    attrs: {
      disable: true
    },
    config: (md) => {
      safeHeadingAttrs(md)
      optimizedImagesPlugin(md)
      md.use(markdownItFootnote)
      katexMarkdown(md)
      MermaidMarkdown(md)
      optimizedMermaidPlugin(md)
      // Custom "output" container for displaying code running results
      md.use(markdownItContainer, 'output', {
        render: function (tokens, idx) {
          if (tokens[idx].nesting === 1) {
            const title = tokens[idx].info.trim().slice(6).trim() || '运行结果'
            return `<div class="custom-block output"><p class="custom-block-title">${title}</p>\n`
          }
          return '</div>\n'
        }
      })
    }
  },
  vite: {
    customLogger: logger,
    plugins: [mermaidConfigPlugin(), normalizeBrokenDocPathPlugin()],
    optimizeDeps: {
      include: [
        '@braintree/sanitize-url',
        'cytoscape',
        'cytoscape-cose-bilkent',
        'dayjs',
        'debug'
      ]
    },
    resolve: {
      alias: {
        'dayjs/plugin/advancedFormat.js': 'dayjs/esm/plugin/advancedFormat',
        'dayjs/plugin/customParseFormat.js':
          'dayjs/esm/plugin/customParseFormat',
        'dayjs/plugin/isoWeek.js': 'dayjs/esm/plugin/isoWeek',
        'cytoscape/dist/cytoscape.umd.js': 'cytoscape/dist/cytoscape.esm.js'
      }
    }
  },
  ignoreDeadLinks: true,
  head: [
    ['link', { rel: 'icon', href: `${base}favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#3f51b5' }],
    [
      'meta',
      { name: 'viewport', content: 'width=device-width, initial-scale=1.0' }
    ],
    ['meta', { name: 'author', content: 'WalkingLabs' }],
    ['meta', { name: 'robots', content: 'index,follow' }],
    ['meta', { property: 'og:title', content: 'Hands-on Modern RL' }],
    [
      'meta',
      {
        property: 'og:description',
        content:
          '现代强化学习实战指南：涵盖经典控制、LLM 后训练、RLVR 与多模态智能体'
      }
    ],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:url', content: siteUrl }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }]
  ],
  locales: {
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      title: 'Hands-on Modern RL',
      description: '现代强化学习实战——从代码到原理',
      themeConfig: {
        nav: zhNav,
        sidebar: zhSidebar,
        editLink: {
          pattern: editLinkPattern,
          text: '在 GitHub 上编辑此页'
        },
        footer: {
          message: '现代强化学习实战课程',
          copyright: 'Copyright © WalkingLabs'
        },
        outline: {
          level: [2, 3],
          label: 'Outline'
        },
        lastUpdated: {
          text: '最后更新'
        },
        docFooter: {
          prev: '上一页',
          next: '下一页'
        },
        darkModeSwitchLabel: '外观',
        lightModeSwitchTitle: '切换到浅色模式',
        darkModeSwitchTitle: '切换到深色模式',
        sidebarMenuLabel: '菜单',
        returnToTopLabel: '返回顶部',
        langMenuLabel: '切换语言',
        skipToContentLabel: '跳转到正文',
        notFound: {
          title: '页面未找到',
          quote: '这个地址不存在，试试从中文首页重新进入。',
          link: '/zh/',
          linkText: '返回中文首页',
          linkLabel: '返回中文首页'
        }
      }
    },
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      title: 'Hands-on Modern RL',
      description:
        'Modern Reinforcement Learning in Practice — From Code to Theory',
      themeConfig: {
        nav: enNav,
        sidebar: enSidebar,
        editLink: {
          pattern: editLinkPattern,
          text: 'Edit this page on GitHub'
        },
        footer: {
          message: '现代强化学习实战课程',
          copyright: 'Copyright © WalkingLabs'
        },
        outline: {
          level: [2, 3],
          label: 'Outline'
        },
        lastUpdated: {
          text: 'Last updated'
        },
        docFooter: {
          prev: 'Previous page',
          next: 'Next page'
        },
        darkModeSwitchLabel: 'Appearance',
        lightModeSwitchTitle: 'Switch to light theme',
        darkModeSwitchTitle: 'Switch to dark theme',
        sidebarMenuLabel: 'Menu',
        returnToTopLabel: 'Return to top',
        langMenuLabel: 'Change language',
        skipToContentLabel: 'Skip to content',
        notFound: {
          title: 'Page not found',
          quote:
            'This page is missing. Try jumping back in from the English home page.',
          link: '/en/',
          linkText: 'Take me to English home',
          linkLabel: 'Go to English home'
        }
      }
    }
  },
  themeConfig: {
    logo: '/readme/logo.png',
    siteTitle: 'Hands-on Modern RL',
    nav: zhNav,
    sidebar: zhSidebar,
    socialLinks: [
      { icon: 'github', link: `https://github.com/${owner}/${repo}` }
    ],
    search: enableLocalSearch
      ? {
          provider: 'local',
          options: {
            _render: renderSearchMarkdown
          }
        }
      : undefined,
    editLink: {
      pattern: editLinkPattern,
      text: 'Edit this page on GitHub'
    },
    footer: {
      message: 'Built for reusable bilingual course delivery',
      copyright: 'Copyright © WalkingLabs'
    },
    outline: {
      level: [2, 3],
      label: 'Outline'
    }
  }
})
