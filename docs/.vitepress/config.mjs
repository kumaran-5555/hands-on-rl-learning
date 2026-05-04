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
    if (match >= max) { match = -1; break }
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
      const lastText = [...children].reverse().find((token) => token.type === 'text')
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
  return /^(?:[a-z][a-z0-9+.-]*:)?\/\//i.test(value) || value.startsWith('data:')
}

function resolveMarkdownAsset(relativePagePath, src) {
  if (!src || isExternalAsset(src) || src.startsWith('/')) return null

  const hashIndex = src.indexOf('#')
  const queryIndex = src.indexOf('?')
  const suffixIndexCandidates = [hashIndex, queryIndex].filter((idx) => idx >= 0)
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

        return `<p class="mermaid-static"><img src="${src}" alt="Mermaid diagram" data-source-src="/src/${source}" data-source-page="/src/${page}" data-source-index="${block.index}" loading="lazy" decoding="async"></p>\n`
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
  { text: 'Start', link: '/en/guide/getting-started' },
  {
    text: 'Demo Course',
    items: [
      { text: 'Course Tour', link: '/en/demo/' },
      { text: '01 Positioning and Map', link: '/en/demo/chapter-01/' },
      { text: '02 Content and Practice', link: '/en/demo/chapter-02/' },
      { text: '03 Release and Delivery', link: '/en/demo/chapter-03/' }
    ]
  },
  { text: 'Deployment', link: '/en/deployment/' }
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
              text: '1.1 状态、动作、奖励与策略',
              link: '/chapter01_cartpole/principles'
            },
            {
              text: '1.2 奖励、熵、Value Loss 与 KL',
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
              text: '2.1 Post-Training 流水线与 DPO 推导',
              link: '/chapter02_dpo/principles'
            },
            {
              text: '2.2 Loss、Reward Margin 与 Accuracy',
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
            { text: '3.1 两台老虎机：RL 的最小问题', link: '/chapter03_mdp/bandit' },
            { text: '3.2 MDP：RL 的形式化框架', link: '/chapter03_mdp/mdp' },
            {
              text: '3.3 V(s) 与贝尔曼方程',
              link: '/chapter03_mdp/value-bellman'
            },
            {
              text: '3.4 DP、MC、TD',
              link: '/chapter03_mdp/dp-mc-td'
            },
            { text: '3.5 Q(s, a)', link: '/chapter03_mdp/value-q' },
            {
              text: '3.6 策略目标 J(theta)',
              link: '/chapter03_mdp/policy-objective'
            },
            {
              text: '3.7 算法数据来源',
              link: '/chapter03_mdp/algorithm-taxonomy'
            },
            { text: '3.8 Reward Shaping', link: '/chapter03_mdp/reward-design' },
            { text: '3.9 本章总结', link: '/chapter03_mdp/panorama' }
          ]
        },
        {
          text: '4. Q-Learning 与 DQN',
          link: '/chapter04_dqn/intro',
          collapsed: false,
          items: [
            {
              text: '4.1 动手：Q-Learning 与 GridWorld',
              link: '/chapter04_dqn/q-learning'
            },
            {
              text: '4.2 从表格 Q 到 DQN',
              link: '/chapter04_dqn/from-q-to-dqn'
            },
            { text: '4.3 Replay、Target 与 CNN', link: '/chapter04_dqn/dqn-components' },
            {
              text: '4.4 训练过程分析',
              link: '/chapter04_dqn/training-analysis'
            },
            {
              text: '4.5 Mountain Car 与稀疏奖励',
              link: '/chapter04_dqn/mountain-car'
            },
            { text: '4.6 Double、Dueling 与 Rainbow', link: '/chapter04_dqn/dqn-family' },
            {
              text: '4.7 项目：DQN 实战与视觉游戏',
              link: '/chapter04_dqn/visual-game-projects'
            }
          ]
        },
        {
          text: '5. 策略梯度与 REINFORCE',
          link: '/chapter05_policy_gradient/intro',
          collapsed: false,
          items: [
            {
              text: '5.1 动手：摇骰子赌博机',
              link: '/chapter05_policy_gradient/dice-game'
            },
            {
              text: '5.2 策略梯度与 REINFORCE',
              link: '/chapter05_policy_gradient/policy-gradient'
            },
            {
              text: '5.3 动手：Baseline 降方差',
              link: '/chapter05_policy_gradient/baseline-experiment'
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
              text: '6.2 TD 误差训练 Critic',
              link: '/chapter06_actor_critic/critic-training'
            },
            {
              text: '6.3 Actor-Critic 架构',
              link: '/chapter06_actor_critic/actor-critic'
            },
            {
              text: '6.4 项目：AlphaGo 简单复现',
              link: '/chapter06_actor_critic/alphago'
            }
          ]
        },
        {
          text: '7. PPO',
          link: '/chapter07_ppo/intro',
          collapsed: false,
          items: [
            {
              text: '7.1 动手：PPO 训练 LunarLander',
              link: '/chapter07_ppo/ppo-lunar-lander'
            },
            {
              text: '7.2 PPO 数学推导',
              link: '/chapter07_ppo/ppo-math'
            },
            {
              text: '7.3 信任域与裁剪',
              link: '/chapter07_ppo/trust-region-clipping'
            },
            {
              text: '7.4 GAE 与奖励模型',
              link: '/chapter07_ppo/gae-reward-model'
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
              text: '8.1 为什么 base model 还不是 assistant',
              link: '/chapter08_rlhf/base-model-to-assistant'
            },
            {
              text: '8.2 标准 RLHF 流水线',
              link: '/chapter08_rlhf/standard-rlhf-pipeline'
            },
            {
              text: '8.3 SFT：教模型按指令回答',
              link: '/chapter08_rlhf/imitation-learning-pipeline'
            },
            {
              text: '8.4 Reward Model：教一个裁判',
              link: '/chapter08_rlhf/reward-function-design'
            },
            {
              text: '8.5 PPO-RLHF：按奖励练习',
              link: '/chapter08_rlhf/ppo-rlhf-loop'
            },
            {
              text: '8.6 评估：RLHF 到底有没有变好',
              link: '/chapter08_rlhf/evaluation'
            },
            {
              text: '8.7 从小参数到大参数',
              link: '/chapter08_rlhf/scaling-to-large-models'
            },
            {
              text: '8.8 扩展实战：Reward Hacking 与数据飞轮',
              link: '/chapter08_rlhf/extended-practice'
            }
          ]
        },
        {
          text: '9. 后训练对齐',
          link: '/chapter09_alignment/intro',
          collapsed: false,
          items: [
            {
              text: '9.1 DPO、IPO 与 KTO',
              link: '/chapter09_alignment/dpo-theory-and-family'
            },
            {
              text: '9.2 动手：DPO 对齐实验',
              link: '/chapter09_alignment/dpo-hands-on'
            },
            {
              text: '9.3 GRPO 实践与机制',
              link: '/chapter09_grpo_rlvr/grpo-practice-and-mechanism'
            },
            {
              text: '9.4 DeepSeek-R1 与 DAPO',
              link: '/chapter09_grpo_rlvr/deepseek-dapo'
            },
            {
              text: '9.5 RLVR：可验证奖励',
              link: '/chapter09_grpo_rlvr/rlvr'
            },
            {
              text: '9.6 On-Policy Distillation',
              link: '/chapter09_grpo_rlvr/on-policy-distillation'
            }
          ]
        },
        {
          text: '10. Agentic RL',
          link: '/chapter10_agentic_rl/intro',
          collapsed: false,
          items: [
            {
              text: '10.1 多轮交互与信用分配（含 ORM vs PRM 实验）',
              link: '/chapter10_agentic_rl/multi-turn-rl'
            },
            {
              text: '10.2 工具调用、轨迹合成与 Agentic 工程',
              link: '/chapter10_agentic_rl/tool-use-and-trajectory'
            },
            {
              text: '10.3 工业实践、评测与 Badcase',
              link: '/chapter10_agentic_rl/industrial-evaluation'
            },
            {
              text: '10.4 项目一：多工具 Code Agent',
              link: '/chapter10_agentic_rl/multi-tool-code-agent'
            },
            {
              text: '10.5 项目二：Deep Research Agent',
              link: '/chapter10_agentic_rl/deep-research-agent'
            },
            {
              text: '10.6 延伸阅读',
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
              text: '11.1 动手：GRPO 训练 VLM',
              link: '/chapter11_vlm_rl/vlm-grpo-hands-on'
            },
            {
              text: '11.2 视觉奖励与幻觉',
              link: '/chapter11_vlm_rl/vlm-challenges'
            },
            {
              text: '11.3 Open-R1、R1-V 与 VLM-R1',
              link: '/chapter11_vlm_rl/vlm-frameworks'
            },
            {
              text: '11.4 视觉生成 RL',
              link: '/chapter11_vlm_rl/visual-generation-rl'
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
              text: '12.2 Model-Based RL',
              link: '/chapter12_future_trends/embodied-intelligence/model-based-rl/'
            },
            {
              text: '12.3 Self-Play 与自进化',
              link: '/chapter12_future_trends/self-play-outlook/'
            },
            {
              text: '12.4 LLM 多智能体 RL',
              link: '/chapter12_future_trends/llm-multi-agent-rl/'
            },
            {
              text: '12.5 离线强化学习',
              link: '/chapter12_future_trends/offline-rl/'
            },
            {
              text: '12.6 RL Scaling 展望',
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
              text: 'B.2 Agent 沙箱与工具调度',
              link: '/appendix_industrial_training/agentic-rl-infra'
            },
            {
              text: 'B.3 RL 与 Agent Benchmark',
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
                { text: '基础对象', link: '/appendix_math/linear-algebra-basics' },
                { text: '贝尔曼矩阵', link: '/appendix_math/linear-algebra-bellman' },
                { text: '函数近似', link: '/appendix_math/linear-algebra-function-approx' },
                { text: '收敛与信任域', link: '/appendix_math/linear-algebra-advanced' },
                { text: '公式与练习', link: '/appendix_math/linear-algebra-formulas-exercises' }
              ]
            },
            {
              text: 'E.2 概率、期望与随机估计',
              link: '/appendix_math/probability-statistics',
              collapsed: true,
              items: [
                { text: '概率基础', link: '/appendix_math/probability-basics' },
                { text: '回报与价值', link: '/appendix_math/probability-value' },
                { text: '采样估计', link: '/appendix_math/probability-sampling' },
                { text: '轨迹与 GAE', link: '/appendix_math/probability-trajectory-td' },
                { text: '贝尔曼期望', link: '/appendix_math/probability-bellman-advanced' },
                { text: '公式与练习', link: '/appendix_math/probability-formulas-exercises' }
              ]
            },
            {
              text: 'E.3 微积分与优化',
              link: '/appendix_math/calculus-optimization',
              collapsed: true,
              items: [
                { text: '导数与梯度', link: '/appendix_math/calculus-basics' },
                { text: '策略梯度', link: '/appendix_math/calculus-policy-gradient' },
                { text: 'PPO 与 Adam', link: '/appendix_math/calculus-ppo' },
                { text: '推导工具', link: '/appendix_math/calculus-derivations' },
                { text: '完整公式', link: '/appendix_math/calculus-advanced-formulas' },
                { text: '公式与练习', link: '/appendix_math/calculus-formulas-exercises' }
              ]
            },
            {
              text: 'E.4 信息论与分布距离',
              link: '/appendix_math/information-theory',
              collapsed: true,
              items: [
                { text: '熵与探索', link: '/appendix_math/information-basics' },
                { text: '交叉熵与 KL', link: '/appendix_math/information-cross-entropy-kl' },
                { text: 'RLHF 与 DPO', link: '/appendix_math/information-rlhf-dpo' },
                { text: '互信息', link: '/appendix_math/information-mutual-info' },
                { text: '完整公式', link: '/appendix_math/information-advanced-formulas' },
                { text: '公式与练习', link: '/appendix_math/information-formulas-exercises' }
              ]
            }
          ]
        }
      ]
    }
  ]
}

const enSidebar = {
  '/en/guide/': [
    {
      text: 'Template Guide',
      items: [
        { text: 'Getting Started', link: '/en/guide/getting-started' },
        { text: 'Course Structure', link: '/en/guide/course-structure' },
        { text: 'Project Setup Checklist', link: '/en/guide/project-setup' },
        { text: 'Deployment Guide', link: '/en/deployment/' }
      ]
    }
  ],
  '/en/demo/': [
    {
      text: 'Demo Course Tour',
      items: [
        { text: 'Overview', link: '/en/demo/' },
        { text: 'Learning Path', link: '/en/demo/#learning-path' },
        { text: 'Deliverables', link: '/en/demo/#stage-deliverables' }
      ]
    },
    {
      text: 'Part I. Shape the Course',
      collapsed: false,
      items: [
        {
          text: 'Chapter 01 Positioning and Map',
          link: '/en/demo/chapter-01/'
        },
        {
          text: 'Chapter 01 Advanced Notes',
          link: '/en/demo/chapter-01/advanced'
        }
      ]
    },
    {
      text: 'Part II. Write Content and Practice',
      collapsed: false,
      items: [
        {
          text: 'Chapter 02 Content and Practice',
          link: '/en/demo/chapter-02/'
        },
        {
          text: 'Chapter 02 Homework Brief',
          link: '/en/demo/chapter-02/homework/'
        }
      ]
    },
    {
      text: 'Part III. Release and Delivery',
      collapsed: false,
      items: [
        {
          text: 'Chapter 03 Release and Delivery',
          link: '/en/demo/chapter-03/'
        }
      ]
    }
  ],
  '/en/': [
    {
      text: 'Start Here',
      items: [
        { text: 'Home', link: '/en/' },
        { text: 'Getting Started', link: '/en/guide/getting-started' },
        { text: 'Demo Course', link: '/en/demo/' },
        { text: 'Deployment', link: '/en/deployment/' }
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
    description: '现代强化学习实战指南：涵盖经典控制、LLM 后训练、RLVR 与多模态智能体',
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
          'dayjs/plugin/advancedFormat.js':
            'dayjs/esm/plugin/advancedFormat',
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
          content: '现代强化学习实战指南：涵盖经典控制、LLM 后训练、RLVR 与多模态智能体'
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
