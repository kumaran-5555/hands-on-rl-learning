/* global process */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const __filename = fileURLToPath(import.meta.url)

export const rootDir = path.resolve(path.dirname(__filename), '..')
export const docsDir = path.join(rootDir, 'docs')
export const distDir = path.join(docsDir, '.vitepress', 'dist')
export const configPath = path.join(docsDir, '.vitepress', 'config.mjs')
export const packageJsonPath = path.join(rootDir, 'package.json')
export const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))
export const logoPath = '../public/readme/readmelogo.png'

// --- Book metadata ---

const sanbuGithubUrl =
  process.env.PDF_SANBU_GITHUB_URL || 'https://github.com/sanbuphy'

export const bookVersion =
  process.env.PDF_VERSION || '0.1.0'

export const bookAuthor =
  process.env.PDF_AUTHORS ||
  process.env.PDF_AUTHOR ||
  packageJson.author ||
  `WalkingLabs; 散步 (${sanbuGithubUrl})`

export const bookTitle = '现代强化学习实战——从代码到原理'

export const bookTagline =
  '现代强化学习实战指南：涵盖经典控制、LLM 后训练、RLVR 与多模态智能体'

export const bookLicense = 'CC-BY-NC-SA-4.0'

export const bookBuildDate =
  process.env.PDF_BUILD_DATE ||
  new Intl.DateTimeFormat('zh-CN', {
    dateStyle: 'long',
    timeZone: 'Asia/Shanghai'
  }).format(new Date())

export function repositoryUrl() {
  if (process.env.PDF_GITHUB_URL) return process.env.PDF_GITHUB_URL

  const githubRepository = process.env.GITHUB_REPOSITORY
  if (githubRepository && githubRepository.includes('/')) {
    return `https://github.com/${githubRepository.replace(/\.git$/, '')}`
  }

  const repository =
    typeof packageJson.repository === 'string'
      ? packageJson.repository
      : packageJson.repository?.url || ''

  const sshMatch = repository.match(/github\.com:(.+?)\/(.+?)(\.git)?$/)
  if (sshMatch) {
    return `https://github.com/${sshMatch[1]}/${sshMatch[2]}`
  }

  const normalized = repository
    .replace(/^git\+/, '')
    .replace(/^https?:\/\/github\.com\//, '')
    .replace(/^git@github\.com:/, '')
    .replace(/\.git$/, '')

  if (normalized.includes('/')) {
    return `https://github.com/${normalized}`
  }

  return 'https://github.com/walkinglabs/hands-on-modern-rl'
}

export const bookGithubUrl = repositoryUrl()

// --- Utility functions ---

export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

export function stripFrontmatter(value) {
  if (!value.startsWith('---\n')) return value
  const end = value.indexOf('\n---\n', 4)
  if (end === -1) return value
  return value.slice(end + 5)
}

export function stripScriptSetup(value) {
  return value.replace(/^\s*<script\b[\s\S]*?<\/script>\s*/i, '')
}

export function flattenSidebar(items, pages = [], seen = new Set()) {
  for (const item of items || []) {
    if (item.link && !item.link.includes('#') && !seen.has(item.link)) {
      seen.add(item.link)
      pages.push({ title: item.text || item.link, link: item.link })
    }
    if (item.items) {
      flattenSidebar(item.items, pages, seen)
    }
  }
  return pages
}

export function splitSuffix(value) {
  const hashIndex = value.indexOf('#')
  const queryIndex = value.indexOf('?')
  const suffixIndexes = [hashIndex, queryIndex].filter((index) => index >= 0)
  const suffixIndex = suffixIndexes.length ? Math.min(...suffixIndexes) : -1

  if (suffixIndex === -1) {
    return { clean: value, suffix: '' }
  }

  return {
    clean: value.slice(0, suffixIndex),
    suffix: value.slice(suffixIndex)
  }
}

export function pagePathForLink(link) {
  const { clean } = splitSuffix(link.replace(/^\//, ''))
  const normalized = clean.replace(/\/$/, '')
  const candidates = [
    path.join(docsDir, `${normalized}.md`),
    path.join(docsDir, normalized, 'index.md')
  ]

  return candidates.find((candidate) => fs.existsSync(candidate)) || null
}

export async function loadVitePressConfig() {
  const module = await import(pathToFileURL(configPath).href)
  return module.default
}

export function loadSidebar(config) {
  return (
    config.locales?.zh?.themeConfig?.sidebar?.['/'] ||
    config.themeConfig?.sidebar?.['/'] ||
    []
  )
}

// --- Front matter content (structured data) ---

export const frontMatter = {
  cover: {
    kicker: 'Open Textbook · Modern Reinforcement Learning',
    titleEn: 'Hands-on Modern RL',
    titleZh: bookTitle,
    description: `${bookTagline}。本书从可运行代码、训练曲线和失败案例出发，逐步进入 MDP、策略梯度、PPO、DPO、GRPO、RLVR、Agentic RL 与多模态强化学习。`,
    publisher: 'WalkingLabs Open Course Series'
  },
  intro: {
    title: '本书简介',
    paragraphs: [
      '<strong>Hands-on Modern RL</strong> 是一本面向现代强化学习实践的开放电子书。它不是把网页机械打印成 PDF，而是把课程内容整理成可离线阅读、可引用、可长期迭代的书籍版本。',
      '本书采用"实践优先"的路径：先从一行行可运行代码、训练曲线、指标诊断和失败现象出发，让读者看到智能体如何在环境中试错、从奖励中改进行为，再回到状态、价值函数、贝尔曼方程、策略梯度、奖励建模和信用分配等核心概念。',
      '内容从 CartPole、DQN、Actor-Critic 与 PPO 进入 LLM 后训练，继续覆盖 RLHF、DPO、GRPO、RLVR、工具调用智能体、Deep Research、VLM 强化学习、具身智能和多智能体自我博弈等现代主题。'
    ],
    audience: {
      title: '本书适合谁',
      items: [
        '从监督学习转向强化学习的机器学习工程师；',
        '准备阅读现代强化学习、LLM 对齐和 Agentic RL 论文的研究人员与学生；',
        '希望理解 RLHF、DPO、GRPO、RLVR 与后训练系统的大模型从业者；',
        '喜欢先看代码、实验和可视化，再进入公式推导的自主学习者。'
      ]
    },
    howToRead: {
      title: '如何阅读',
      paragraphs: [
        '如果你是第一次接触强化学习，建议按章节顺序阅读；如果你已经熟悉经典 RL，可以从 PPO、DPO、GRPO 与 Agentic RL 章节开始，并在需要时回到前面的数学和算法章节补齐概念。'
      ]
    }
  },
  publicationNote: {
    title: '版本说明',
    iterationNote:
      '这是一本随开源课程持续迭代的电子书。v0.1.0 版本已经包含完整目录、章节开篇页、局部参考文献、PDF 书签、封面和前置说明；后续版本会继续补齐更精细的版式、索引、术语表、练习答案与纸质书级别的排版细节。',
    disclaimer:
      '正文中如遇到仍处在快速迭代的章节，请优先参考最新 GitHub 仓库。欢迎通过 issue、pull request 或课程讨论渠道修正错误、补充案例和完善练习。',
    licenseSection: {
      title: '授权与引用',
      text: '本书遵循仓库中的开源协议发布，适合学习、教学和非商业传播。引用本书时，请注明书名、版本、作者与 GitHub 仓库地址。',
    }
  }
}
