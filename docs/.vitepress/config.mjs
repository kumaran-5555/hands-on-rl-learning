/* global process */
import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { createRequire } from 'module'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const require = createRequire(import.meta.url)
const markdownItFootnote = require('markdown-it-footnote')

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const packageJsonPath = path.resolve(__dirname, '../../package.json')
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))

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

const zhNav = [
  { text: '前言与导论', link: '/preface/intro' },
  { text: '基础导论', link: '/chapter01_cartpole/intro' },
  { text: '核心理论与方法', link: '/chapter03_mdp/intro' },
  { text: '大模型强化学习', link: '/chapter07_alignment/intro' },
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
        { text: '写在开头', link: '/preface/intro' },
        { text: '强化学习简史', link: '/preface/brief-history' }
      ]
    },
    {
      text: '第一篇：基础导论',
      items: [
        {
          text: '第1章：RL初印象',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter01_cartpole/intro' },
            { text: '核心原理', link: '/chapter01_cartpole/principles' },
            { text: '训练与指标', link: '/chapter01_cartpole/metrics' }
          ]
        },
        {
          text: '第2章：现代RL初体验',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter02_dpo/intro' },
            { text: '核心原理', link: '/chapter02_dpo/principles' },
            { text: '训练与指标', link: '/chapter02_dpo/metrics' }
          ]
        },
        { text: '第一篇总结', link: '/summaries/part1-summary' }
      ]
    },
    {
      text: '第二篇：核心理论与方法',
      items: [
        {
          text: '第3章：RL 通识——理论地图与全局视野',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter03_mdp/intro' },
            { text: '动手：两台老虎机', link: '/chapter03_mdp/bandit' },
            { text: 'MDP 五元组、折扣回报与策略', link: '/chapter03_mdp/mdp' },
            { text: 'V(s) 与贝尔曼方程', link: '/chapter03_mdp/value-bellman' },
            { text: '经典方法速览：DP、MC 与 TD', link: '/chapter03_mdp/dp-mc-td' },
            { text: '路线一：Q(s,a)', link: '/chapter03_mdp/value-q' },
            { text: '路线二：J(θ)', link: '/chapter03_mdp/policy-objective' },
            { text: '奖励函数设计', link: '/chapter03_mdp/reward-design' },
            { text: '全景地图', link: '/chapter03_mdp/panorama' }
          ]
        },
        {
          text: '第4章：Value-Based 方法：Q-Learning 与深度 Q 网络',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter04_dqn/intro' },
            {
              text: 'Q-Learning：TD 方法 for Q',
              link: '/chapter04_dqn/q-learning'
            },
            {
              text: '从 Q-Learning 到 DQN',
              link: '/chapter04_dqn/from-q-to-dqn'
            },
            { text: 'DQN 三大组件', link: '/chapter04_dqn/dqn-components' },
            {
              text: '动手：DQN 玩 CartPole',
              link: '/chapter04_dqn/cartpole-dqn'
            },
            {
              text: '动手：从像素学玩 Atari',
              link: '/chapter04_dqn/atari-dqn'
            },
            {
              text: '动手：3D 第一人称 ViZDoom',
              link: '/chapter04_dqn/vizdoom-dqn'
            },
            {
              text: '动手：stable-retro 玩宝可梦',
              link: '/chapter04_dqn/retro-pokemon'
            },
            { text: '观察训练过程', link: '/chapter04_dqn/training-analysis' },
            { text: 'DQN 家族与视角迁移', link: '/chapter04_dqn/dqn-family' }
          ]
        },
        {
          text: '第5章：Policy-Based 方法：策略梯度与 REINFORCE',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter05_policy_gradient/intro' },
            {
              text: '动手：摇骰子赌博机',
              link: '/chapter05_policy_gradient/dice-game'
            },
            {
              text: '策略梯度定理与 REINFORCE',
              link: '/chapter05_policy_gradient/policy-gradient'
            },
            {
              text: '基线实验',
              link: '/chapter05_policy_gradient/baseline-experiment'
            }
          ]
        },
        {
          text: '第6章：Actor-Critic 架构',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter06_actor_critic/intro' },
            {
              text: '优势函数',
              link: '/chapter06_actor_critic/advantage-function'
            },
            {
              text: 'Critic 训练方法',
              link: '/chapter06_actor_critic/critic-training'
            },
            {
              text: 'Actor-Critic 架构',
              link: '/chapter06_actor_critic/actor-critic'
            },
            {
              text: '动手：AlphaGo 简单复现',
              link: '/chapter06_actor_critic/alphago'
            }
          ]
        },
        {
          text: '第7章：PPO——稳定训练的艺术',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter06_ppo/intro' },
            {
              text: '动手：PPO 训练 LunarLander',
              link: '/chapter06_ppo/ppo-lunar-lander'
            },
            { text: 'PPO 数学推导', link: '/chapter06_ppo/ppo-math' },
            {
              text: '信任域与裁剪机制',
              link: '/chapter06_ppo/trust-region-clipping'
            },
            {
              text: 'GAE、奖励模型与 LLM 对齐',
              link: '/chapter06_ppo/gae-reward-model'
            }
          ]
        },
        { text: '第二篇总结', link: '/summaries/part2-summary' }
      ]
    },
    {
      text: '第三篇：大模型强化学习',
      items: [
        {
          text: '第7章：RLHF 完整流水线',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter10_rlhf/intro' },
            {
              text: '动手：亲手制造一场 Reward Hacking',
              link: '/chapter10_rlhf/reward-hacking-hands-on'
            },
            {
              text: '模仿学习与数据工程',
              link: '/chapter10_rlhf/imitation-learning-pipeline'
            },
            {
              text: '奖励函数设计',
              link: '/chapter10_rlhf/reward-function-design'
            },
            {
              text: '训练稳定性与奖励黑客',
              link: '/chapter10_rlhf/training-stability-hacking'
            },
            {
              text: '自我博弈与数据飞轮',
              link: '/chapter10_rlhf/rlaif-and-data-cycle'
            }
          ]
        },
        {
          text: '第8章：对齐与推理强化（DPO / GRPO / RLVR）',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter07_alignment/intro' },
            {
              text: '动手：DPO 对齐实验',
              link: '/chapter07_alignment/dpo-hands-on'
            },
            {
              text: 'DPO 原理、数学与选型',
              link: '/chapter07_alignment/dpo-theory-and-family'
            },
            {
              text: '动手：GRPO 训练与核心机制',
              link: '/chapter08_grpo_rlvr/grpo-practice-and-mechanism'
            },
            {
              text: 'DeepSeek、DAPO 与 RLVR',
              link: '/chapter08_grpo_rlvr/deepseek-dapo-rlvr'
            },
            {
              text: 'RL Scaling 与前沿展望',
              link: '/chapter08_grpo_rlvr/rl-scaling-outlook'
            },
            {
              text: '知识蒸馏——从大模型到小模型的知识迁移',
              link: '/chapter08_grpo_rlvr/on-policy-distillation'
            }
          ]
        },
        {
          text: '第9章：Agentic RL——工具调用、多轮交互与智能体训练',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter12_agentic_rl/intro' },
            {
              text: '动手：Mini Agent Loop——ORM vs PRM 对比',
              link: '/chapter12_agentic_rl/agent-loop-hands-on'
            },
            {
              text: '多轮交互 RL 与信用分配',
              link: '/chapter12_agentic_rl/multi-turn-rl'
            },
            {
              text: '轨迹合成与数据工程',
              link: '/chapter12_agentic_rl/trajectory-synthesis'
            },
            {
              text: '工具调用 RL：Web Agent 与 Code Agent',
              link: '/chapter12_agentic_rl/tool-use-agents'
            },
            {
              text: 'Agentic RL 工程实战与总结',
              link: '/chapter12_agentic_rl/agentic-engineering'
            },
            {
              text: '工业界实战：各家的 Agentic RL 都怎么做的？',
              link: '/chapter12_agentic_rl/industrial-practice'
            },
            {
              text: 'Agentic 评测体系与 Benchmark 全景',
              link: '/chapter12_agentic_rl/evaluation-benchmarks'
            },
            {
              text: '深度研究智能体：Deep Research Agent',
              link: '/chapter12_agentic_rl/deep-research-agent'
            },
            {
              text: '延伸阅读索引',
              link: '/chapter12_agentic_rl/extended-readings'
            }
          ]
        },
        { text: '第三篇总结', link: '/summaries/part3-summary' }
      ]
    },
    {
      text: '第四篇：前沿与进阶专题',
      items: [
        {
          text: '第10章：VLM 强化学习',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter11_vlm_rl/intro' },
            {
              text: '动手：GRPO 训练 VLM',
              link: '/chapter11_vlm_rl/vlm-grpo-hands-on'
            },
            {
              text: 'VLM RL 的特殊挑战',
              link: '/chapter11_vlm_rl/vlm-challenges'
            },
            {
              text: 'VLM RL 框架与前沿',
              link: '/chapter11_vlm_rl/vlm-frameworks'
            },
            {
              text: '视觉生成模型的 RL 后训练',
              link: '/chapter11_vlm_rl/visual-generation-rl'
            }
          ]
        },
        {
          text: '第11章：连续动作算法与具身智能',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter09_continuous_control/intro' },
            {
              text: '动手：PyBullet 机器人仿真',
              link: '/chapter09_continuous_control/pybullet-hands-on'
            },
            {
              text: '连续策略与 DDPG/TD3',
              link: '/chapter09_continuous_control/continuous-policy-ddpg-td3'
            },
            {
              text: 'SAC、算法对比与并行采样',
              link: '/chapter09_continuous_control/sac-comparison'
            },
            {
              text: 'HER：把失败变成成功',
              link: '/chapter09_continuous_control/her-sparse-reward'
            },
            {
              text: '扩散策略：生成式连续控制',
              link: '/chapter09_continuous_control/diffusion-policy'
            },
            {
              text: '从仿真到现实：具身智能',
              link: '/chapter09_continuous_control/embodied-intelligence'
            }
          ]
        },
        {
          text: '第12章：未来趋势',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/chapter13_future_trends/intro' },
            {
              text: '测试时计算与 RL 推理',
              link: '/chapter13_future_trends/test-time-reasoning'
            },
            {
              text: '多智能体 RL 与基于模型的 RL',
              link: '/chapter13_future_trends/marl-model-based'
            },
            {
              text: '自博弈、自进化与学习路线',
              link: '/chapter13_future_trends/self-play-outlook'
            },
            {
              text: 'LLM 多智能体强化学习',
              link: '/chapter13_future_trends/llm-multi-agent-rl'
            },
            {
              text: '离线强化学习（CQL / IQL / DT）',
              link: '/chapter13_future_trends/offline-rl'
            },
            {
              text: '动手：PettingZoo 多智能体',
              link: '/chapter13_future_trends/pettingzoo'
            }
          ]
        },
        { text: '第四篇总结', link: '/summaries/part4-summary' }
      ]
    },
    {
      text: '附录',
      items: [
        {
          text: '附录A：强化学习训练调试指南',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/appendix_common_pitfalls/intro' },
            {
              text: '策略崩溃与奖励投机',
              link: '/appendix_common_pitfalls/policy-collapse-reward-hacking'
            },
            {
              text: '资源溢出与收敛失效',
              link: '/appendix_common_pitfalls/oom-nonconvergence'
            }
          ]
        },
        {
          text: '附录B：RL 工程实践指南',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/appendix_industrial_training/intro' },
            {
              text: 'B.1 RL 采样基础设施',
              link: '/appendix_industrial_training/rl-infrastructure'
            },
            {
              text: 'B.2 异步训练架构',
              link: '/appendix_industrial_training/async-training'
            },
            {
              text: 'B.3 分布式并行策略',
              link: '/appendix_industrial_training/parallelism'
            },
            {
              text: 'B.4 Agentic RL 基础设施',
              link: '/appendix_industrial_training/agentic-rl-infra'
            },
            {
              text: 'B.5 评测体系与 Badcase 分析',
              link: '/appendix_industrial_training/evaluation-badcase'
            },
            {
              text: 'B.6 训练监控与故障排查',
              link: '/appendix_industrial_training/monitoring'
            },
            {
              text: 'B.7 工业实战练习',
              link: '/appendix_industrial_training/industrial-exercises'
            },
            {
              text: 'B.8 大模型 RL 训练指标词典',
              link: '/appendix_industrial_training/metrics-glossary'
            }
          ]
        },
        {
          text: '附录C：算法选型与工程框架',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/appendix_algorithm_guide/intro' },
            {
              text: '算法选型决策',
              link: '/appendix_algorithm_guide/algorithm-selection'
            },
            {
              text: '训练框架与模型基方法',
              link: '/appendix_algorithm_guide/framework-mbrl'
            }
          ]
        },
        {
          text: '附录D：强化学习经典项目',
          link: '/appendix_game_projects/intro'
        },
        {
          text: '附录E：数学基础',
          collapsed: false,
          items: [
            { text: '章节导览', link: '/appendix_math/intro' },
            { text: 'E.1 线性代数', link: '/appendix_math/linear-algebra' },
            {
              text: 'E.2 概率与统计',
              link: '/appendix_math/probability-statistics'
            },
            {
              text: 'E.3 微积分与优化',
              link: '/appendix_math/calculus-optimization'
            },
            { text: 'E.4 信息论', link: '/appendix_math/information-theory' }
          ]
        },
        { text: '附录G：术语对照表', link: '/appendix_terminology/intro' },
        { text: '附录H：环境安装指南', link: '/appendix_env_install/intro' }
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

export default withMermaid(
  defineConfig({
    lang: 'zh-CN',
    title: 'Course Template',
    description: 'WalkingLabs course template powered by VitePress',
    base,
    cleanUrls: true,
    lastUpdated: true,
    markdown: {
      math: true,
      config: (md) => {
        md.use(markdownItFootnote)

        // Workaround: markdown-it-mathjax3 may inject <style> tags that
        // Vue's DOM compiler rejects in dev mode.  Strip them from the
        // rendered output so the page compiles cleanly.
        const stripStyles = (html) =>
          html.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
        const origInline = md.renderer.rules.math_inline
        const origBlock = md.renderer.rules.math_block
        if (origInline) {
          md.renderer.rules.math_inline = function (...args) {
            return stripStyles(origInline.apply(this, args))
          }
        }
        if (origBlock) {
          md.renderer.rules.math_block = function (...args) {
            return stripStyles(origBlock.apply(this, args))
          }
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
      ['meta', { property: 'og:title', content: 'Course Template' }],
      [
        'meta',
        {
          property: 'og:description',
          content:
            'A reusable bilingual documentation and deployment template for future WalkingLabs courses'
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
        title: 'Course Template',
        description: 'WalkingLabs 双语课程仓库模板',
        themeConfig: {
          nav: zhNav,
          sidebar: zhSidebar,
          editLink: {
            pattern: editLinkPattern,
            text: '在 GitHub 上编辑此页'
          },
          footer: {
            message: '为可复用的双语课程交付而构建',
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
        title: 'Course Template',
        description: 'WalkingLabs bilingual course repository template',
        themeConfig: {
          nav: enNav,
          sidebar: enSidebar,
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
      logo: '/logo.svg',
      siteTitle: 'Hands-on Modern RL',
      nav: zhNav,
      sidebar: zhSidebar,
      socialLinks: [
        { icon: 'github', link: `https://github.com/${owner}/${repo}` }
      ],
      search: {
        provider: 'local'
      },
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
)
