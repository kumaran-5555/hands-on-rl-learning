<script setup>
import DefaultTheme from 'vitepress/theme'
import { useData, useRoute, useRouter, withBase } from 'vitepress'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  PopoverContent,
  PopoverPortal,
  PopoverRoot,
  PopoverTrigger
} from 'reka-ui'
import { HandHeart, Moon, Settings, Sun } from 'lucide-vue-next'
import ReadingProgress from './components/ReadingProgress.vue'
import SidebarFooter from './components/SidebarFooter.vue'
import TextType from './components/TextType.vue'
import mediumZoom from 'medium-zoom'
import { initGithubStars } from './githubStars.js'

const { frontmatter, site, theme, isDark } = useData()
const route = useRoute()
const router = useRouter()

const FONT_SIZE_STORAGE_KEY = 'ct-doc-font-size'
const LINE_HEIGHT_STORAGE_KEY = 'ct-doc-line-height'
const SIDEBAR_COLLAPSED_KEY = 'ct-sidebar-collapsed'
const SIDEBAR_WIDTH_KEY = 'ct-sidebar-width-compact-v3'

const MIN_FONT_SIZE = 15
const MAX_FONT_SIZE = 20
const DEFAULT_FONT_SIZE = 16
const MIN_LINE_HEIGHT = 1.55
const MAX_LINE_HEIGHT = 2
const DEFAULT_LINE_HEIGHT = 1.75

const DEFAULT_SIDEBAR_WIDTH = 212
const MIN_SIDEBAR_WIDTH = 160
const MAX_SIDEBAR_WIDTH = 520

const fontSize = ref(DEFAULT_FONT_SIZE)
const lineHeight = ref(DEFAULT_LINE_HEIGHT)
const readingToolsOpen = ref(false)
const supportOpen = ref(false)
const sidebarCollapsed = ref(false)
const sidebarWidth = ref(DEFAULT_SIDEBAR_WIDTH)
const sidebarResizing = ref(false)
const routeLoading = ref(false)

const mermaidViewerOpen = ref(false)
const mermaidViewerSrc = ref('')
const mermaidViewerAlt = ref('')
const mermaidViewerScale = ref(1)
const mermaidViewerScroll = ref(null)
const mermaidViewerNaturalWidth = ref(0)
const mermaidViewerNaturalHeight = ref(0)
const mermaidViewerCustomZoom = ref(false)
const mermaidViewerDragging = ref(false)

const isHomePage = computed(() => frontmatter.value.layout === 'home')
const showDocChrome = computed(() => !isHomePage.value)
const mobileRoutePath = computed(() => {
  const base = site.value.base || '/'
  if (base === '/') return route.path

  const normalizedBase = base.replace(/\/$/, '')
  const pathWithoutBase = route.path.replace(normalizedBase, '')
  return pathWithoutBase || '/'
})
const isEnglishRoute = computed(
  () =>
    mobileRoutePath.value === '/en' || mobileRoutePath.value.startsWith('/en/')
)
const mobileCurrentLanguage = computed(() =>
  isEnglishRoute.value ? 'English' : '简体中文'
)
const mobileAlternateLanguage = computed(() =>
  isEnglishRoute.value ? '简体中文' : 'English'
)
const mobileAlternateLanguageLink = computed(() => {
  if (isEnglishRoute.value) {
    const zhPath = mobileRoutePath.value.replace(/^\/en(?=\/|$)/, '') || '/'
    return withBase(zhPath)
  }

  const enPath =
    mobileRoutePath.value === '/' ? '/en/' : `/en${mobileRoutePath.value}`
  return withBase(enPath)
})
const supportButtonLabel = computed(() =>
  isEnglishRoute.value ? 'Give the creator a like' : '给制作者一个赞吧'
)
const settingsButtonLabel = computed(() =>
  isEnglishRoute.value ? 'Reading and appearance settings' : '阅读与外观设置'
)
const supportNote = computed(() =>
  isEnglishRoute.value
    ? 'Thanks for following this project. Sharing it or joining the discussion is welcome; your attention is the greatest support.'
    : '感谢关注项目，欢迎帮忙宣传或者一起交流，你的关注就是最大的支持。'
)
const supportQrLabel = computed(() =>
  isEnglishRoute.value
    ? 'Official account / community QR code'
    : '公众号 / 社群二维码'
)
const readingToolsCopy = computed(() =>
  isEnglishRoute.value
    ? {
        appearance: 'Appearance',
        light: 'Light',
        dark: 'Dark',
        fontSize: 'Font size',
        lineHeight: 'Line height',
        decreaseFont: 'A-',
        increaseFont: 'A+',
        default: 'Default',
        tighter: 'Tighter',
        looser: 'Looser',
        switchLight: 'Switch to light mode',
        switchDark: 'Switch to dark mode'
      }
    : {
        appearance: '外观',
        light: '浅色',
        dark: '深色',
        fontSize: '字号',
        lineHeight: '行距',
        decreaseFont: 'A-',
        increaseFont: 'A+',
        default: '默认',
        tighter: '更紧',
        looser: '更松',
        switchLight: '切换到浅色模式',
        switchDark: '切换到深色模式'
      }
)
const fontSizeLabel = computed(() => `${clampFontSize(fontSize.value)}px`)
const lineHeightLabel = computed(() =>
  clampLineHeight(lineHeight.value).toFixed(2)
)
const homeTypingText = computed(
  () =>
    frontmatter.value.hero?.typingTagline ||
    frontmatter.value.hero?.tagline ||
    ''
)

let sidebarResizeLeft = 0
let outlineObserver = null
let sidebarObserver = null
let navigationSyncTimer = null
let routeLoadingTimer = null
let zoom = null
let mermaidViewerDragState = null

const MERMAID_VIEWER_MIN_SCALE = 0.02
const MERMAID_VIEWER_MAX_SCALE = 6
const MERMAID_VIEWER_SCALE_STEP = 0.25

const mermaidViewerScaleLabel = computed(
  () => `${Math.round(mermaidViewerScale.value * 100)}%`
)

const mermaidViewerImageStyle = computed(() => {
  if (!mermaidViewerNaturalWidth.value) {
    return {
      maxHeight: '100%',
      maxWidth: '100%'
    }
  }

  return {
    height: `${Math.max(
      1,
      mermaidViewerNaturalHeight.value * mermaidViewerScale.value
    )}px`,
    width: `${Math.max(1, mermaidViewerNaturalWidth.value * mermaidViewerScale.value)}px`
  }
})

const mermaidViewerStageStyle = computed(() => {
  if (!mermaidViewerNaturalWidth.value) return {}

  return {
    height: `${Math.max(
      1,
      mermaidViewerNaturalHeight.value * mermaidViewerScale.value
    )}px`,
    width: `${Math.max(1, mermaidViewerNaturalWidth.value * mermaidViewerScale.value)}px`
  }
})

const homeTaglineTyping = {
  typingSpeed: 42,
  deletingSpeed: 20,
  pauseDuration: 2600,
  initialDelay: 120
}

function clampFontSize(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return DEFAULT_FONT_SIZE
  return Math.min(MAX_FONT_SIZE, Math.max(MIN_FONT_SIZE, numeric))
}

function clampLineHeight(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return DEFAULT_LINE_HEIGHT
  return Math.min(MAX_LINE_HEIGHT, Math.max(MIN_LINE_HEIGHT, numeric))
}

function getSidebarWidthBounds() {
  if (typeof window === 'undefined') {
    return { min: MIN_SIDEBAR_WIDTH, max: MAX_SIDEBAR_WIDTH }
  }

  return {
    min: MIN_SIDEBAR_WIDTH,
    max: Math.min(
      MAX_SIDEBAR_WIDTH,
      Math.max(MIN_SIDEBAR_WIDTH, window.innerWidth - 260)
    )
  }
}

function clampSidebarWidth(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return DEFAULT_SIDEBAR_WIDTH
  const bounds = getSidebarWidthBounds()
  return Math.min(bounds.max, Math.max(bounds.min, numeric))
}

function applyFontSize(size) {
  if (typeof document === 'undefined') return
  document.documentElement.style.setProperty('--ct-doc-font-size', `${size}px`)
}

function applyLineHeight(value) {
  if (typeof document === 'undefined') return
  document.documentElement.style.setProperty(
    '--ct-doc-line-height',
    String(value)
  )
}

function applySidebarWidth(width) {
  if (typeof document === 'undefined') return
  document.documentElement.style.setProperty('--vp-sidebar-width', `${width}px`)
}

function updateSidebarEdgePosition() {
  if (typeof document === 'undefined') return
  const sidebar = document.querySelector('.VPSidebar')
  if (!sidebar) return
  document.documentElement.style.setProperty(
    '--ct-sidebar-edge-right',
    `${sidebar.getBoundingClientRect().right}px`
  )
}

function setSidebarWidth(value, persist = true) {
  const normalized = clampSidebarWidth(value)
  sidebarWidth.value = normalized
  applySidebarWidth(normalized)
  window.requestAnimationFrame(updateSidebarEdgePosition)

  if (persist && typeof localStorage !== 'undefined') {
    localStorage.setItem(SIDEBAR_WIDTH_KEY, String(normalized))
  }
}

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value
}

function closeReadingTools() {
  readingToolsOpen.value = false
}

function closeSupportPanel() {
  supportOpen.value = false
}

function setAppearance(dark) {
  isDark.value = dark
}

function showRouteLoading() {
  window.clearTimeout(routeLoadingTimer)
  routeLoadingTimer = window.setTimeout(() => {
    routeLoading.value = true
  }, 120)
}

function hideRouteLoading() {
  window.clearTimeout(routeLoadingTimer)
  routeLoading.value = false
}

function clampMermaidViewerScale(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return 1
  return Math.min(
    MERMAID_VIEWER_MAX_SCALE,
    Math.max(MERMAID_VIEWER_MIN_SCALE, numeric)
  )
}

function openMermaidViewer(image) {
  if (!image) return
  mermaidViewerSrc.value = image.currentSrc || image.src
  mermaidViewerAlt.value = image.alt || 'Mermaid diagram'
  mermaidViewerNaturalWidth.value = image.naturalWidth || 0
  mermaidViewerNaturalHeight.value = image.naturalHeight || 0
  mermaidViewerScale.value = 1
  mermaidViewerCustomZoom.value = false
  mermaidViewerOpen.value = true
  document.body.classList.add('ct-mermaid-viewer-open')
  nextTick(() => {
    window.requestAnimationFrame(fitMermaidViewerToScreen)
  })
}

function closeMermaidViewer() {
  mermaidViewerOpen.value = false
  mermaidViewerSrc.value = ''
  mermaidViewerAlt.value = ''
  mermaidViewerNaturalWidth.value = 0
  mermaidViewerNaturalHeight.value = 0
  mermaidViewerCustomZoom.value = false
  mermaidViewerDragging.value = false
  mermaidViewerDragState = null
  document.body.classList.remove('ct-mermaid-viewer-open')
}

function centerMermaidViewer() {
  const scroll = mermaidViewerScroll.value
  if (!scroll) return
  scroll.scrollLeft = Math.max(0, (scroll.scrollWidth - scroll.clientWidth) / 2)
  scroll.scrollTop = Math.max(
    0,
    (scroll.scrollHeight - scroll.clientHeight) / 2
  )
}

function getMermaidViewerFitScale() {
  const scroll = mermaidViewerScroll.value
  const width = mermaidViewerNaturalWidth.value
  const height = mermaidViewerNaturalHeight.value
  if (!scroll || !width || !height) return 1

  const styles = window.getComputedStyle(scroll)
  const availableWidth =
    scroll.clientWidth -
    Number.parseFloat(styles.paddingLeft || 0) -
    Number.parseFloat(styles.paddingRight || 0)
  const availableHeight =
    scroll.clientHeight -
    Number.parseFloat(styles.paddingTop || 0) -
    Number.parseFloat(styles.paddingBottom || 0)

  return clampMermaidViewerScale(
    Math.min(availableWidth / width, availableHeight / height, 1)
  )
}

function fitMermaidViewerToScreen() {
  mermaidViewerScale.value = getMermaidViewerFitScale()
  nextTick(() => {
    window.requestAnimationFrame(centerMermaidViewer)
  })
}

function setMermaidViewerScale(value, anchorEvent = null) {
  const scroll = mermaidViewerScroll.value
  const previousScale = mermaidViewerScale.value
  const nextScale = clampMermaidViewerScale(value)
  if (Math.abs(nextScale - previousScale) < 0.001) return

  let anchor = null
  if (scroll && anchorEvent) {
    const rect = scroll.getBoundingClientRect()
    anchor = {
      offsetX: anchorEvent.clientX - rect.left,
      offsetY: anchorEvent.clientY - rect.top,
      scrollX: scroll.scrollLeft,
      scrollY: scroll.scrollTop,
      ratio: nextScale / previousScale
    }
  }

  mermaidViewerCustomZoom.value = true
  mermaidViewerScale.value = nextScale

  nextTick(() => {
    window.requestAnimationFrame(() => {
      if (!scroll || !anchor) {
        centerMermaidViewer()
        return
      }

      scroll.scrollLeft =
        (anchor.scrollX + anchor.offsetX) * anchor.ratio - anchor.offsetX
      scroll.scrollTop =
        (anchor.scrollY + anchor.offsetY) * anchor.ratio - anchor.offsetY
    })
  })
}

function zoomMermaidViewer(delta) {
  setMermaidViewerScale(mermaidViewerScale.value + delta)
}

function resetMermaidViewerZoom() {
  mermaidViewerCustomZoom.value = false
  fitMermaidViewerToScreen()
}

function handleMermaidViewerImageLoad(event) {
  const image = event.currentTarget
  mermaidViewerNaturalWidth.value = image.naturalWidth || 0
  mermaidViewerNaturalHeight.value = image.naturalHeight || 0
  if (!mermaidViewerCustomZoom.value) {
    fitMermaidViewerToScreen()
  }
}

function handleMermaidViewerWheel(event) {
  if (!mermaidViewerOpen.value) return
  event.preventDefault()

  const delta = event.deltaY || event.deltaX
  if (!delta) return

  const factor = Math.exp(-delta * 0.0015)
  setMermaidViewerScale(mermaidViewerScale.value * factor, event)
}

function handleMermaidViewerPointerDown(event) {
  if (!mermaidViewerOpen.value || event.button !== 0) return
  const scroll = mermaidViewerScroll.value
  if (!scroll) return

  mermaidViewerDragState = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    scrollLeft: scroll.scrollLeft,
    scrollTop: scroll.scrollTop
  }
  mermaidViewerDragging.value = true
  event.currentTarget.setPointerCapture?.(event.pointerId)
  event.preventDefault()
}

function handleMermaidViewerPointerMove(event) {
  const scroll = mermaidViewerScroll.value
  const drag = mermaidViewerDragState
  if (!scroll || !drag || drag.pointerId !== event.pointerId) return

  scroll.scrollLeft = drag.scrollLeft - (event.clientX - drag.startX)
  scroll.scrollTop = drag.scrollTop - (event.clientY - drag.startY)
}

function stopMermaidViewerDrag(event = null) {
  if (event && mermaidViewerDragState?.pointerId !== event.pointerId) return
  mermaidViewerDragging.value = false
  mermaidViewerDragState = null
}

function handleMermaidViewerKeydown(event) {
  if (!mermaidViewerOpen.value) return

  if (event.key === 'Escape') {
    event.preventDefault()
    closeMermaidViewer()
    return
  }

  if (event.key === '+' || event.key === '=') {
    event.preventDefault()
    zoomMermaidViewer(MERMAID_VIEWER_SCALE_STEP)
    return
  }

  if (event.key === '-' || event.key === '_') {
    event.preventDefault()
    zoomMermaidViewer(-MERMAID_VIEWER_SCALE_STEP)
    return
  }

  if (event.key === '0') {
    event.preventDefault()
    resetMermaidViewerZoom()
  }
}

function handleMermaidImageClick(event) {
  const image = event.target.closest('img[data-mermaid-viewer="true"]')
  if (!image) return
  event.preventDefault()
  event.stopPropagation()
  openMermaidViewer(image)
}

function initMermaidViewer() {
  if (typeof document === 'undefined') return
  document.querySelectorAll('.main').forEach((main) => {
    main.removeEventListener('click', handleMermaidImageClick)
    main.addEventListener('click', handleMermaidImageClick)
  })
}

function cleanupMermaidViewer() {
  if (typeof document === 'undefined') return
  document.querySelectorAll('.main').forEach((main) => {
    main.removeEventListener('click', handleMermaidImageClick)
  })
}

function resetFontSize() {
  setFontSize(DEFAULT_FONT_SIZE)
}

function resetLineHeight() {
  setLineHeight(DEFAULT_LINE_HEIGHT)
}

function setFontSize(value) {
  fontSize.value = clampFontSize(value)
}

function setLineHeight(value) {
  lineHeight.value = clampLineHeight(value)
}

function decreaseFontSize() {
  setFontSize(fontSize.value - 1)
}

function increaseFontSize() {
  setFontSize(fontSize.value + 1)
}

function decreaseLineHeight() {
  setLineHeight(lineHeight.value - 0.05)
}

function increaseLineHeight() {
  setLineHeight(lineHeight.value + 0.05)
}

function updateFontSizeFromRange(event) {
  setFontSize(event.currentTarget.valueAsNumber)
}

function updateLineHeightFromRange(event) {
  setLineHeight(event.currentTarget.valueAsNumber)
}

function getSidebarLeftBoundary() {
  const sidebar = document.querySelector('.VPSidebar')
  if (!sidebar) return 0
  return sidebar.getBoundingClientRect().left
}

function updateSidebarWidthFromPointer(clientX) {
  const nextWidth = clientX - sidebarResizeLeft
  setSidebarWidth(nextWidth, false)
}

function handleSidebarResizeMove(event) {
  if (!sidebarResizing.value) return
  updateSidebarWidthFromPointer(event.clientX)
}

function stopSidebarResize() {
  if (!sidebarResizing.value) return

  sidebarResizing.value = false
  document.body.classList.remove('ct-sidebar-resizing')
  localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth.value))
  window.removeEventListener('pointermove', handleSidebarResizeMove)
  window.removeEventListener('pointerup', stopSidebarResize)
  window.removeEventListener('pointercancel', stopSidebarResize)
}

function startSidebarResize(event) {
  if (typeof window === 'undefined') return
  if (window.innerWidth < 960 || sidebarCollapsed.value) return

  event.preventDefault()
  sidebarResizeLeft = getSidebarLeftBoundary()
  sidebarResizing.value = true
  document.body.classList.add('ct-sidebar-resizing')
  updateSidebarWidthFromPointer(event.clientX)

  window.addEventListener('pointermove', handleSidebarResizeMove)
  window.addEventListener('pointerup', stopSidebarResize)
  window.addEventListener('pointercancel', stopSidebarResize)
}

function handleViewportResize() {
  setSidebarWidth(sidebarWidth.value, false)
  updateSidebarEdgePosition()
  if (mermaidViewerOpen.value && !mermaidViewerCustomZoom.value) {
    window.requestAnimationFrame(fitMermaidViewerToScreen)
  }
}

function handleWindowKeydown(event) {
  handleMermaidViewerKeydown(event)
  if (event.defaultPrevented) return

  if (event.key === 'Escape') {
    closeReadingTools()
    closeSupportPanel()
  }
}

function scrollOutlineToActiveItem(activeLink) {
  const outlineContainer = document.querySelector('.VPDocAsideOutline')
  if (!outlineContainer || !activeLink) return

  const containerRect = outlineContainer.getBoundingClientRect()
  const linkRect = activeLink.getBoundingClientRect()
  const linkTop = linkRect.top - containerRect.top + outlineContainer.scrollTop
  const targetScrollTop =
    linkTop - containerRect.height / 2 + linkRect.height / 2

  const isAbove = linkRect.top < containerRect.top + 20
  const isBelow = linkRect.bottom > containerRect.bottom - 20

  if (isAbove || isBelow) {
    outlineContainer.scrollTo({
      top: targetScrollTop,
      behavior: 'smooth'
    })
  }
}

function scrollSidebarToActiveItem(activeItem) {
  const sidebarContainer = document.querySelector('.VPSidebar')
  if (!sidebarContainer || !activeItem) return

  const target =
    activeItem.querySelector('.item') ||
    activeItem.querySelector('a') ||
    activeItem

  const containerRect = sidebarContainer.getBoundingClientRect()
  const targetRect = target.getBoundingClientRect()
  const targetTop =
    targetRect.top - containerRect.top + sidebarContainer.scrollTop
  const targetScrollTop =
    targetTop - containerRect.height / 2 + targetRect.height / 2
  const isInside =
    targetRect.top >= containerRect.top + 16 &&
    targetRect.bottom <= containerRect.bottom - 16

  if (!isInside) {
    sidebarContainer.scrollTo({
      top: targetScrollTop,
      behavior: 'smooth'
    })
  }
}

function cleanupNavigationSync() {
  outlineObserver?.disconnect()
  sidebarObserver?.disconnect()
  outlineObserver = null
  sidebarObserver = null

  if (navigationSyncTimer) {
    window.clearTimeout(navigationSyncTimer)
    navigationSyncTimer = null
  }
}

function initMediumZoom() {
  if (typeof document === 'undefined') return
  if (zoom) zoom.detach()
  zoom = mediumZoom('.main img:not([data-mermaid-viewer="true"])', {
    background: 'rgba(15, 23, 42, 0.62)',
    margin: 24
  })
}

function renderSidebarKatex() {
  // no-op: sidebar labels are plain text
}

function enhanceNavTitle() {
  if (typeof document === 'undefined') return
  const title = document.querySelector('.VPNavBar .title')
  if (!title) return

  if (title.tagName === 'A') {
    title.href = withBase('/preface/intro')
  }

  const titleText = title.querySelector('span:last-of-type') || title
  if (titleText.dataset.ctEnhancedTitle === 'true') return

  const text = titleText.textContent?.trim()
  if (text !== 'Hands on Modern RL') return

  titleText.dataset.ctEnhancedTitle = 'true'
  titleText.classList.add('ct-nav-title-text')
  titleText.innerHTML =
    '<span class="ct-nav-title-main">Hands on </span><span class="ct-nav-title-accent">Modern RL</span>'
}

function initNavigationSync() {
  cleanupNavigationSync()

  navigationSyncTimer = window.setTimeout(() => {
    const outlineContainer = document.querySelector('.VPDocAsideOutline')
    const sidebarContainer = document.querySelector('.VPSidebar')

    if (outlineContainer) {
      outlineObserver = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
          const target = mutation.target
          if (target.classList?.contains('active') && target.tagName === 'A') {
            scrollOutlineToActiveItem(target)
          }
        }
      })

      outlineObserver.observe(outlineContainer, {
        attributes: true,
        subtree: true,
        attributeFilter: ['class']
      })

      const currentActive = outlineContainer.querySelector('.active')
      if (currentActive) {
        scrollOutlineToActiveItem(currentActive)
      }
    }

    if (sidebarContainer) {
      sidebarObserver = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
          const target = mutation.target
          if (target.classList?.contains('is-active')) {
            scrollSidebarToActiveItem(target)
          }
        }
      })

      sidebarObserver.observe(sidebarContainer, {
        attributes: true,
        subtree: true,
        attributeFilter: ['class']
      })

      const currentSidebarActive = sidebarContainer.querySelector('.is-active')
      if (currentSidebarActive) {
        scrollSidebarToActiveItem(currentSidebarActive)
      }
    }

    updateSidebarEdgePosition()
  }, 80)
}

onMounted(() => {
  const savedFontSize = clampFontSize(
    localStorage.getItem(FONT_SIZE_STORAGE_KEY)
  )
  const savedLineHeight = clampLineHeight(
    localStorage.getItem(LINE_HEIGHT_STORAGE_KEY)
  )
  const savedSidebarWidth = localStorage.getItem(SIDEBAR_WIDTH_KEY)
  const savedCollapsed = localStorage.getItem(SIDEBAR_COLLAPSED_KEY)

  fontSize.value = savedFontSize
  lineHeight.value = savedLineHeight
  applyFontSize(savedFontSize)
  applyLineHeight(savedLineHeight)

  if (savedSidebarWidth) {
    setSidebarWidth(savedSidebarWidth, false)
  } else {
    setSidebarWidth(DEFAULT_SIDEBAR_WIDTH, false)
  }

  sidebarCollapsed.value = savedCollapsed === 'true'
  document.body.classList.toggle('ct-sidebar-collapsed', sidebarCollapsed.value)

  window.addEventListener('resize', handleViewportResize)
  window.addEventListener('keydown', handleWindowKeydown)
  initNavigationSync()
  updateSidebarEdgePosition()
  initMediumZoom()
  initMermaidViewer()
  enhanceNavTitle()
  renderSidebarKatex()
  initGithubStars(theme)

  router.onBeforeRouteChange = () => {
    showRouteLoading()
  }

  router.onAfterRouteChanged = () => {
    hideRouteLoading()
  }
})

onBeforeUnmount(() => {
  stopSidebarResize()
  cleanupNavigationSync()
  cleanupMermaidViewer()
  closeMermaidViewer()
  hideRouteLoading()
  router.onBeforeRouteChange = undefined
  router.onAfterRouteChanged = undefined
  window.removeEventListener('resize', handleViewportResize)
  window.removeEventListener('keydown', handleWindowKeydown)
})

watch(fontSize, (next) => {
  const normalized = clampFontSize(next)
  if (fontSize.value !== normalized) {
    fontSize.value = normalized
  }
  applyFontSize(normalized)
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(FONT_SIZE_STORAGE_KEY, String(normalized))
  }
})

watch(lineHeight, (next) => {
  const normalized = clampLineHeight(next)
  if (lineHeight.value !== normalized) {
    lineHeight.value = normalized
  }
  applyLineHeight(normalized)
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(LINE_HEIGHT_STORAGE_KEY, String(normalized))
  }
})

watch(sidebarCollapsed, (collapsed) => {
  if (typeof document === 'undefined') return
  document.body.classList.toggle('ct-sidebar-collapsed', collapsed)
  localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(collapsed))
  window.requestAnimationFrame(updateSidebarEdgePosition)
})

watch(readingToolsOpen, (open) => {
  if (open) supportOpen.value = false
})

watch(supportOpen, (open) => {
  if (open) readingToolsOpen.value = false
})

watch(
  () => route.path,
  async () => {
    await nextTick()
    initNavigationSync()
    initMediumZoom()
    initMermaidViewer()
    enhanceNavTitle()
    renderSidebarKatex()
    window.requestAnimationFrame(updateSidebarEdgePosition)
  }
)
</script>

<template>
  <DefaultTheme.Layout>
    <template v-if="showDocChrome" #nav-bar-content-after>
      <div class="ct-nav-tools">
        <PopoverRoot v-model:open="readingToolsOpen">
          <PopoverTrigger as-child>
            <button
              class="ct-nav-tool-button"
              type="button"
              :aria-label="settingsButtonLabel"
            >
              <Settings :size="18" :stroke-width="2" aria-hidden="true" />
            </button>
          </PopoverTrigger>

          <PopoverPortal>
            <Transition name="ct-reading-tools-fade">
              <PopoverContent
                class="ct-popover-content"
                :side-offset="10"
                align="end"
                side="bottom"
              >
                <div class="ct-popover-surface ct-reading-tools-panel">
                  <div class="ct-reading-tools-group">
                    <div class="ct-reading-tools-header">
                      <div class="ct-reading-tools-title">
                        {{ readingToolsCopy.appearance }}
                      </div>
                      <div class="ct-reading-tools-value">
                        {{
                          isDark
                            ? readingToolsCopy.dark
                            : readingToolsCopy.light
                        }}
                      </div>
                    </div>
                    <div
                      class="ct-appearance-toggle"
                      role="group"
                      :aria-label="readingToolsCopy.appearance"
                    >
                      <button
                        class="ct-reading-tools-action"
                        :class="{ active: !isDark }"
                        type="button"
                        :aria-label="readingToolsCopy.switchLight"
                        @click="setAppearance(false)"
                      >
                        <Sun :size="18" :stroke-width="2" aria-hidden="true" />
                      </button>
                      <button
                        class="ct-reading-tools-action"
                        :class="{ active: isDark }"
                        type="button"
                        :aria-label="readingToolsCopy.switchDark"
                        @click="setAppearance(true)"
                      >
                        <Moon
                          :size="18"
                          :stroke-width="2"
                          aria-hidden="true"
                        />
                      </button>
                    </div>
                  </div>

                  <div class="ct-reading-tools-group">
                    <div class="ct-reading-tools-header">
                      <div class="ct-reading-tools-title">
                        {{ readingToolsCopy.fontSize }}
                      </div>
                      <div class="ct-reading-tools-value">
                        {{ fontSizeLabel }}
                      </div>
                    </div>
                    <div class="ct-reading-tools-actions">
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="decreaseFontSize"
                      >
                        {{ readingToolsCopy.decreaseFont }}
                      </button>
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="resetFontSize"
                      >
                        {{ readingToolsCopy.default }}
                      </button>
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="increaseFontSize"
                      >
                        {{ readingToolsCopy.increaseFont }}
                      </button>
                    </div>
                    <input
                      class="ct-reading-tools-range"
                      type="range"
                      :value="fontSize"
                      :min="MIN_FONT_SIZE"
                      :max="MAX_FONT_SIZE"
                      step="1"
                      @input="updateFontSizeFromRange"
                    />
                  </div>

                  <div class="ct-reading-tools-group">
                    <div class="ct-reading-tools-header">
                      <div class="ct-reading-tools-title">
                        {{ readingToolsCopy.lineHeight }}
                      </div>
                      <div class="ct-reading-tools-value">
                        {{ lineHeightLabel }}
                      </div>
                    </div>
                    <div class="ct-reading-tools-actions">
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="decreaseLineHeight"
                      >
                        {{ readingToolsCopy.tighter }}
                      </button>
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="resetLineHeight"
                      >
                        {{ readingToolsCopy.default }}
                      </button>
                      <button
                        class="ct-reading-tools-action"
                        type="button"
                        @click="increaseLineHeight"
                      >
                        {{ readingToolsCopy.looser }}
                      </button>
                    </div>
                    <input
                      class="ct-reading-tools-range"
                      type="range"
                      :value="lineHeight"
                      :min="MIN_LINE_HEIGHT"
                      :max="MAX_LINE_HEIGHT"
                      step="0.05"
                      @input="updateLineHeightFromRange"
                    />
                  </div>
                </div>
              </PopoverContent>
            </Transition>
          </PopoverPortal>
        </PopoverRoot>

        <PopoverRoot v-model:open="supportOpen">
          <PopoverTrigger as-child>
            <button
              class="ct-nav-tool-button"
              type="button"
              :aria-label="supportButtonLabel"
              :title="supportButtonLabel"
            >
              <HandHeart :size="18" :stroke-width="2" aria-hidden="true" />
            </button>
          </PopoverTrigger>

          <PopoverPortal>
            <Transition name="ct-reading-tools-fade">
              <PopoverContent
                class="ct-popover-content"
                :side-offset="10"
                align="end"
                side="bottom"
              >
                <div class="ct-popover-surface ct-support-panel">
                  <a
                    class="ct-support-link"
                    href="https://github.com/walkinglabs"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <span>WalkingLab</span>
                    <span>GitHub</span>
                  </a>
                  <div class="ct-support-qr-card">
                    <img
                      src="https://github.com/walkinglabs/.github/raw/main/profile/wechat.png"
                      alt="WalkingLab 微信二维码"
                      loading="lazy"
                      decoding="async"
                    />
                    <div>{{ supportQrLabel }}</div>
                  </div>
                  <p class="ct-support-note">
                    {{ supportNote }}
                  </p>
                </div>
              </PopoverContent>
            </Transition>
          </PopoverPortal>
        </PopoverRoot>
      </div>
    </template>

    <template v-if="showDocChrome" #sidebar-nav-after>
      <SidebarFooter @open-settings="readingToolsOpen = true" />
    </template>

    <template v-if="isHomePage && homeTypingText" #home-hero-info-after>
      <div class="ct-home-typed-tagline">
        <ClientOnly>
          <TextType :text="homeTypingText" v-bind="homeTaglineTyping" />
        </ClientOnly>
      </div>
    </template>

    <template #nav-screen-content-after>
      <div class="ct-mobile-language-switcher">
        <div class="ct-mobile-language-title">切换语言</div>
        <div class="ct-mobile-language-options">
          <span class="ct-mobile-language-current">
            {{ mobileCurrentLanguage }}
          </span>
          <a
            class="ct-mobile-language-link"
            :href="mobileAlternateLanguageLink"
          >
            {{ mobileAlternateLanguage }}
          </a>
        </div>
      </div>
    </template>
  </DefaultTheme.Layout>

  <ClientOnly>
    <div
      v-if="showDocChrome"
      class="ct-sidebar-hover-area"
      :class="{ collapsed: sidebarCollapsed, resizing: sidebarResizing }"
    >
      <div
        v-if="!sidebarCollapsed"
        class="ct-sidebar-resizer"
        role="separator"
        aria-orientation="vertical"
        @pointerdown="startSidebarResize"
      />

      <button
        class="ct-sidebar-toggle-btn"
        :class="{ collapsed: sidebarCollapsed }"
        type="button"
        :aria-label="sidebarCollapsed ? '展开目录' : '收起目录'"
        @click="toggleSidebar"
      >
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
          <path
            v-if="!sidebarCollapsed"
            d="M8 1L3 6l5 5"
            stroke="currentColor"
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="1.5"
          />
          <path
            v-else
            d="M4 1l5 5-5 5"
            stroke="currentColor"
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="1.5"
          />
        </svg>
      </button>
    </div>
  </ClientOnly>

  <ClientOnly>
    <ReadingProgress v-if="showDocChrome" />
  </ClientOnly>

  <ClientOnly>
    <Transition name="ct-route-loading-fade">
      <div v-if="routeLoading" class="ct-route-loading" aria-live="polite">
        <span class="ct-route-loading-spinner" aria-hidden="true"></span>
        <span class="ct-route-loading-text">加载中</span>
      </div>
    </Transition>
  </ClientOnly>

  <ClientOnly>
    <Teleport to="body">
      <div
        v-if="mermaidViewerOpen"
        class="ct-mermaid-viewer"
        role="dialog"
        aria-modal="true"
        aria-label="查看图表"
        @click.self="closeMermaidViewer"
      >
        <div class="ct-mermaid-viewer-toolbar">
          <div class="ct-mermaid-viewer-help">
            <strong>{{ mermaidViewerScaleLabel }}</strong>
            <span>滚轮缩放 · 拖拽移动 · + / - 缩放 · 0 重置 · Esc 关闭</span>
          </div>
          <div class="ct-mermaid-viewer-actions">
            <button
              type="button"
              aria-label="缩小图表"
              @click="zoomMermaidViewer(-MERMAID_VIEWER_SCALE_STEP)"
            >
              -
            </button>
            <button type="button" @click="resetMermaidViewerZoom">重置</button>
            <button
              type="button"
              aria-label="放大图表"
              @click="zoomMermaidViewer(MERMAID_VIEWER_SCALE_STEP)"
            >
              +
            </button>
            <button
              type="button"
              aria-label="关闭图表"
              @click="closeMermaidViewer"
            >
              关闭
            </button>
          </div>
        </div>
        <div
          ref="mermaidViewerScroll"
          class="ct-mermaid-viewer-scroll"
          :class="{ 'is-dragging': mermaidViewerDragging }"
          @pointerdown="handleMermaidViewerPointerDown"
          @pointermove="handleMermaidViewerPointerMove"
          @pointerup="stopMermaidViewerDrag"
          @pointercancel="stopMermaidViewerDrag"
          @pointerleave="stopMermaidViewerDrag"
          @wheel="handleMermaidViewerWheel"
        >
          <div class="ct-mermaid-viewer-stage" :style="mermaidViewerStageStyle">
            <img
              class="ct-mermaid-viewer-image"
              :src="mermaidViewerSrc"
              :alt="mermaidViewerAlt"
              :style="mermaidViewerImageStyle"
              @load="handleMermaidViewerImageLoad"
              @dragstart.prevent
            />
          </div>
        </div>
      </div>
    </Teleport>
  </ClientOnly>
</template>

<style>
.ct-nav-tools {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-left: 6px;
}

.VPNavBar .ct-nav-tools {
  order: 21;
}

.VPNavBar .appearance {
  display: none;
}

.ct-nav-tool-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 34px;
  min-width: 34px;
  padding: 0 10px;
  border: 1px solid transparent;
  border-radius: 999px;
  background: transparent;
  color: rgba(29, 29, 31, 0.58);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0;
  box-shadow: none;
  cursor: pointer;
  transition:
    background-color 0.18s ease,
    color 0.2s ease,
    border-color 0.18s ease;
}

.ct-nav-tool-button:hover,
.ct-nav-tool-button:focus-visible,
.ct-nav-tool-button[data-state='open'] {
  border-color: rgba(0, 0, 0, 0.05);
  background: rgba(0, 0, 0, 0.04);
  color: rgba(29, 29, 31, 0.82);
}

.ct-popover-content {
  z-index: 40;
  outline: none;
}

.ct-popover-surface {
  transform: translateY(0) scale(1);
  transform-origin: var(--reka-popover-content-transform-origin, top right);
  will-change: transform;
}

.ct-reading-tools-panel {
  width: 280px;
  padding: 14px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 12px;
  background: #fff;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.1);
}

.ct-reading-tools-group {
  display: grid;
  gap: 10px;
}

.ct-reading-tools-group + .ct-reading-tools-group {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid rgba(148, 163, 184, 0.16);
}

.ct-reading-tools-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}

.ct-reading-tools-title {
  font-size: 13px;
  font-weight: 700;
  color: var(--vp-c-text-1);
}

.ct-reading-tools-value {
  font-size: 12px;
  color: var(--vp-c-text-2);
}

.ct-reading-tools-actions {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}

.ct-reading-tools-action {
  height: 34px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.02);
  color: var(--vp-c-text-1);
  font-size: 13px;
  cursor: pointer;
  transition:
    background-color 0.2s ease,
    border-color 0.2s ease,
    color 0.2s ease;
}

.ct-reading-tools-action:hover {
  border-color: rgba(15, 118, 110, 0.32);
  color: var(--vp-c-brand-1);
}

.ct-reading-tools-action.active {
  border-color: rgba(63, 81, 181, 0.36);
  background: rgba(63, 81, 181, 0.1);
  color: var(--vp-c-brand-1);
  font-weight: 700;
}

.ct-appearance-toggle {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}

.ct-appearance-toggle .ct-reading-tools-action {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
}

.ct-reading-tools-range {
  width: 100%;
  accent-color: var(--vp-c-brand-1);
}

.ct-support-panel {
  width: 220px;
  padding: 12px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 12px;
  background: #fff;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.1);
}

.ct-support-link {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 38px;
  padding: 0 10px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 8px;
  color: var(--vp-c-text-1);
  font-size: 13px;
  font-weight: 700;
  text-decoration: none;
  transition:
    background-color 0.2s ease,
    border-color 0.2s ease,
    color 0.2s ease;
}

.ct-support-link span:last-child {
  color: var(--vp-c-text-2);
  font-size: 12px;
  font-weight: 500;
}

.ct-support-link:hover {
  border-color: rgba(15, 118, 110, 0.32);
  background: rgba(63, 81, 181, 0.06);
  color: var(--vp-c-brand-1);
}

.ct-support-qr-card {
  display: grid;
  gap: 8px;
  margin-top: 10px;
  color: var(--vp-c-text-2);
  font-size: 12px;
  text-align: center;
}

.ct-support-qr-card img {
  display: block;
  width: 100%;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 8px;
  background: #fff;
}

.ct-support-note {
  margin: 10px 0 0;
  color: var(--vp-c-text-2);
  font-size: 12px;
  line-height: 1.65;
}

.ct-reading-tools-fade-enter-active,
.ct-reading-tools-fade-leave-active {
  transition: opacity 0.18s cubic-bezier(0.22, 1, 0.36, 1);
}

.ct-reading-tools-fade-enter-active .ct-popover-surface {
  animation: ct-popover-surface-enter 0.18s cubic-bezier(0.22, 1, 0.36, 1)
    both;
}

.ct-reading-tools-fade-leave-active .ct-popover-surface {
  animation: ct-popover-surface-leave 0.14s ease both;
}

.ct-reading-tools-fade-enter-from,
.ct-reading-tools-fade-leave-to {
  opacity: 0;
}

@keyframes ct-popover-surface-enter {
  from {
    transform: translateY(-4px) scale(0.98);
  }

  to {
    transform: translateY(0) scale(1);
  }
}

@keyframes ct-popover-surface-leave {
  from {
    transform: translateY(0) scale(1);
  }

  to {
    transform: translateY(-4px) scale(0.98);
  }
}

@media (prefers-reduced-motion: reduce) {
  .ct-reading-tools-fade-enter-active,
  .ct-reading-tools-fade-leave-active,
  .ct-reading-tools-fade-enter-active .ct-popover-surface,
  .ct-reading-tools-fade-leave-active .ct-popover-surface {
    transition: none;
  }

  .ct-reading-tools-fade-enter-active .ct-popover-surface,
  .ct-reading-tools-fade-leave-active .ct-popover-surface {
    animation: none;
  }

  .ct-popover-surface {
    transform: none;
  }
}

.ct-route-loading {
  position: fixed;
  top: calc(var(--vp-nav-height, 64px) + 14px);
  left: 50%;
  z-index: 80;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border: 1px solid rgba(63, 81, 181, 0.12);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.92);
  color: rgba(29, 29, 31, 0.72);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.12);
  font-size: 13px;
  font-weight: 500;
  line-height: 1;
  transform: translateX(-50%);
  -webkit-backdrop-filter: saturate(180%) blur(16px);
  backdrop-filter: saturate(180%) blur(16px);
}

.ct-route-loading-spinner {
  width: 14px;
  height: 14px;
  border: 2px solid rgba(63, 81, 181, 0.18);
  border-top-color: var(--vp-c-brand-1);
  border-radius: 50%;
  animation: ct-route-loading-spin 0.72s linear infinite;
}

.ct-route-loading-fade-enter-active,
.ct-route-loading-fade-leave-active {
  transition:
    opacity 0.16s ease,
    transform 0.16s ease;
}

.ct-route-loading-fade-enter-from,
.ct-route-loading-fade-leave-to {
  opacity: 0;
  transform: translate(-50%, -6px);
}

.dark .ct-route-loading {
  border-color: rgba(154, 168, 255, 0.18);
  background: rgba(24, 24, 27, 0.88);
  color: rgba(245, 245, 247, 0.76);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.28);
}

.dark .ct-route-loading-spinner {
  border-color: rgba(154, 168, 255, 0.22);
  border-top-color: #9aa8ff;
}

@keyframes ct-route-loading-spin {
  to {
    transform: rotate(360deg);
  }
}

.ct-mobile-language-switcher {
  display: none;
  margin-top: 24px;
  padding-top: 22px;
  border-top: 1px solid var(--vp-c-divider);
}

.ct-mobile-language-title {
  font-size: 14px;
  font-weight: 500;
  color: var(--vp-c-text-1);
}

.ct-mobile-language-options {
  display: flex;
  align-items: center;
  gap: 18px;
  margin-top: 12px;
  font-size: 13px;
  line-height: 32px;
}

.ct-mobile-language-current {
  color: var(--vp-c-brand-1);
  font-weight: 500;
}

.ct-mobile-language-link {
  color: var(--vp-c-text-1);
  transition: color 0.2s;
}

.ct-mobile-language-link:hover {
  color: var(--vp-c-brand-1);
}

.ct-sidebar-hover-area {
  display: none;
  position: fixed;
  top: 0;
  left: calc(
    var(--ct-sidebar-edge-right, var(--vp-sidebar-width, 212px)) - 14px
  );
  width: 24px;
  height: 100vh;
  z-index: 30;
}

.ct-sidebar-hover-area.collapsed {
  left: 0;
  width: 28px;
}

.ct-sidebar-resizer {
  position: absolute;
  left: 12px;
  top: 0;
  width: 2px;
  height: 100%;
  background: var(--vp-c-divider);
  opacity: 0;
  cursor: col-resize;
  transition:
    opacity 0.2s ease,
    background-color 0.2s ease;
}

.ct-sidebar-hover-area:hover .ct-sidebar-resizer,
.ct-sidebar-hover-area.resizing .ct-sidebar-resizer {
  opacity: 1;
  background: var(--vp-c-brand-1);
}

.ct-sidebar-toggle-btn {
  position: absolute;
  top: 50%;
  left: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 38px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 0 4px 4px 0;
  background: rgba(255, 255, 255, 0.92);
  color: var(--vp-c-text-3);
  box-shadow:
    0 2px 2px 0 rgba(0, 0, 0, 0.14),
    0 3px 1px -2px rgba(0, 0, 0, 0.2),
    0 1px 5px 0 rgba(0, 0, 0, 0.12);
  cursor: pointer;
  transform: translateY(-50%);
  opacity: 0;
  transition:
    opacity 0.2s ease,
    color 0.2s ease,
    background-color 0.2s ease;
}

.ct-sidebar-hover-area:hover .ct-sidebar-toggle-btn,
.ct-sidebar-hover-area.resizing .ct-sidebar-toggle-btn {
  opacity: 1;
}

.ct-sidebar-hover-area.collapsed .ct-sidebar-toggle-btn {
  opacity: 0.92;
}

.ct-sidebar-toggle-btn:hover {
  background: var(--vp-c-bg);
  color: var(--vp-c-brand-1);
}

@media (min-width: 960px) {
  .ct-sidebar-hover-area {
    display: block;
  }
}

@media (min-width: 1440px) {
  .VPContent.has-sidebar,
  .VPNavBar.has-sidebar .content,
  .VPNavBar.has-sidebar .divider {
    transition:
      padding-left 0.26s ease,
      transform 0.26s ease;
  }
}

.ct-sidebar-resizing,
.ct-sidebar-resizing * {
  cursor: col-resize !important;
  user-select: none;
}

.ct-sidebar-resizing .VPSidebar,
.ct-sidebar-resizing .VPContent.has-sidebar,
.ct-sidebar-resizing .VPNavBar.has-sidebar .content,
.ct-sidebar-resizing .VPNavBar.has-sidebar .divider {
  transition: none !important;
}

.dark .ct-nav-tool-button,
.dark .ct-sidebar-toggle-btn {
  border-color: rgba(255, 255, 255, 0.12);
  background: rgba(30, 30, 40, 0.92);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
}

.dark .ct-reading-tools-panel,
.dark .ct-support-panel {
  border-color: rgba(255, 255, 255, 0.12);
  background: rgb(30, 30, 40);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
}

.dark .ct-nav-tool-button {
  border-color: transparent;
  background: transparent;
  color: rgba(245, 245, 247, 0.58);
  box-shadow: none;
}

.dark .ct-nav-tool-button:hover,
.dark .ct-nav-tool-button:focus-visible,
.dark .ct-nav-tool-button[data-state='open'] {
  border-color: rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.08);
  color: rgba(245, 245, 247, 0.84);
}

.dark .ct-reading-tools-action {
  border-color: rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.04);
}

.dark .ct-reading-tools-action.active {
  border-color: rgba(154, 168, 255, 0.34);
  background: rgba(154, 168, 255, 0.12);
}

.dark .ct-support-link,
.dark .ct-support-qr-card img {
  border-color: rgba(255, 255, 255, 0.1);
}

@media (min-width: 960px) {
  .ct-sidebar-collapsed .VPSidebar {
    display: none !important;
  }

  .ct-sidebar-collapsed .VPNavBar.has-sidebar .wrapper {
    padding: 0 32px;
  }

  .ct-sidebar-collapsed .VPNavBar.has-sidebar .container {
    max-width: calc(var(--vp-layout-max-width) - 64px);
  }

  .ct-sidebar-collapsed .VPNavBar.has-sidebar .title {
    position: static !important;
    width: auto !important;
    padding: 0 !important;
    background: transparent !important;
  }

  .ct-sidebar-collapsed .VPNavBarTitle.has-sidebar .title {
    border-bottom-color: transparent !important;
  }

  .ct-sidebar-collapsed .VPContent.has-sidebar,
  .ct-sidebar-collapsed .VPNavBar.has-sidebar .content,
  .ct-sidebar-collapsed .VPNavBar.has-sidebar .divider {
    padding-left: 0 !important;
  }
}

@media (min-width: 1440px) {
  .ct-sidebar-collapsed .VPContent.has-sidebar {
    padding-left: calc(
      (100% - var(--vp-layout-max-width, 1440px)) / 2
    ) !important;
  }

  .ct-sidebar-collapsed .VPNavBar.has-sidebar .wrapper {
    padding: 0;
  }

  .ct-sidebar-collapsed .VPNavBar.has-sidebar .container {
    max-width: var(--vp-layout-max-width);
  }

  .ct-sidebar-collapsed .VPNavBar.has-sidebar .content,
  .ct-sidebar-collapsed .VPNavBar.has-sidebar .divider {
    padding-left: calc(
      (100% - var(--vp-layout-max-width, 1440px)) / 2
    ) !important;
  }
}

.VPHomeHero .tagline {
  display: none !important;
}

.ct-home-typed-tagline {
  margin-top: 10px;
  min-height: 30px;
  font-size: 18px;
  line-height: 1.6;
  font-weight: 500;
  color: var(--vp-c-text-2);
  text-align: center;
  white-space: pre-wrap;
}

.VPHomeHero .container,
.VPHomeHero .main,
.VPHomeHero .name,
.VPHomeHero .text,
.VPHomeHero .actions {
  text-align: center;
}

.VPHomeHero .actions {
  justify-content: center;
}

@media (min-width: 960px) {
  .ct-home-typed-tagline {
    font-size: 22px;
    line-height: 1.7;
  }
}

@media (max-width: 768px) {
  .ct-nav-tools {
    margin-left: 4px;
  }

  .ct-support-panel,
  .ct-reading-tools-panel {
    right: -6px;
    width: min(280px, calc(100vw - 24px));
  }

  .ct-mobile-language-switcher {
    display: block;
  }
}

.medium-zoom-overlay {
  z-index: 999;
}

.medium-zoom--opened .medium-zoom-overlay {
  background: rgba(15, 23, 42, 0.62) !important;
}

.medium-zoom-image--opened {
  z-index: 1000;
}

.ct-mermaid-viewer-open {
  overflow: hidden;
}

.ct-mermaid-viewer {
  position: fixed;
  inset: 0;
  z-index: 1001;
  display: grid;
  grid-template-rows: auto 1fr;
  background: rgba(15, 23, 42, 0.62);
}

.ct-mermaid-viewer-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
  padding: 10px 12px;
}

.ct-mermaid-viewer-help {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 10px;
  color: rgba(255, 255, 255, 0.9);
  font-size: 12px;
  line-height: 1.4;
}

.ct-mermaid-viewer-help strong {
  flex: 0 0 auto;
  min-width: 44px;
  color: #fff;
  font-weight: 700;
}

.ct-mermaid-viewer-help span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ct-mermaid-viewer-actions {
  display: flex;
  flex: 0 0 auto;
  gap: 8px;
}

.ct-mermaid-viewer-actions button {
  height: 34px;
  min-width: 44px;
  padding: 0 12px;
  border: 1px solid rgba(255, 255, 255, 0.22);
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.92);
  color: #111827;
  font-size: 13px;
  cursor: pointer;
}

.ct-mermaid-viewer-actions button:hover {
  background: #fff;
}

.ct-mermaid-viewer-scroll {
  overflow: auto;
  padding: 24px;
  cursor: grab;
  overscroll-behavior: contain;
  user-select: none;
}

.ct-mermaid-viewer-scroll.is-dragging {
  cursor: grabbing;
}

.ct-mermaid-viewer-stage {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 100%;
  min-height: 100%;
  margin: 0 auto;
}

.ct-mermaid-viewer-image {
  display: block;
  width: auto;
  max-width: none;
  max-height: none;
  height: auto;
  background: #fff;
  cursor: inherit;
  user-select: none;
  -webkit-user-drag: none;
  touch-action: none;
}

.main img {
  cursor: zoom-in;
  transition: transform 0.2s ease;
}

.main img:not(.ct-mermaid-viewer-image):hover {
  transform: scale(1.01);
}

@media (max-width: 640px) {
  .ct-mermaid-viewer-toolbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .ct-mermaid-viewer-help span {
    white-space: normal;
  }

  .ct-mermaid-viewer-actions {
    width: 100%;
  }

  .ct-mermaid-viewer-actions button {
    flex: 1 1 0;
  }
}
</style>
