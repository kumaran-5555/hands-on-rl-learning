<script setup>
import DefaultTheme from 'vitepress/theme'
import { useData, useRoute } from 'vitepress'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import ReadingProgress from './components/ReadingProgress.vue'
import TextType from './components/TextType.vue'
import mediumZoom from 'medium-zoom'

const { frontmatter } = useData()
const route = useRoute()

const FONT_SIZE_STORAGE_KEY = 'ct-doc-font-size'
const LINE_HEIGHT_STORAGE_KEY = 'ct-doc-line-height'
const SIDEBAR_COLLAPSED_KEY = 'ct-sidebar-collapsed'
const SIDEBAR_WIDTH_KEY = 'ct-sidebar-width'

const MIN_FONT_SIZE = 15
const MAX_FONT_SIZE = 20
const DEFAULT_FONT_SIZE = 16
const MIN_LINE_HEIGHT = 1.55
const MAX_LINE_HEIGHT = 2
const DEFAULT_LINE_HEIGHT = 1.75

const DEFAULT_SIDEBAR_WIDTH = 272
const MIN_SIDEBAR_WIDTH = 190
const MAX_SIDEBAR_WIDTH = 520

const fontSize = ref(DEFAULT_FONT_SIZE)
const lineHeight = ref(DEFAULT_LINE_HEIGHT)
const readingToolsOpen = ref(false)
const sidebarCollapsed = ref(false)
const sidebarWidth = ref(DEFAULT_SIDEBAR_WIDTH)
const sidebarResizing = ref(false)

const readingToolsButton = ref(null)
const readingToolsPanel = ref(null)

const isHomePage = computed(() => frontmatter.value.layout === 'home')
const showDocChrome = computed(() => !isHomePage.value)
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
let zoom = null

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

function toggleReadingTools() {
  readingToolsOpen.value = !readingToolsOpen.value
}

function closeReadingTools() {
  readingToolsOpen.value = false
}

function resetFontSize() {
  fontSize.value = DEFAULT_FONT_SIZE
}

function resetLineHeight() {
  lineHeight.value = DEFAULT_LINE_HEIGHT
}

function decreaseFontSize() {
  fontSize.value = clampFontSize(fontSize.value - 1)
}

function increaseFontSize() {
  fontSize.value = clampFontSize(fontSize.value + 1)
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
}

function handleDocumentPointerDown(event) {
  if (!readingToolsOpen.value) return

  const target = event.target
  if (readingToolsPanel.value?.contains(target)) return
  if (readingToolsButton.value?.contains(target)) return

  closeReadingTools()
}

function handleWindowKeydown(event) {
  if (event.key === 'Escape') {
    closeReadingTools()
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
  zoom = mediumZoom('.main img', {
    background: 'var(--vp-c-bg)',
    margin: 24,
  })
}

function renderSidebarKatex() {
  if (typeof document === 'undefined') return
  import('katex').then((katex) => {
    const items = document.querySelectorAll('.VPSidebar .VPSidebarItem .text')
    items.forEach((el) => {
      const raw = el.textContent
      if (!raw || !raw.includes('$')) return
      const html = raw.replace(/\$([^$]+)\$/g, (_match, formula) => {
        try {
          return katex.default.renderToString(formula, {
            throwOnError: false,
            output: 'htmlAndMathml',
          })
        } catch {
          return `$${formula}$`
        }
      })
      if (html !== raw) {
        el.innerHTML = html
      }
    })
  }).catch(() => {})
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
  document.addEventListener('pointerdown', handleDocumentPointerDown)
  window.addEventListener('keydown', handleWindowKeydown)
  initNavigationSync()
  updateSidebarEdgePosition()
  initMediumZoom()
  renderSidebarKatex()
})

onBeforeUnmount(() => {
  stopSidebarResize()
  cleanupNavigationSync()
  window.removeEventListener('resize', handleViewportResize)
  document.removeEventListener('pointerdown', handleDocumentPointerDown)
  window.removeEventListener('keydown', handleWindowKeydown)
})

watch(fontSize, (next) => {
  const normalized = clampFontSize(next)
  applyFontSize(normalized)
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(FONT_SIZE_STORAGE_KEY, String(normalized))
  }
})

watch(lineHeight, (next) => {
  const normalized = clampLineHeight(next)
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

watch(
  () => route.path,
  async () => {
    await nextTick()
    initNavigationSync()
    initMediumZoom()
    renderSidebarKatex()
    window.requestAnimationFrame(updateSidebarEdgePosition)
  }
)
</script>

<template>
  <DefaultTheme.Layout>
    <template v-if="showDocChrome" #nav-bar-content-after>
      <div class="ct-reading-tools">
        <button
          ref="readingToolsButton"
          class="ct-reading-tools-button"
          type="button"
          aria-label="阅读设置"
          :aria-expanded="readingToolsOpen ? 'true' : 'false'"
          @click="toggleReadingTools"
        >
          <span>Aa</span>
        </button>

        <Transition name="ct-reading-tools-fade">
          <div
            v-if="readingToolsOpen"
            ref="readingToolsPanel"
            class="ct-reading-tools-panel"
          >
            <div class="ct-reading-tools-group">
              <div class="ct-reading-tools-header">
                <div class="ct-reading-tools-title">字号</div>
                <div class="ct-reading-tools-value">{{ fontSize }}px</div>
              </div>
              <div class="ct-reading-tools-actions">
                <button
                  class="ct-reading-tools-action"
                  type="button"
                  @click="decreaseFontSize"
                >
                  A-
                </button>
                <button
                  class="ct-reading-tools-action"
                  type="button"
                  @click="resetFontSize"
                >
                  默认
                </button>
                <button
                  class="ct-reading-tools-action"
                  type="button"
                  @click="increaseFontSize"
                >
                  A+
                </button>
              </div>
              <input
                v-model="fontSize"
                class="ct-reading-tools-range"
                type="range"
                :min="MIN_FONT_SIZE"
                :max="MAX_FONT_SIZE"
                step="1"
              />
            </div>

            <div class="ct-reading-tools-group">
              <div class="ct-reading-tools-header">
                <div class="ct-reading-tools-title">行距</div>
                <div class="ct-reading-tools-value">
                  {{ lineHeight.toFixed(2) }}
                </div>
              </div>
              <div class="ct-reading-tools-actions">
                <button
                  class="ct-reading-tools-action"
                  type="button"
                  @click="lineHeight = clampLineHeight(lineHeight - 0.05)"
                >
                  更紧
                </button>
                <button
                  class="ct-reading-tools-action"
                  type="button"
                  @click="resetLineHeight"
                >
                  默认
                </button>
                <button
                  class="ct-reading-tools-action"
                  type="button"
                  @click="lineHeight = clampLineHeight(lineHeight + 0.05)"
                >
                  更松
                </button>
              </div>
              <input
                v-model="lineHeight"
                class="ct-reading-tools-range"
                type="range"
                :min="MIN_LINE_HEIGHT"
                :max="MAX_LINE_HEIGHT"
                step="0.05"
              />
            </div>
          </div>
        </Transition>
      </div>
    </template>

    <template v-if="isHomePage && homeTypingText" #home-hero-info-after>
      <div class="ct-home-typed-tagline">
        <ClientOnly>
          <TextType :text="homeTypingText" v-bind="homeTaglineTyping" />
        </ClientOnly>
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
</template>

<style>
.ct-reading-tools {
  position: relative;
  margin-left: 14px;
}

.ct-reading-tools-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 34px;
  min-width: 34px;
  padding: 0 12px;
  border: 1px solid rgba(0, 0, 0, 0.1);
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.95);
  color: var(--vp-c-text-2);
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.02em;
  box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.14),
    0 3px 1px -2px rgba(0, 0, 0, 0.2),
    0 1px 5px 0 rgba(0, 0, 0, 0.12);
  cursor: pointer;
  transition:
    color 0.2s ease,
    border-color 0.2s ease,
    box-shadow 0.2s ease;
}

.ct-reading-tools-button:hover {
  border-color: rgba(15, 118, 110, 0.4);
  color: var(--vp-c-brand-1);
  box-shadow: 0 3px 4px 0 rgba(0, 0, 0, 0.14),
    0 3px 3px -2px rgba(0, 0, 0, 0.2),
    0 1px 8px 0 rgba(0, 0, 0, 0.12);
}

.ct-reading-tools-panel {
  position: absolute;
  top: calc(100% + 10px);
  right: 0;
  width: 280px;
  padding: 14px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.14),
    0 3px 1px -2px rgba(0, 0, 0, 0.2),
    0 1px 5px 0 rgba(0, 0, 0, 0.12);
  z-index: 40;
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
  border-radius: 4px;
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

.ct-reading-tools-range {
  width: 100%;
  accent-color: var(--vp-c-brand-1);
}

.ct-reading-tools-fade-enter-active,
.ct-reading-tools-fade-leave-active {
  transition:
    opacity 0.18s ease,
    transform 0.18s ease;
}

.ct-reading-tools-fade-enter-from,
.ct-reading-tools-fade-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

.ct-sidebar-hover-area {
  display: none;
  position: fixed;
  top: 0;
  left: calc(
    var(--ct-sidebar-edge-right, var(--vp-sidebar-width, 272px)) - 14px
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
  box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.14),
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

.dark .ct-reading-tools-button,
.dark .ct-reading-tools-panel,
.dark .ct-sidebar-toggle-btn {
  border-color: rgba(255, 255, 255, 0.12);
  background: rgba(30, 30, 40, 0.92);
  box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.3),
    0 3px 1px -2px rgba(0, 0, 0, 0.4),
    0 1px 5px 0 rgba(0, 0, 0, 0.25);
}

.dark .ct-reading-tools-action {
  border-color: rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.04);
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
  .ct-reading-tools {
    margin-left: 10px;
  }

  .ct-reading-tools-panel {
    right: -6px;
    width: min(280px, calc(100vw - 24px));
  }
}

.medium-zoom-overlay {
  z-index: 999;
}

.medium-zoom-image--opened {
  z-index: 1000;
}

.main img {
  cursor: zoom-in;
  transition: transform 0.2s ease;
}

.main img:hover {
  transform: scale(1.01);
}
</style>
