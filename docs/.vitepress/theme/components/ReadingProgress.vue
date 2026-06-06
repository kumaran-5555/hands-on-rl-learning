<template>
  <Transition name="ct-progress-fade">
    <div
      v-if="showProgress"
      class="ct-reading-progress"
      :class="{ 'is-dragging': isDragging }"
      :title="isDragging ? '拖动调整位置' : `阅读进度 ${progress}%`"
      @mousedown="startDrag"
      @touchstart="startDrag"
      @click="handleClick"
    >
      <svg class="ct-progress-ring" viewBox="0 0 56 56">
        <circle class="ct-progress-ring-bg" cx="28" cy="28" r="24" />
        <circle
          class="ct-progress-ring-circle"
          cx="28"
          cy="28"
          r="24"
          :style="{
            strokeDashoffset: circumference - (progress / 100) * circumference
          }"
        />
      </svg>

      <Transition name="ct-progress-switch">
        <div
          v-if="showArrow && !isDragging"
          key="arrow"
          class="ct-progress-arrow"
        >
          ↑
        </div>
        <div v-else key="percent" class="ct-progress-text">{{ progress }}%</div>
      </Transition>

      <div v-if="isDragging" class="ct-progress-hint">拖动调整</div>
    </div>
  </Transition>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from 'vue'

const progress = ref(0)
const showProgress = ref(false)
const showArrow = ref(false)
const isDragging = ref(false)
const didDrag = ref(false)

const circumference = 2 * Math.PI * 24

let scrollTimer = null
let dragFrame = null
let dragStartY = 0
let dragStartProgress = 0

function updateProgress() {
  if (isDragging.value) return

  const scrollTop = window.scrollY
  const docHeight = document.documentElement.scrollHeight - window.innerHeight
  const scrollPercent = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0

  progress.value = Math.min(Math.round(scrollPercent), 100)
  showProgress.value = scrollTop > 0
  showArrow.value = false

  if (scrollTimer) {
    clearTimeout(scrollTimer)
  }

  scrollTimer = window.setTimeout(() => {
    if (window.scrollY > 0) {
      showArrow.value = true
    }
  }, 1200)
}

function startDrag(event) {
  event.preventDefault()

  isDragging.value = true
  didDrag.value = false
  dragStartY = 'touches' in event ? event.touches[0].clientY : event.clientY
  dragStartProgress = progress.value

  document.addEventListener('mousemove', onDrag, { passive: false })
  document.addEventListener('mouseup', endDrag)
  document.addEventListener('touchmove', onDrag, { passive: false })
  document.addEventListener('touchend', endDrag)
}

function onDrag(event) {
  if (!isDragging.value) return

  event.preventDefault()

  const currentY = 'touches' in event ? event.touches[0].clientY : event.clientY
  const deltaY = dragStartY - currentY
  const nextProgress = Math.max(
    0,
    Math.min(100, dragStartProgress + deltaY / 3)
  )

  if (Math.abs(deltaY) > 4) {
    didDrag.value = true
  }

  if (dragFrame) {
    cancelAnimationFrame(dragFrame)
  }

  dragFrame = requestAnimationFrame(() => {
    progress.value = Math.round(nextProgress)
    const docHeight = document.documentElement.scrollHeight - window.innerHeight

    if (docHeight > 0) {
      window.scrollTo({
        top: (progress.value / 100) * docHeight,
        behavior: 'auto'
      })
    }
  })
}

function endDrag() {
  isDragging.value = false

  document.removeEventListener('mousemove', onDrag)
  document.removeEventListener('mouseup', endDrag)
  document.removeEventListener('touchmove', onDrag)
  document.removeEventListener('touchend', endDrag)

  if (dragFrame) {
    cancelAnimationFrame(dragFrame)
    dragFrame = null
  }

  if (window.scrollY > 0) {
    showArrow.value = true
  }

  window.setTimeout(() => {
    didDrag.value = false
  }, 80)
}

function handleClick() {
  if (isDragging.value || didDrag.value) return

  window.scrollTo({
    top: 0,
    behavior: 'smooth'
  })
}

onMounted(() => {
  window.addEventListener('scroll', updateProgress, { passive: true })
  updateProgress()
})

onUnmounted(() => {
  window.removeEventListener('scroll', updateProgress)

  if (scrollTimer) {
    clearTimeout(scrollTimer)
  }

  document.removeEventListener('mousemove', onDrag)
  document.removeEventListener('mouseup', endDrag)
  document.removeEventListener('touchmove', onDrag)
  document.removeEventListener('touchend', endDrag)

  if (dragFrame) {
    cancelAnimationFrame(dragFrame)
  }
})
</script>

<style scoped>
.ct-reading-progress {
  position: fixed;
  right: 28px;
  bottom: 28px;
  width: 56px;
  height: 56px;
  z-index: 70;
  cursor: grab;
  user-select: none;
  touch-action: none;
  filter: drop-shadow(0 14px 28px rgba(15, 23, 42, 0.16));
  transition: transform 0.24s ease;
}

.ct-reading-progress:hover {
  transform: scale(1.08);
}

.ct-reading-progress.is-dragging {
  cursor: grabbing;
  transform: scale(1.12);
}

.ct-progress-ring {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  transform: rotate(-90deg);
}

.ct-progress-ring-bg {
  fill: rgba(255, 255, 255, 0.92);
  stroke: rgba(148, 163, 184, 0.28);
  stroke-width: 3;
}

.ct-progress-ring-circle {
  fill: none;
  stroke: var(--vp-c-brand-1);
  stroke-width: 3;
  stroke-linecap: round;
  stroke-dasharray: 150.796;
  transition: stroke-dashoffset 0.1s ease;
}

.ct-reading-progress.is-dragging .ct-progress-ring-circle {
  transition: none;
}

.ct-progress-text,
.ct-progress-arrow {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  pointer-events: none;
}

.ct-progress-text {
  font-size: 12px;
  font-weight: 700;
  color: var(--vp-c-text-1);
}

.ct-progress-arrow {
  font-size: 24px;
  font-weight: 800;
  color: var(--vp-c-brand-1);
  animation: ct-progress-bounce 1s ease-in-out infinite;
}

.ct-progress-hint {
  position: absolute;
  left: 50%;
  bottom: calc(100% + 8px);
  transform: translateX(-50%);
  padding: 4px 8px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.94);
  color: var(--vp-c-text-2);
  font-size: 11px;
  white-space: nowrap;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
}

.ct-progress-fade-enter-active,
.ct-progress-fade-leave-active,
.ct-progress-switch-enter-active,
.ct-progress-switch-leave-active {
  transition:
    opacity 0.2s ease,
    transform 0.2s ease;
}

.ct-progress-fade-enter-from,
.ct-progress-fade-leave-to,
.ct-progress-switch-enter-from,
.ct-progress-switch-leave-to {
  opacity: 0;
  transform: translateY(6px);
}

@keyframes ct-progress-bounce {
  0%,
  100% {
    transform: translate(-50%, -50%);
  }

  50% {
    transform: translate(-50%, -60%);
  }
}

.dark .ct-progress-ring-bg {
  fill: rgba(15, 23, 42, 0.94);
  stroke: rgba(148, 163, 184, 0.28);
}

.dark .ct-progress-text {
  color: #f8fafc;
  text-shadow: 0 1px 2px rgba(2, 6, 23, 0.55);
}

.dark .ct-progress-hint {
  border-color: rgba(148, 163, 184, 0.18);
  background: rgba(15, 23, 42, 0.88);
  color: #e2e8f0;
}

@media (max-width: 768px) {
  .ct-reading-progress {
    right: 18px;
    bottom: 18px;
    width: 48px;
    height: 48px;
  }

  .ct-progress-text {
    font-size: 11px;
  }

  .ct-progress-arrow {
    font-size: 21px;
  }
}
</style>
