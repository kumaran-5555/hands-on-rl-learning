const keywordSet = new Set([
  'and',
  'as',
  'assert',
  'async',
  'await',
  'break',
  'class',
  'continue',
  'def',
  'del',
  'elif',
  'else',
  'except',
  'finally',
  'for',
  'from',
  'global',
  'if',
  'import',
  'in',
  'is',
  'lambda',
  'nonlocal',
  'not',
  'or',
  'pass',
  'raise',
  'return',
  'try',
  'while',
  'with',
  'yield'
])

const constantSet = new Set(['False', 'None', 'True'])

const builtinSet = new Set([
  'Exception',
  '__name__',
  'abs',
  'all',
  'any',
  'bool',
  'callable',
  'dict',
  'dir',
  'enumerate',
  'filter',
  'float',
  'getattr',
  'hasattr',
  'int',
  'isinstance',
  'iter',
  'len',
  'list',
  'map',
  'max',
  'min',
  'object',
  'open',
  'print',
  'property',
  'range',
  'repr',
  'set',
  'str',
  'sum',
  'super',
  'tuple',
  'type',
  'zip'
])

const operatorSet = new Set([
  '+',
  '-',
  '*',
  '**',
  '/',
  '//',
  '%',
  '=',
  '==',
  '!=',
  '<',
  '<=',
  '>',
  '>=',
  '+=',
  '-=',
  '*=',
  '/=',
  '->',
  ':=',
  '|'
])

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function wrap(className, value) {
  return `<span class="${className}">${escapeHtml(value)}</span>`
}

function isIdentifierStart(char) {
  return /[A-Za-z_]/.test(char)
}

function isIdentifierPart(char) {
  return /[A-Za-z0-9_]/.test(char)
}

function isDigit(char) {
  return /\d/.test(char)
}

function readString(line, index, stringStart) {
  const quote = line[stringStart]
  const triple = line.slice(stringStart, stringStart + 3) === quote.repeat(3)
  let end = stringStart + (triple ? 3 : 1)

  while (end < line.length) {
    if (line[end] === '\\') {
      end += 2
      continue
    }

    if (triple && line.slice(end, end + 3) === quote.repeat(3)) {
      end += 3
      break
    }

    if (!triple && line[end] === quote) {
      end += 1
      break
    }

    end += 1
  }

  return { type: 'string', value: line.slice(index, end) }
}

function getStringStart(line, index) {
  const threeChars = line.slice(index, index + 3).toLowerCase()
  const twoChars = line.slice(index, index + 2).toLowerCase()

  if (['fr"', "fr'", 'rf"', "rf'"].includes(threeChars)) return index + 2
  if (['r"', "r'", 'f"', "f'", 'b"', "b'", 'u"', "u'"].includes(twoChars)) {
    return index + 1
  }
  if (line[index] === '"' || line[index] === "'") return index
  return -1
}

function tokenizePythonLine(line) {
  const tokens = []
  let index = 0

  while (index < line.length) {
    const char = line[index]

    if (/\s/.test(char)) {
      let end = index + 1
      while (end < line.length && /\s/.test(line[end])) end += 1
      tokens.push({ type: 'space', value: line.slice(index, end) })
      index = end
      continue
    }

    if (char === '#') {
      tokens.push({ type: 'comment', value: line.slice(index) })
      break
    }

    const stringStart = getStringStart(line, index)
    if (stringStart >= 0) {
      const token = readString(line, index, stringStart)
      tokens.push(token)
      index += token.value.length
      continue
    }

    if (isDigit(char)) {
      let end = index + 1
      while (
        end < line.length &&
        /[A-Za-z0-9_.]/.test(line[end]) &&
        !(line[end] === '.' && line[end + 1] === '.')
      ) {
        end += 1
      }
      tokens.push({ type: 'number', value: line.slice(index, end) })
      index = end
      continue
    }

    if (isIdentifierStart(char)) {
      let end = index + 1
      while (end < line.length && isIdentifierPart(line[end])) end += 1
      tokens.push({ type: 'identifier', value: line.slice(index, end) })
      index = end
      continue
    }

    const twoChar = line.slice(index, index + 2)
    if (operatorSet.has(twoChar)) {
      tokens.push({ type: 'operator', value: twoChar })
      index += 2
      continue
    }

    tokens.push({
      type: operatorSet.has(char) ? 'operator' : 'punctuation',
      value: char
    })
    index += 1
  }

  return tokens
}

function previousSignificant(tokens, index) {
  for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
    if (tokens[cursor].type !== 'space') return tokens[cursor]
  }
  return null
}

function nextSignificant(tokens, index) {
  for (let cursor = index + 1; cursor < tokens.length; cursor += 1) {
    if (tokens[cursor].type !== 'space') return tokens[cursor]
  }
  return null
}

function classifyIdentifier(token, tokens, index) {
  const previous = previousSignificant(tokens, index)
  const next = nextSignificant(tokens, index)

  if (constantSet.has(token)) return 'py-constant'
  if (keywordSet.has(token)) return 'py-keyword'
  if (previous?.value === '@') return 'py-decorator'
  if (previous?.value === 'class') return 'py-class'
  if (previous?.value === 'def') return 'py-function'
  if (previous?.value === '.') {
    return next?.value === '(' ? 'py-function' : 'py-property'
  }
  if (previous?.value === 'import' || previous?.value === 'from') {
    return /^[A-Z][A-Za-z0-9_]+$/.test(token) ? 'py-class' : 'py-module'
  }
  if (token === 'self' || token === 'cls') return 'py-self'
  if (builtinSet.has(token)) return 'py-builtin'
  if (next?.value === '(') return 'py-function'
  if (/^[A-Z][A-Za-z0-9_]+$/.test(token)) return 'py-class'

  return 'py-variable'
}

function renderToken(token, tokens, index) {
  if (token.type === 'space' || token.type === 'punctuation') {
    return escapeHtml(token.value)
  }

  if (token.type === 'identifier') {
    return wrap(classifyIdentifier(token.value, tokens, index), token.value)
  }

  if (token.type === 'operator') return wrap('py-operator', token.value)
  return wrap(`py-${token.type}`, token.value)
}

export function highlightPython(line) {
  if (!line) return '&nbsp;'

  const tokens = tokenizePythonLine(line)
  return tokens
    .map((token, index) => renderToken(token, tokens, index))
    .join('')
}

export function scrollCodeLineIntoView(
  container,
  lineNumber,
  { behavior = 'smooth' } = {}
) {
  if (!container || !lineNumber) return

  const line = container.querySelector(`[data-line-number="${lineNumber}"]`)
  if (!line) return

  const targetTop = Math.max(0, line.offsetTop - container.clientHeight * 0.28)

  container.scrollTo({
    top: targetTop,
    behavior
  })
}
