/**
 * Correlation Network Graph (SVG)
 *
 * Interactive force-directed network visualization:
 * - Nodes = markets (size reflects uncertainty, i.e., closer to 50%)
 * - Edges = correlation strength (thickness + opacity)
 * - Click node → side panel with details
 * - Category-based coloring
 *
 * Uses a simple static force-directed layout computed on mount.
 */

import { useState, useMemo, useCallback, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { X, ExternalLink } from 'lucide-react'

interface CorrelationPair {
  market_a_id: number
  market_a_question: string
  market_a_platform: string
  market_b_id: number
  market_b_question: string
  market_b_platform: string
  correlation: number
}

interface MarketNode {
  id: number
  question: string
  platform: string
  category: string | null
}

interface CorrelationData {
  markets: MarketNode[]
  correlations: CorrelationPair[]
  total_pairs: number
}

interface LayoutNode extends MarketNode {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
}

interface Props {
  data: CorrelationData
}

const CATEGORY_COLORS: Record<string, string> = {
  politics: '#5EB4EF',
  crypto: '#F59E0B',
  sports: '#4CAF70',
  business: '#8B5CF6',
  science: '#EC4899',
  tech: '#06B6D4',
  other: '#737378',
}

function getCategoryColor(cat: string | null): string {
  if (!cat) return CATEGORY_COLORS.other
  return CATEGORY_COLORS[cat.toLowerCase()] ?? CATEGORY_COLORS.other
}

/**
 * Simple force-directed layout computed synchronously.
 * Runs N iterations to find stable positions.
 */
function computeLayout(
  markets: MarketNode[],
  correlations: CorrelationPair[],
  width: number,
  height: number,
): LayoutNode[] {
  const cx = width / 2
  const cy = height / 2

  // Initialize nodes in a circle
  const nodes: LayoutNode[] = markets.map((m, i) => {
    const angle = (2 * Math.PI * i) / markets.length
    const r = Math.min(width, height) * 0.3
    return {
      ...m,
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
      vx: 0,
      vy: 0,
      radius: 18,
    }
  })

  const idToIdx = new Map<number, number>()
  nodes.forEach((n, i) => idToIdx.set(n.id, i))

  // Build edge list for attraction
  const edges = correlations
    .map(c => ({
      source: idToIdx.get(c.market_a_id),
      target: idToIdx.get(c.market_b_id),
      strength: Math.abs(c.correlation),
    }))
    .filter(e => e.source !== undefined && e.target !== undefined) as {
      source: number
      target: number
      strength: number
    }[]

  // Run force simulation (100 iterations)
  const iterations = 120
  const repulsion = 8000
  const attraction = 0.08
  const centerGravity = 0.01
  const damping = 0.85

  for (let iter = 0; iter < iterations; iter++) {
    const temp = 1 - iter / iterations // Cool down over time

    // Repulsion between all pairs (Coulomb)
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[j].x - nodes[i].x
        const dy = nodes[j].y - nodes[i].y
        const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy))
        const force = (repulsion * temp) / (dist * dist)
        const fx = (dx / dist) * force
        const fy = (dy / dist) * force
        nodes[i].vx -= fx
        nodes[i].vy -= fy
        nodes[j].vx += fx
        nodes[j].vy += fy
      }
    }

    // Attraction along edges (Hooke)
    for (const edge of edges) {
      const a = nodes[edge.source]
      const b = nodes[edge.target]
      const dx = b.x - a.x
      const dy = b.y - a.y
      const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy))
      const force = attraction * edge.strength * dist * temp
      const fx = (dx / dist) * force
      const fy = (dy / dist) * force
      a.vx += fx
      a.vy += fy
      b.vx -= fx
      b.vy -= fy
    }

    // Center gravity
    for (const node of nodes) {
      node.vx += (cx - node.x) * centerGravity
      node.vy += (cy - node.y) * centerGravity
    }

    // Apply velocity + damping
    for (const node of nodes) {
      node.vx *= damping
      node.vy *= damping
      node.x += node.vx
      node.y += node.vy
      // Keep in bounds
      node.x = Math.max(30, Math.min(width - 30, node.x))
      node.y = Math.max(30, Math.min(height - 30, node.y))
    }
  }

  return nodes
}

export default function CorrelationGraph({ data }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const [selectedNode, setSelectedNode] = useState<number | null>(null)
  const [hoveredNode, setHoveredNode] = useState<number | null>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 })

  // Zoom and pan state
  const [transform, setTransform] = useState({ scale: 1, translateX: 0, translateY: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })

  // Measure container
  useEffect(() => {
    if (!containerRef.current) return
    const obs = new ResizeObserver(([entry]) => {
      setDimensions({
        width: entry.contentRect.width,
        height: Math.max(450, entry.contentRect.height),
      })
    })
    obs.observe(containerRef.current)
    return () => obs.disconnect()
  }, [])

  const nodes = useMemo(
    () => computeLayout(data.markets, data.correlations, dimensions.width, dimensions.height),
    [data, dimensions],
  )

  const idToNode = useMemo(() => {
    const map = new Map<number, LayoutNode>()
    nodes.forEach(n => map.set(n.id, n))
    return map
  }, [nodes])

  // Edges with positions
  const edges = useMemo(() => {
    return data.correlations
      .map(c => {
        const a = idToNode.get(c.market_a_id)
        const b = idToNode.get(c.market_b_id)
        if (!a || !b) return null
        return { ...c, ax: a.x, ay: a.y, bx: b.x, by: b.y }
      })
      .filter(Boolean) as (CorrelationPair & { ax: number; ay: number; bx: number; by: number })[]
  }, [data.correlations, idToNode])

  // Get correlations for selected node
  const selectedCorrelations = useMemo(() => {
    if (selectedNode === null) return []
    return data.correlations
      .filter(c => c.market_a_id === selectedNode || c.market_b_id === selectedNode)
      .sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))
  }, [data.correlations, selectedNode])

  const selectedMarket = selectedNode !== null ? idToNode.get(selectedNode) : null

  // Check if a node is connected to selected/hovered node
  const isConnected = useCallback((nodeId: number) => {
    const target = hoveredNode ?? selectedNode
    if (target === null) return true // No selection = all visible
    if (nodeId === target) return true
    return data.correlations.some(
      c => (c.market_a_id === target && c.market_b_id === nodeId) ||
           (c.market_b_id === target && c.market_a_id === nodeId)
    )
  }, [hoveredNode, selectedNode, data.correlations])

  const isEdgeHighlighted = useCallback((c: CorrelationPair) => {
    const target = hoveredNode ?? selectedNode
    if (target === null) return true
    return c.market_a_id === target || c.market_b_id === target
  }, [hoveredNode, selectedNode])

  // Zoom and pan handlers
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setTransform(prev => {
      const newScale = Math.max(0.5, Math.min(3, prev.scale * delta))
      return { ...prev, scale: newScale }
    })
  }, [])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    // Allow drag from SVG background or any non-interactive element (lines, circles without click handlers)
    // Only block if the target is a <g> with an onClick (i.e., a node group)
    const target = e.target as SVGElement
    const isNodeClick = target.closest?.('g[style*="cursor: pointer"]')
    if (e.button === 0 && !isNodeClick) {
      setIsDragging(true)
      setDragStart({ x: e.clientX - transform.translateX, y: e.clientY - transform.translateY })
    }
  }, [transform])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging) {
      setTransform(prev => ({
        ...prev,
        translateX: e.clientX - dragStart.x,
        translateY: e.clientY - dragStart.y,
      }))
    }
  }, [isDragging, dragStart])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  const resetView = useCallback(() => {
    setTransform({ scale: 1, translateX: 0, translateY: 0 })
  }, [])

  return (
    <div className="relative" ref={containerRef}>
      {/* Zoom Controls */}
      <div className="absolute top-3 right-3 flex flex-col gap-1 z-10">
        <button
          onClick={() => setTransform(prev => ({ ...prev, scale: Math.min(3, prev.scale * 1.2) }))}
          className="w-8 h-8 rounded-lg flex items-center justify-center text-[14px] font-bold"
          style={{ background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--text)' }}
          title="Zoom in"
        >
          +
        </button>
        <button
          onClick={() => setTransform(prev => ({ ...prev, scale: Math.max(0.5, prev.scale * 0.8) }))}
          className="w-8 h-8 rounded-lg flex items-center justify-center text-[14px] font-bold"
          style={{ background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--text)' }}
          title="Zoom out"
        >
          −
        </button>
        <button
          onClick={resetView}
          className="w-8 h-8 rounded-lg flex items-center justify-center text-[10px] font-bold"
          style={{ background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--text-3)' }}
          title="Reset view"
        >
          ⟲
        </button>
      </div>

      {/* SVG Graph */}
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        style={{ background: 'rgba(255,255,255,0.01)', borderRadius: '12px', cursor: isDragging ? 'grabbing' : 'grab' }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <g transform={`translate(${transform.translateX},${transform.translateY}) scale(${transform.scale})`}>
          {/* Edges */}
          {edges.map((e, i) => {
          const highlighted = isEdgeHighlighted(e)
          const thickness = 1 + Math.abs(e.correlation) * 4
          return (
            <line
              key={i}
              x1={e.ax}
              y1={e.ay}
              x2={e.bx}
              y2={e.by}
              stroke={e.correlation > 0 ? '#4CAF70' : '#CF6679'}
              strokeWidth={highlighted ? thickness : thickness * 0.5}
              strokeOpacity={highlighted ? 0.5 : 0.08}
              style={{ transition: 'stroke-opacity 0.2s, stroke-width 0.2s' }}
            />
          )
        })}

        {/* Nodes */}
        {nodes.map(node => {
          const connected = isConnected(node.id)
          const isSelected = node.id === selectedNode
          const isHovered = node.id === hoveredNode
          const color = getCategoryColor(node.category)
          const r = node.radius

          return (
            <g
              key={node.id}
              style={{
                cursor: 'pointer',
                opacity: connected ? 1 : 0.15,
                transition: 'opacity 0.2s',
              }}
              onClick={() => setSelectedNode(isSelected ? null : node.id)}
              onMouseEnter={() => setHoveredNode(node.id)}
              onMouseLeave={() => setHoveredNode(null)}
            >
              {/* Glow ring */}
              {(isSelected || isHovered) && (
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={r + 4}
                  fill="none"
                  stroke={isSelected ? '#C4A24D' : color}
                  strokeWidth={2}
                  strokeOpacity={0.5}
                />
              )}
              {/* Main circle */}
              <circle
                cx={node.x}
                cy={node.y}
                r={r}
                fill={color}
                fillOpacity={0.2}
                stroke={color}
                strokeWidth={1.5}
              />
              {/* Platform initial */}
              <text
                x={node.x}
                y={node.y + 1}
                textAnchor="middle"
                dominantBaseline="central"
                fill={color}
                fontSize="11"
                fontWeight="600"
                style={{ pointerEvents: 'none' }}
              >
                {node.platform.charAt(0).toUpperCase()}
              </text>
              {/* Label (on hover/select) */}
              {(isSelected || isHovered) && (
                <text
                  x={node.x}
                  y={node.y - r - 8}
                  textAnchor="middle"
                  fill="#fff"
                  fontSize="11"
                  fontWeight="500"
                  style={{ pointerEvents: 'none' }}
                >
                  {node.question.length > 30 ? node.question.slice(0, 30) + '...' : node.question}
                </text>
              )}
            </g>
          )
        })}
        </g>
      </svg>

      {/* Category Legend */}
      <div className="absolute top-3 left-3 flex flex-wrap gap-2">
        {Object.entries(CATEGORY_COLORS).map(([cat, color]) => {
          const hasNodes = nodes.some(n => (n.category?.toLowerCase() ?? 'other') === cat)
          if (!hasNodes) return null
          return (
            <div key={cat} className="flex items-center gap-1.5 text-[10px]" style={{ color: 'var(--text-3)' }}>
              <div className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
              <span className="capitalize">{cat}</span>
            </div>
          )
        })}
      </div>

      {/* Edge Legend */}
      <div className="absolute bottom-3 left-3 flex items-center gap-3 text-[10px]" style={{ color: 'var(--text-3)' }}>
        <span className="flex items-center gap-1">
          <div className="w-4 h-0.5 rounded" style={{ background: '#4CAF70' }} /> Positive
        </span>
        <span className="flex items-center gap-1">
          <div className="w-4 h-0.5 rounded" style={{ background: '#CF6679' }} /> Negative
        </span>
        <span>Line thickness = correlation strength</span>
      </div>

      {/* Side Panel (selected node) */}
      {selectedMarket && (
        <div
          className="absolute top-0 right-0 w-72 h-full overflow-y-auto p-4 space-y-3"
          style={{
            background: 'var(--card)',
            borderLeft: '1px solid var(--border)',
            borderRadius: '0 12px 12px 0',
          }}
        >
          {/* Close */}
          <div className="flex items-center justify-between">
            <span className="text-[10px] uppercase font-semibold" style={{ color: 'var(--text-3)' }}>
              Market Details
            </span>
            <button
              onClick={() => setSelectedNode(null)}
              className="p-1 rounded-lg"
              style={{ color: 'var(--text-3)' }}
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          {/* Market Info */}
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="pill pill-accent capitalize text-[10px]">{selectedMarket.platform}</span>
              <span className="pill text-[10px]">{selectedMarket.category ?? 'other'}</span>
            </div>
            <p className="text-[13px] font-medium leading-snug" style={{ color: 'var(--text)' }}>
              {selectedMarket.question}
            </p>
          </div>

          {/* View Market Link */}
          <Link
            to={`/markets/${selectedMarket.id}`}
            className="flex items-center gap-2 text-[12px] font-medium px-3 py-2 rounded-lg"
            style={{
              background: 'var(--accent-dim)',
              color: 'var(--accent)',
              textDecoration: 'none',
            }}
          >
            <ExternalLink className="h-3.5 w-3.5" />
            View Market
          </Link>

          {/* Correlated Markets */}
          <div>
            <p className="text-[10px] uppercase font-semibold mb-2" style={{ color: 'var(--text-3)' }}>
              Correlated Markets ({selectedCorrelations.length})
            </p>
            <div className="space-y-2">
              {selectedCorrelations.map((c, i) => {
                const isA = c.market_a_id === selectedNode
                const otherQ = isA ? c.market_b_question : c.market_a_question
                const otherP = isA ? c.market_b_platform : c.market_a_platform
                const otherId = isA ? c.market_b_id : c.market_a_id
                const corrColor = c.correlation > 0 ? 'var(--green)' : 'var(--red)'
                return (
                  <Link
                    key={i}
                    to={`/markets/${otherId}`}
                    className="block p-2 rounded-lg"
                    style={{
                      background: 'rgba(255,255,255,0.03)',
                      textDecoration: 'none',
                    }}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="pill text-[9px] capitalize">{otherP}</span>
                      <span className="text-[12px] font-bold font-mono" style={{ color: corrColor }}>
                        {(c.correlation * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="text-[11px] line-clamp-2" style={{ color: 'var(--text-2)' }}>
                      {otherQ}
                    </p>
                  </Link>
                )
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
