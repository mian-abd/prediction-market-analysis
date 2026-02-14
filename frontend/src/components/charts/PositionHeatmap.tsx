/**
 * Position Heatmap (Treemap)
 *
 * Visual portfolio concentration map:
 * - Each rectangle = 1 open position
 * - Size = notional value
 * - Color = unrealized P&L
 * - Hover shows full details
 */

import { useEffect, useState } from 'react'
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts'
import { Loader2, AlertCircle, Layout } from 'lucide-react'
import apiClient from '../../api/client'
import EmptyState from '../EmptyState'

interface PositionData {
  id: number
  market_id: number
  question: string
  platform: string
  side: string
  entry_price: number
  current_price: number
  quantity: number
  entry_time: string
  unrealized_pnl: number
  unrealized_pnl_pct: number
  notional_value: number
  days_held: number
}

interface TreemapNode {
  name: string
  size: number
  pnl_pct: number
  position: PositionData
  [key: string]: unknown
}

export default function PositionHeatmap() {
  const [data, setData] = useState<TreemapNode[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchPositions = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await apiClient.get('/portfolio/positions?status=open&limit=100')
      const positions = response.data.positions || []

      if (positions.length === 0) {
        setData([])
        return
      }

      // Convert to treemap format
      const treemapData = positions.map((pos: any) => {
        const notionalValue = pos.quantity * pos.entry_price
        const unrealizedPnlPct =
          pos.unrealized_pnl && notionalValue > 0
            ? (pos.unrealized_pnl / notionalValue) * 100
            : 0

        // Calculate days held
        const entryDate = new Date(pos.entry_time)
        const now = new Date()
        const daysHeld = Math.floor((now.getTime() - entryDate.getTime()) / (1000 * 60 * 60 * 24))

        return {
          name: pos.question.slice(0, 40) + (pos.question.length > 40 ? '...' : ''),
          size: notionalValue,
          pnl_pct: unrealizedPnlPct,
          position: {
            ...pos,
            unrealized_pnl_pct: unrealizedPnlPct,
            notional_value: notionalValue,
            days_held: daysHeld,
          },
        }
      })

      setData(treemapData)
    } catch (err: any) {
      setError('Failed to load positions')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPositions()

    // Auto-refresh every 15 seconds
    const interval = setInterval(fetchPositions, 15_000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-6 w-6 animate-spin" style={{ color: 'var(--text-3)' }} />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex flex-col items-center justify-center h-96 gap-3">
        <AlertCircle className="h-6 w-6" style={{ color: 'var(--red)' }} />
        <p className="text-[14px]" style={{ color: 'var(--text-3)' }}>
          {error || 'No position data'}
        </p>
      </div>
    )
  }

  if (data.length === 0) {
    return (
      <EmptyState
        icon={Layout}
        title="No open positions"
        message="Your portfolio heatmap will appear here once you start trading."
      />
    )
  }

  // Color function based on P&L %
  const getColor = (pnlPct: number): string => {
    if (pnlPct > 10) return '#4CAF70' // Dark green (big win)
    if (pnlPct > 5) return '#66BB6A' // Medium green
    if (pnlPct > 0) return '#81C784' // Light green
    if (pnlPct > -5) return '#EF9A9A' // Light red
    if (pnlPct > -10) return '#E57373' // Medium red
    return '#CF6679' // Dark red (big loss)
  }

  // Custom content renderer for treemap cells
  const CustomizedContent = (props: any) => {
    const { x, y, width, height, name, pnl_pct } = props

    if (width < 40 || height < 40) {
      // Too small to render text
      return (
        <g>
          <rect
            x={x}
            y={y}
            width={width}
            height={height}
            style={{
              fill: getColor(pnl_pct),
              stroke: 'var(--bg)',
              strokeWidth: 2,
            }}
          />
        </g>
      )
    }

    return (
      <g>
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          style={{
            fill: getColor(pnl_pct),
            stroke: 'var(--bg)',
            strokeWidth: 2,
          }}
        />
        <text
          x={x + width / 2}
          y={y + height / 2 - 8}
          textAnchor="middle"
          fill="#FFF"
          fontSize="11"
          fontWeight="500"
        >
          {name}
        </text>
        <text
          x={x + width / 2}
          y={y + height / 2 + 8}
          textAnchor="middle"
          fill="#FFF"
          fontSize="14"
          fontWeight="bold"
        >
          {pnl_pct > 0 ? '+' : ''}
          {pnl_pct.toFixed(1)}%
        </text>
      </g>
    )
  }

  return (
    <div className="space-y-4">
      {/* Color legend */}
      <div className="flex items-center justify-between">
        <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
          {data.length} open position{data.length !== 1 ? 's' : ''}
        </p>
        <div className="flex items-center gap-2 text-[11px]">
          <span style={{ color: 'var(--text-3)' }}>P&L:</span>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 rounded" style={{ background: '#CF6679' }} />
            <span style={{ color: 'var(--text-3)' }}>-10%</span>
          </div>
          <div className="w-4 h-4 rounded" style={{ background: '#E57373' }} />
          <div className="w-4 h-4 rounded" style={{ background: '#EF9A9A' }} />
          <div className="w-4 h-4 rounded" style={{ background: '#81C784' }} />
          <div className="w-4 h-4 rounded" style={{ background: '#66BB6A' }} />
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 rounded" style={{ background: '#4CAF70' }} />
            <span style={{ color: 'var(--text-3)' }}>+10%</span>
          </div>
        </div>
      </div>

      {/* Treemap */}
      <div style={{ height: '384px', minHeight: '384px', width: '100%', background: 'var(--card)', borderRadius: '12px', padding: '4px' }}>
        <ResponsiveContainer width="100%" height={384}>
          <Treemap
            data={data}
            dataKey="size"
            aspectRatio={16 / 9}
            stroke="var(--bg)"
            fill="#8884d8"
            content={<CustomizedContent />}
          >
            <Tooltip
              contentStyle={{
                backgroundColor: '#1A1A1C',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '12px',
                color: '#FFF',
                fontSize: '12px',
                padding: '10px',
              }}
              formatter={((_value: number | undefined, name: string | undefined, props: any) => {
                const { payload } = props
                if (!payload || !payload.position) return null

                const pos = payload.position
                return [
                  <div key={name} className="space-y-1.5">
                    <div className="font-semibold text-[13px] mb-2" style={{ color: 'var(--text)' }}>
                      {pos.question}
                    </div>
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Platform:</span>
                      <span className="uppercase">{pos.platform}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Side:</span>
                      <span className="uppercase">{pos.side}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Entry:</span>
                      <span className="font-mono">{pos.entry_price.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Current:</span>
                      <span className="font-mono">{(pos.current_price || 0).toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Notional:</span>
                      <span className="font-mono">${pos.notional_value.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Unrealized P&L:</span>
                      <span
                        className="font-mono font-semibold"
                        style={{ color: pos.unrealized_pnl >= 0 ? 'var(--green)' : 'var(--red)' }}
                      >
                        ${(pos.unrealized_pnl || 0).toFixed(2)} ({pos.unrealized_pnl_pct >= 0 ? '+' : ''}
                        {pos.unrealized_pnl_pct.toFixed(1)}%)
                      </span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Days Held:</span>
                      <span>{pos.days_held}</span>
                    </div>
                  </div>,
                ]
              }) as any}
            />
          </Treemap>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
