/**
 * Equity Curve Chart
 *
 * Displays cumulative P&L over time with:
 * - Multiple strategy lines
 * - Trade markers (entry/exit dots with hover details)
 * - Toggle between $ and % returns
 */

import { useEffect, useState, useCallback } from 'react'
import {
  ComposedChart,
  Line,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
  Cell,
} from 'recharts'
import { Loader2, AlertCircle, TrendingUp, TrendingDown } from 'lucide-react'
import apiClient from '../../api/client'
import EmptyState from '../EmptyState'

interface EquityPoint {
  timestamp: string
  cumulative_pnl: number
}

interface StrategyData {
  name: string
  data: EquityPoint[]
  final_pnl: number
}

interface TradeEvent {
  timestamp: string
  type: 'entry' | 'exit'
  side: string
  strategy: string
  market: string
  price: number
  quantity: number
  pnl: number | null
}

interface EquityCurveData {
  data: StrategyData[]
  strategies: string[]
  total_pnl: number
  trade_events?: TradeEvent[]
}

interface EquityCurveProps {
  timeRange?: '7d' | '30d' | '90d' | 'all'
  showDrawdown?: boolean
  autoRefresh?: boolean
  portfolioType?: 'all' | 'manual' | 'auto'
}

const STRATEGY_COLORS: Record<string, string> = {
  all: '#C4A24D', // Accent
  single_market_arb: '#4CAF70', // Green
  cross_platform_arb: '#5EB4EF', // Blue
  calibration: '#B27BCC', // Purple
  manual: '#8E8E93', // Gray
}

function TradeMarkerDot(props: any) {
  const { cx, cy, payload } = props
  if (!cx || !cy || !payload?.tradeType) return null

  const isExit = payload.tradeType === 'exit'
  const isWin = isExit && payload.tradePnl != null && payload.tradePnl >= 0
  const isLoss = isExit && payload.tradePnl != null && payload.tradePnl < 0

  const color = isWin ? '#4CAF70' : isLoss ? '#EF5350' : '#C4A24D'
  const size = isExit ? 6 : 4

  return (
    <g>
      <circle cx={cx} cy={cy} r={size} fill={color} fillOpacity={0.9} stroke="#000" strokeWidth={1.5} />
      {!isExit && (
        // Entry marker: small upward triangle
        <polygon
          points={`${cx},${cy - 8} ${cx - 4},${cy - 3} ${cx + 4},${cy - 3}`}
          fill={color}
          fillOpacity={0.7}
        />
      )}
    </g>
  )
}

function TradeTooltipContent({ active, payload, label }: any) {
  if (!active || !payload?.length) return null

  const tradePayload = payload.find((p: any) => p.dataKey === 'tradeMarker')

  return (
    <div
      style={{
        backgroundColor: '#1A1A1C',
        border: '1px solid rgba(255,255,255,0.12)',
        borderRadius: '12px',
        padding: '10px 14px',
        fontSize: '12px',
        color: '#FFF',
        maxWidth: '280px',
      }}
    >
      <p style={{ color: '#8E8E93', marginBottom: '6px' }}>{label}</p>

      {/* Equity values */}
      {payload
        .filter((p: any) => p.dataKey !== 'tradeMarker' && p.value != null)
        .map((p: any) => {
          const val = typeof p.value === 'number' ? p.value : Number(p.value) || 0
          return (
            <div key={p.dataKey} style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', marginBottom: '2px' }}>
              <span style={{ color: '#8E8E93' }}>
                {p.dataKey === 'all' ? 'Total' : p.dataKey.replace(/_/g, ' ')}
              </span>
              <span className="font-mono" style={{ color: val >= 0 ? '#4CAF70' : '#EF5350' }}>
                ${val.toFixed(2)}
              </span>
            </div>
          )
        })}

      {/* Trade event details */}
      {tradePayload?.payload?.tradeType && (
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', marginTop: '6px', paddingTop: '6px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
            <span
              style={{
                display: 'inline-block',
                padding: '1px 6px',
                borderRadius: '4px',
                fontSize: '10px',
                fontWeight: 600,
                textTransform: 'uppercase',
                background: tradePayload.payload.tradeType === 'entry' ? 'rgba(196,162,77,0.2)' : tradePayload.payload.tradePnl >= 0 ? 'rgba(76,175,112,0.2)' : 'rgba(239,83,80,0.2)',
                color: tradePayload.payload.tradeType === 'entry' ? '#C4A24D' : tradePayload.payload.tradePnl >= 0 ? '#4CAF70' : '#EF5350',
              }}
            >
              {tradePayload.payload.tradeType}
            </span>
            <span style={{ color: '#8E8E93', fontSize: '10px' }}>
              {tradePayload.payload.tradeSide?.toUpperCase()} · {tradePayload.payload.tradeStrategy?.replace(/_/g, ' ')}
            </span>
          </div>
          {tradePayload.payload.tradeMarket && (
            <p style={{ color: '#CCC', fontSize: '11px', lineHeight: '1.3', marginBottom: '4px' }}>
              {tradePayload.payload.tradeMarket}
            </p>
          )}
          <div style={{ display: 'flex', gap: '12px', fontSize: '11px' }}>
            <span style={{ color: '#8E8E93' }}>
              @ ${tradePayload.payload.tradePrice?.toFixed(2)} × {tradePayload.payload.tradeQty}
            </span>
            {tradePayload.payload.tradePnl != null && (
              <span className="font-mono" style={{ fontWeight: 600, color: tradePayload.payload.tradePnl >= 0 ? '#4CAF70' : '#EF5350' }}>
                {tradePayload.payload.tradePnl >= 0 ? '+' : ''}${tradePayload.payload.tradePnl.toFixed(2)}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default function EquityCurve({
  timeRange = '30d',
  showDrawdown: _showDrawdown = false,
  autoRefresh = false,
  portfolioType = 'all',
}: EquityCurveProps) {
  const [data, setData] = useState<EquityCurveData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showMarkers, setShowMarkers] = useState(true)

  const fetchEquityCurve = useCallback(async (showSpinner = false) => {
    if (showSpinner) setLoading(true)
    try {
      // Don't send portfolio_type param when 'all' is selected (API doesn't accept 'all')
      const ptParam = (portfolioType && portfolioType !== 'all') ? `&portfolio_type=${portfolioType}` : ''
      const response = await apiClient.get(`/portfolio/equity-curve?time_range=${timeRange}${ptParam}`)
      setData(response.data)
      setError(null)
    } catch (err: any) {
      setError('Failed to load equity curve')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [timeRange, portfolioType])

  // Re-fetch with spinner when time range or portfolio type changes
  useEffect(() => {
    fetchEquityCurve(true)
  }, [timeRange, portfolioType]) // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-refresh silently (no spinner) every 15 seconds for live P&L tracking
  useEffect(() => {
    if (!autoRefresh) return
    const interval = setInterval(() => fetchEquityCurve(false), 15_000)
    return () => clearInterval(interval)
  }, [fetchEquityCurve, autoRefresh])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-80">
        <Loader2 className="h-6 w-6 animate-spin" style={{ color: 'var(--text-3)' }} />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex flex-col items-center justify-center h-80 gap-3">
        <AlertCircle className="h-6 w-6" style={{ color: 'var(--red)' }} />
        <p className="text-[14px]" style={{ color: 'var(--text-3)' }}>
          {error || 'No equity data'}
        </p>
      </div>
    )
  }

  if (data.data.length === 0) {
    return (
      <EmptyState
        icon={TrendingUp}
        title="No trading history yet"
        message="Your equity curve will appear here once you start trading."
      />
    )
  }

  // Prepare chart data: merge all strategy timelines
  const chartDataMap = new Map<string, any>()

  data.data.forEach((strategy) => {
    strategy.data.forEach((point) => {
      const timestamp = point.timestamp
      if (!chartDataMap.has(timestamp)) {
        chartDataMap.set(timestamp, { timestamp })
      }
      chartDataMap.get(timestamp)![strategy.name] = point.cumulative_pnl
    })
  })

  // Merge trade events into chart data as marker fields
  const tradeEvents = data.trade_events || []
  tradeEvents.forEach((event) => {
    const ts = event.timestamp
    if (!chartDataMap.has(ts)) {
      chartDataMap.set(ts, { timestamp: ts })
    }
    const point = chartDataMap.get(ts)!
    // Store trade marker at the 'all' cumulative value (resolved below after forward-fill)
    point.tradeType = event.type
    point.tradeSide = event.side
    point.tradeStrategy = event.strategy
    point.tradeMarket = event.market
    point.tradePrice = event.price
    point.tradeQty = event.quantity
    point.tradePnl = event.pnl
  })

  // Convert to array and sort by timestamp
  const chartData = Array.from(chartDataMap.values()).sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  )

  // Determine time span to choose appropriate x-axis labels
  const firstTs = chartData.length > 0 ? new Date(chartData[0].timestamp).getTime() : 0
  const lastTs = chartData.length > 0 ? new Date(chartData[chartData.length - 1].timestamp).getTime() : 0
  const spanMs = lastTs - firstTs
  const spanDays = spanMs / (1000 * 60 * 60 * 24)
  const isIntraday = spanDays < 2

  // Format dates based on time span
  chartData.forEach((point) => {
    const d = new Date(point.timestamp)
    if (isIntraday) {
      point.date = d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
    } else {
      point.date = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    }
  })

  // Forward-fill missing values (carry last value forward)
  const strategyNames = data.strategies
  const lastValues: Record<string, number> = {}
  chartData.forEach((point) => {
    strategyNames.forEach((strategy) => {
      if (point[strategy] !== undefined) {
        lastValues[strategy] = point[strategy]
      } else if (lastValues[strategy] !== undefined) {
        point[strategy] = lastValues[strategy]
      }
    })
  })

  // Set trade marker Y position to the 'all' equity value at that point
  chartData.forEach((point) => {
    if (point.tradeType) {
      point.tradeMarker = point.all ?? 0
    }
  })

  // Count trades
  const totalTrades = tradeEvents.length
  const exitEvents = tradeEvents.filter((e) => e.type === 'exit')
  const wins = exitEvents.filter((e) => e.pnl != null && e.pnl >= 0).length
  const winRate = exitEvents.length > 0 ? ((wins / exitEvents.length) * 100).toFixed(0) : '—'

  // Calculate metrics
  const totalPnL = data.total_pnl
  const isPositive = totalPnL >= 0

  return (
    <div className="space-y-4">
      {/* Header with metrics */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-4">
          <div>
            <p className="text-[10px] uppercase mb-0.5" style={{ color: 'var(--text-3)' }}>
              Total P&L
            </p>
            <div className="flex items-center gap-2">
              {isPositive ? (
                <TrendingUp className="h-4 w-4" style={{ color: 'var(--green)' }} />
              ) : (
                <TrendingDown className="h-4 w-4" style={{ color: 'var(--red)' }} />
              )}
              <p
                className="text-[20px] font-mono font-bold"
                style={{ color: isPositive ? 'var(--green)' : 'var(--red)' }}
              >
                ${totalPnL.toFixed(2)}
              </p>
            </div>
          </div>

          <div>
            <p className="text-[10px] uppercase mb-0.5" style={{ color: 'var(--text-3)' }}>
              Trades
            </p>
            <p className="text-[18px] font-semibold" style={{ color: 'var(--text)' }}>
              {Math.floor(totalTrades / 2)}
            </p>
          </div>

          <div>
            <p className="text-[10px] uppercase mb-0.5" style={{ color: 'var(--text-3)' }}>
              Win Rate
            </p>
            <p className="text-[18px] font-semibold" style={{ color: 'var(--text)' }}>
              {winRate}%
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Trade markers toggle */}
          <button
            onClick={() => setShowMarkers(!showMarkers)}
            className="text-[11px] px-2.5 py-1 rounded-md transition-colors"
            style={{
              background: showMarkers ? 'rgba(196,162,77,0.15)' : 'rgba(255,255,255,0.05)',
              color: showMarkers ? '#C4A24D' : '#8E8E93',
              border: `1px solid ${showMarkers ? 'rgba(196,162,77,0.3)' : 'rgba(255,255,255,0.08)'}`,
            }}
          >
            Trades
          </button>

          {/* Strategy breakdown */}
          <div className="flex items-center gap-3 text-[12px]">
            {data.data
              .filter((s) => s.name !== 'all')
              .map((strategy) => (
                <div key={strategy.name} className="flex items-center gap-1.5">
                  <div
                    className="w-3 h-3 rounded"
                    style={{ background: STRATEGY_COLORS[strategy.name] || '#8E8E93' }}
                  />
                  <span style={{ color: 'var(--text-3)' }}>
                    {strategy.name.replace(/_/g, ' ')}:
                  </span>
                  <span
                    className="font-mono font-medium"
                    style={{ color: strategy.final_pnl >= 0 ? 'var(--green)' : 'var(--red)' }}
                  >
                    ${strategy.final_pnl.toFixed(2)}
                  </span>
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* Equity curve chart */}
      <div style={{ width: '100%', height: '320px', minHeight: '320px' }}>
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              {strategyNames.map((strategy) => (
                <linearGradient
                  key={`gradient-${strategy}`}
                  id={`gradient-${strategy}`}
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop
                    offset="5%"
                    stopColor={STRATEGY_COLORS[strategy] || '#8E8E93'}
                    stopOpacity={0.2}
                  />
                  <stop
                    offset="95%"
                    stopColor={STRATEGY_COLORS[strategy] || '#8E8E93'}
                    stopOpacity={0}
                  />
                </linearGradient>
              ))}
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />

            <XAxis
              dataKey="date"
              stroke="rgba(255,255,255,0.06)"
              tick={{ fill: '#48484A', fontSize: 11 }}
            />

            <YAxis
              stroke="rgba(255,255,255,0.06)"
              tick={{ fill: '#48484A', fontSize: 11 }}
              tickFormatter={(value) => `$${value.toFixed(0)}`}
            />

            <Tooltip content={<TradeTooltipContent />} />

            <Legend
              wrapperStyle={{ fontSize: '12px', color: '#8E8E93' }}
              formatter={(value) => (value === 'all' ? 'Total' : value === 'tradeMarker' ? 'Trades' : value.replace(/_/g, ' '))}
            />

            {/* Zero reference line */}
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" strokeDasharray="3 3" />

            {/* Strategy lines */}
            {strategyNames.map((strategy) => (
              <Line
                key={strategy}
                type="stepAfter"
                dataKey={strategy}
                stroke={STRATEGY_COLORS[strategy] || '#8E8E93'}
                strokeWidth={strategy === 'all' ? 3 : 2}
                dot={false}
                connectNulls
                name={strategy}
              />
            ))}

            {/* Trade markers as scatter overlay */}
            {showMarkers && (
              <Scatter
                dataKey="tradeMarker"
                name="tradeMarker"
                shape={<TradeMarkerDot />}
                legendType="none"
              >
                {chartData.map((point, index) => (
                  <Cell key={`cell-${index}`} fill={point.tradeType === 'exit' ? (point.tradePnl >= 0 ? '#4CAF70' : '#EF5350') : '#C4A24D'} />
                ))}
              </Scatter>
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
