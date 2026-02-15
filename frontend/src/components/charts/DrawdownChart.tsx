/**
 * Drawdown Chart
 *
 * Visualizes portfolio drawdown (underwater equity):
 * - % drawdown from peak equity
 * - Max drawdown highlight
 * - Recovery periods
 * - Risk metrics (Calmar ratio, max DD)
 */

import { useEffect, useState } from 'react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { Loader2, AlertCircle, TrendingDown } from 'lucide-react'
import apiClient from '../../api/client'
import EmptyState from '../EmptyState'

interface DrawdownPoint {
  date: string
  timestamp: string
  drawdown_pct: number
  equity: number
  peak_equity: number
}

interface DrawdownChartProps {
  timeRange?: '7d' | '30d' | '90d' | 'all'
  portfolioType?: 'all' | 'manual' | 'auto'
}

export default function DrawdownChart({ timeRange = '30d', portfolioType = 'all' }: DrawdownChartProps) {
  const [data, setData] = useState<DrawdownPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [maxDrawdown, setMaxDrawdown] = useState(0)

  const fetchDrawdownData = async () => {
    setLoading(true)
    setError(null)

    try {
      // Fetch equity curve data (don't send portfolio_type='all', API doesn't accept it)
      const ptParam = (portfolioType && portfolioType !== 'all') ? `&portfolio_type=${portfolioType}` : ''
      const response = await apiClient.get(`/portfolio/equity-curve?time_range=${timeRange}${ptParam}`)
      const equityCurveData = response.data

      if (!equityCurveData.data || equityCurveData.data.length === 0) {
        setData([])
        return
      }

      // Find the "all" strategy (total P&L)
      const allStrategy = equityCurveData.data.find((s: any) => s.name === 'all')
      if (!allStrategy || allStrategy.data.length === 0) {
        setData([])
        return
      }

      // Compute drawdown from equity curve
      const equityPoints = allStrategy.data
      const drawdownPoints: DrawdownPoint[] = []
      let peakEquity = 0
      let maxDD = 0

      equityPoints.forEach((point: any) => {
        const equity = point.cumulative_pnl

        // Update peak
        if (equity > peakEquity) {
          peakEquity = equity
        }

        // Calculate drawdown percentage
        const drawdownPct = peakEquity > 0 ? ((equity - peakEquity) / Math.abs(peakEquity)) * 100 : 0

        // Track max drawdown
        if (drawdownPct < maxDD) {
          maxDD = drawdownPct
        }

        drawdownPoints.push({
          timestamp: point.timestamp,
          date: new Date(point.timestamp).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
          }),
          drawdown_pct: drawdownPct,
          equity: equity,
          peak_equity: peakEquity,
        })
      })

      setData(drawdownPoints)
      setMaxDrawdown(maxDD)
    } catch (err: any) {
      setError('Failed to load drawdown data')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDrawdownData()
  }, [timeRange, portfolioType])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-5 w-5 animate-spin" style={{ color: 'var(--text-3)' }} />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-2">
        <AlertCircle className="h-5 w-5" style={{ color: 'var(--red)' }} />
        <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
          {error || 'No drawdown data'}
        </p>
      </div>
    )
  }

  if (data.length === 0) {
    return (
      <EmptyState
        icon={TrendingDown}
        title="No drawdown data available"
        message="Drawdown analysis will appear here once you have trading history."
      />
    )
  }

  // Calculate risk metrics
  const finalEquity = data[data.length - 1]?.equity || 0
  const calmarRatio = maxDrawdown !== 0 ? (finalEquity / Math.abs(maxDrawdown)).toFixed(2) : 'N/A'

  return (
    <div className="space-y-4">
      {/* Risk metrics header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <TrendingDown className="h-4 w-4" style={{ color: 'var(--red)' }} />
            <div>
              <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
                Max Drawdown
              </p>
              <p className="text-[18px] font-mono font-bold" style={{ color: 'var(--red)' }}>
                {maxDrawdown.toFixed(2)}%
              </p>
            </div>
          </div>

          <div>
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Calmar Ratio
            </p>
            <p className="text-[18px] font-mono font-semibold" style={{ color: 'var(--accent)' }}>
              {calmarRatio}
            </p>
          </div>

          <div>
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Current Drawdown
            </p>
            <p className="text-[18px] font-mono font-semibold" style={{ color: 'var(--text)' }}>
              {data[data.length - 1]?.drawdown_pct.toFixed(2)}%
            </p>
          </div>
        </div>

        {/* Legend */}
        <div className="text-[11px]" style={{ color: 'var(--text-3)' }}>
          Underwater equity (% from peak)
        </div>
      </div>

      {/* Drawdown chart */}
      <div style={{ width: '100%', height: '256px', minHeight: '256px' }}>
        <ResponsiveContainer width="100%" height={256}>
          <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#CF6679" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#CF6679" stopOpacity={0.05} />
              </linearGradient>
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
              tickFormatter={(value) => `${value.toFixed(0)}%`}
              domain={[maxDrawdown * 1.1, 0]} // Always show 0 at top
            />

            <Tooltip
              contentStyle={{
                backgroundColor: '#1A1A1C',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '12px',
                color: '#FFF',
                fontSize: '12px',
                padding: '8px 12px',
              }}
              formatter={(value: number | undefined) => [`${(value ?? 0).toFixed(2)}%`, 'Drawdown']}
              labelFormatter={(label) => `Date: ${label}`}
            />

            {/* Zero reference line (no drawdown) */}
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)" strokeDasharray="3 3" />

            {/* Max drawdown reference line */}
            <ReferenceLine
              y={maxDrawdown}
              stroke="#CF6679"
              strokeDasharray="3 3"
              strokeWidth={2}
              label={{
                value: `Max DD: ${maxDrawdown.toFixed(1)}%`,
                position: 'left',
                fill: '#CF6679',
                fontSize: 11,
              }}
            />

            {/* Drawdown area (always below zero, shown as negative space) */}
            <Area
              type="monotone"
              dataKey="drawdown_pct"
              stroke="#CF6679"
              fill="url(#drawdownGradient)"
              strokeWidth={2}
              name="Drawdown"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Recovery indicator */}
      {data[data.length - 1]?.drawdown_pct === 0 && (
        <div className="flex items-center justify-center gap-2 p-3 rounded-lg" style={{ background: 'rgba(76,175,112,0.1)' }}>
          <div className="w-2 h-2 rounded-full" style={{ background: 'var(--green)' }} />
          <p className="text-[12px] font-medium" style={{ color: 'var(--green)' }}>
            Portfolio at new peak (fully recovered)
          </p>
        </div>
      )}
    </div>
  )
}
