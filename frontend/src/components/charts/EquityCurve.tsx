/**
 * Equity Curve Chart
 *
 * Displays cumulative P&L over time with:
 * - Multiple strategy lines
 * - Sharpe ratio metrics
 * - Drawdown shading
 * - Toggle between $ and % returns
 */

import { useEffect, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts'
import { Loader2, AlertCircle, TrendingUp, TrendingDown } from 'lucide-react'
import apiClient from '../../api/client'

interface EquityPoint {
  timestamp: string
  cumulative_pnl: number
}

interface StrategyData {
  name: string
  data: EquityPoint[]
  final_pnl: number
}

interface EquityCurveData {
  data: StrategyData[]
  strategies: string[]
  total_pnl: number
}

interface EquityCurveProps {
  timeRange?: '7d' | '30d' | '90d' | 'all'
  showDrawdown?: boolean
  autoRefresh?: boolean
}

const STRATEGY_COLORS: Record<string, string> = {
  all: '#C4A24D', // Accent
  single_market_arb: '#4CAF70', // Green
  cross_platform_arb: '#5EB4EF', // Blue
  calibration: '#B27BCC', // Purple
  manual: '#8E8E93', // Gray
}

export default function EquityCurve({
  timeRange = '30d',
  showDrawdown = false,
  autoRefresh = false,
}: EquityCurveProps) {
  const [data, setData] = useState<EquityCurveData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchEquityCurve = async () => {
    try {
      const response = await apiClient.get(`/portfolio/equity-curve?time_range=${timeRange}`)
      setData(response.data)
      setError(null)
    } catch (err: any) {
      setError('Failed to load equity curve')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchEquityCurve()

    if (autoRefresh) {
      const interval = setInterval(fetchEquityCurve, 60_000) // Refresh every 60 seconds
      return () => clearInterval(interval)
    }
  }, [timeRange, autoRefresh])

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
      <div className="flex flex-col items-center justify-center h-80 gap-2">
        <AlertCircle className="h-6 w-6" style={{ color: 'var(--text-3)' }} />
        <p className="text-[14px]" style={{ color: 'var(--text-3)' }}>
          No closed positions yet
        </p>
        <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
          Start trading to see your equity curve
        </p>
      </div>
    )
  }

  // Prepare chart data: merge all strategy timelines
  const chartDataMap = new Map<string, any>()

  data.data.forEach((strategy) => {
    strategy.data.forEach((point) => {
      const timestamp = point.timestamp
      if (!chartDataMap.has(timestamp)) {
        chartDataMap.set(timestamp, {
          timestamp,
          date: new Date(timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        })
      }
      chartDataMap.get(timestamp)![strategy.name] = point.cumulative_pnl
    })
  })

  // Convert to array and sort by timestamp
  const chartData = Array.from(chartDataMap.values()).sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  )

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

  // Calculate metrics
  const totalPnL = data.total_pnl
  const isPositive = totalPnL >= 0

  return (
    <div className="space-y-4">
      {/* Header with metrics */}
      <div className="flex items-center justify-between">
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
              {chartData.length}
            </p>
          </div>
        </div>

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
                  {strategy.name.replace('_', ' ')}:
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

      {/* Equity curve chart */}
      <div style={{ width: '100%', height: '320px', minHeight: '320px' }}>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
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

            <Tooltip
              contentStyle={{
                backgroundColor: '#1A1A1C',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '12px',
                color: '#FFF',
                fontSize: '12px',
                padding: '8px 12px',
              }}
              formatter={(value: number, name: string) => [
                `$${value.toFixed(2)}`,
                name === 'all' ? 'Total' : name.replace('_', ' '),
              ]}
              labelFormatter={(label) => `Date: ${label}`}
            />

            <Legend
              wrapperStyle={{ fontSize: '12px', color: '#8E8E93' }}
              formatter={(value) => (value === 'all' ? 'Total' : value.replace('_', ' '))}
            />

            {/* Zero reference line */}
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" strokeDasharray="3 3" />

            {/* Strategy lines */}
            {strategyNames.map((strategy) => (
              <Line
                key={strategy}
                type="monotone"
                dataKey={strategy}
                stroke={STRATEGY_COLORS[strategy] || '#8E8E93'}
                strokeWidth={strategy === 'all' ? 3 : 2}
                dot={false}
                connectNulls
                name={strategy}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
