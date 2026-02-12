/**
 * Performance Attribution (Waterfall Chart)
 *
 * Shows how each strategy contributed to total returns:
 * - Starting capital → Strategy P&Ls → Fees → Ending capital
 * - Waterfall visualization
 * - Identifies top and bottom contributors
 */

import { useEffect, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts'
import { Loader2, AlertCircle } from 'lucide-react'
import apiClient from '../../api/client'

interface WaterfallItem {
  name: string
  value: number
  cumulative: number
  type: 'start' | 'positive' | 'negative' | 'end'
}

export default function PerformanceAttribution() {
  const [data, setData] = useState<WaterfallItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchPerformance = async () => {
    setLoading(true)
    setError(null)

    try {
      // Fetch portfolio summary with strategy breakdown
      const response = await apiClient.get('/portfolio/summary')
      const summary = response.data

      const startingCapital = 1000 // Assume $1000 starting capital
      let cumulative = startingCapital

      const waterfall: WaterfallItem[] = []

      // Starting capital
      waterfall.push({
        name: 'Starting Capital',
        value: startingCapital,
        cumulative: startingCapital,
        type: 'start',
      })

      // Add each strategy's P&L
      const strategies = summary.by_strategy || []
      strategies.forEach((strategy: any) => {
        const pnl = strategy.total_pnl || 0
        cumulative += pnl

        waterfall.push({
          name: strategy.strategy.replace('_', ' '),
          value: pnl,
          cumulative,
          type: pnl >= 0 ? 'positive' : 'negative',
        })
      })

      // Estimate fees (assume 1% of total volume as fees)
      const totalVolume = Math.abs(summary.total_realized_pnl || 0) * 10 // Rough estimate
      const estimatedFees = totalVolume * 0.01
      if (estimatedFees > 0) {
        cumulative -= estimatedFees
        waterfall.push({
          name: 'Fees',
          value: -estimatedFees,
          cumulative,
          type: 'negative',
        })
      }

      // Ending capital
      waterfall.push({
        name: 'Ending Capital',
        value: cumulative,
        cumulative,
        type: 'end',
      })

      setData(waterfall)
    } catch (err: any) {
      setError('Failed to load performance data')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPerformance()
  }, [])

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
          {error || 'No performance data'}
        </p>
      </div>
    )
  }

  if (data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-2">
        <AlertCircle className="h-5 w-5" style={{ color: 'var(--text-3)' }} />
        <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
          No performance data available
        </p>
      </div>
    )
  }

  // Prepare data for waterfall chart
  // Need to calculate the base (starting Y position) for each bar
  const chartData = data.map((item, index) => {
    if (item.type === 'start') {
      return {
        ...item,
        base: 0,
        height: item.value,
      }
    } else if (item.type === 'end') {
      return {
        ...item,
        base: 0,
        height: item.cumulative,
      }
    } else {
      const prevCumulative = data[index - 1]?.cumulative || 0
      if (item.value >= 0) {
        // Positive: bar goes up from previous cumulative
        return {
          ...item,
          base: prevCumulative,
          height: item.value,
        }
      } else {
        // Negative: bar goes down from previous cumulative
        return {
          ...item,
          base: prevCumulative + item.value,
          height: Math.abs(item.value),
        }
      }
    }
  })

  const getColor = (item: WaterfallItem): string => {
    switch (item.type) {
      case 'start':
        return '#8E8E93' // Gray
      case 'positive':
        return '#4CAF70' // Green
      case 'negative':
        return '#CF6679' // Red
      case 'end':
        return '#C4A24D' // Accent
      default:
        return '#8E8E93'
    }
  }

  return (
    <div className="space-y-4">
      {/* Summary stats */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div>
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Starting
            </p>
            <p className="text-[16px] font-mono font-semibold" style={{ color: 'var(--text)' }}>
              ${data[0]?.value.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Ending
            </p>
            <p className="text-[16px] font-mono font-semibold" style={{ color: 'var(--accent)' }}>
              ${data[data.length - 1]?.cumulative.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Total Return
            </p>
            <p
              className="text-[16px] font-mono font-semibold"
              style={{
                color:
                  data[data.length - 1]?.cumulative >= data[0]?.value
                    ? 'var(--green)'
                    : 'var(--red)',
              }}
            >
              ${(data[data.length - 1]?.cumulative - data[0]?.value).toFixed(2)}
            </p>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-[11px]">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded" style={{ background: '#4CAF70' }} />
            <span style={{ color: 'var(--text-3)' }}>Gains</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded" style={{ background: '#CF6679' }} />
            <span style={{ color: 'var(--text-3)' }}>Losses</span>
          </div>
        </div>
      </div>

      {/* Waterfall chart */}
      <div style={{ width: '100%', height: '320px', minHeight: '320px' }}>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 20, left: 20, bottom: 80 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />

            <XAxis
              dataKey="name"
              stroke="rgba(255,255,255,0.06)"
              tick={{ fill: '#48484A', fontSize: 11 }}
              angle={-45}
              textAnchor="end"
              height={80}
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
              formatter={(value: number, name: string, props: any) => {
                const { payload } = props
                return [
                  <div key={name} className="space-y-1">
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Change:</span>
                      <span className="font-mono font-semibold">
                        ${payload.value >= 0 ? '+' : ''}${payload.value.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Cumulative:</span>
                      <span className="font-mono">${payload.cumulative.toFixed(2)}</span>
                    </div>
                  </div>,
                ]
              }}
            />

            {/* Reference line at starting capital */}
            <ReferenceLine
              y={data[0]?.value}
              stroke="rgba(255,255,255,0.2)"
              strokeDasharray="3 3"
            />

            {/* Stacked bars showing waterfall effect */}
            <Bar dataKey="height" stackId="a">
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getColor(entry)} />
              ))}
            </Bar>

            {/* Invisible bars to create waterfall effect (offset) */}
            <Bar dataKey="base" stackId="a" fill="transparent" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Strategy breakdown */}
      <div className="space-y-2">
        <p className="text-[12px] font-medium" style={{ color: 'var(--text)' }}>
          Strategy Contributions
        </p>
        {data
          .filter((item) => item.type !== 'start' && item.type !== 'end')
          .sort((a, b) => b.value - a.value)
          .map((item, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-2.5 rounded-lg"
              style={{ background: 'var(--card)' }}
            >
              <span className="text-[12px]" style={{ color: 'var(--text)' }}>
                {item.name}
              </span>
              <span
                className="text-[12px] font-mono font-semibold"
                style={{ color: item.value >= 0 ? 'var(--green)' : 'var(--red)' }}
              >
                {item.value >= 0 ? '+' : ''}${item.value.toFixed(2)}
              </span>
            </div>
          ))}
      </div>
    </div>
  )
}
