/**
 * Win Rate Pie Chart
 *
 * Displays win rate breakdown by strategy:
 * - Donut chart with color-coded segments
 * - Overall win rate in center
 * - Per-strategy metrics on hover
 */

import { useEffect, useState } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts'
import { Loader2, AlertCircle, TrendingUp, TrendingDown } from 'lucide-react'
import apiClient from '../../api/client'
import EmptyState from '../EmptyState'

interface StrategyWinRate {
  strategy: string
  win_rate: number
  total_trades: number
  wins: number
  losses: number
  total_pnl: number
  avg_win: number
  avg_loss: number
  max_loss: number
}

interface WinRateData {
  strategies: StrategyWinRate[]
  overall_win_rate: number
  total_trades: number
}

interface WinRateChartProps {
  minTrades?: number
  portfolioType?: 'all' | 'manual' | 'auto'
}

// Color based on win rate
const getWinRateColor = (winRate: number): string => {
  if (winRate >= 60) return '#4CAF70' // Green (good)
  if (winRate >= 50) return '#C4A24D' // Yellow/Accent (neutral)
  return '#CF6679' // Red (poor)
}

export default function WinRateChart({ minTrades = 1, portfolioType = 'all' }: WinRateChartProps) {
  const [data, setData] = useState<WinRateData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchWinRate = async () => {
    try {
      const ptParam = (portfolioType && portfolioType !== 'all') ? `&portfolio_type=${portfolioType}` : ''
      const response = await apiClient.get(`/portfolio/win-rate?min_trades=${minTrades}${ptParam}`)
      setData(response.data)
      setError(null)
    } catch (err: any) {
      setError('Failed to load win rate data')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchWinRate()
  }, [minTrades, portfolioType])

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
          {error || 'No win rate data'}
        </p>
      </div>
    )
  }

  if (data.strategies.length === 0) {
    return (
      <EmptyState
        icon={TrendingUp}
        title="No trades yet"
        message="Win rate statistics will appear here once you start trading."
      />
    )
  }

  // Prepare pie chart data
  const pieData = data.strategies.map((s) => ({
    name: s.strategy.replace('_', ' '),
    value: s.total_trades,
    win_rate: s.win_rate,
    wins: s.wins,
    losses: s.losses,
    avg_win: s.avg_win,
    avg_loss: s.avg_loss,
  }))

  return (
    <div className="space-y-4">
      {/* Overall win rate */}
      <div className="flex items-center justify-center">
        <div className="text-center">
          <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
            Overall Win Rate
          </p>
          <div className="flex items-center justify-center gap-2">
            {data.overall_win_rate >= 50 ? (
              <TrendingUp className="h-5 w-5" style={{ color: 'var(--green)' }} />
            ) : (
              <TrendingDown className="h-5 w-5" style={{ color: 'var(--red)' }} />
            )}
            <p
              className="text-[32px] font-mono font-bold"
              style={{ color: getWinRateColor(data.overall_win_rate) }}
            >
              {(data.overall_win_rate ?? 0).toFixed(1)}%
            </p>
          </div>
          <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
            {data.total_trades} trades
          </p>
        </div>
      </div>

      {/* Pie chart */}
      <div style={{ width: '100%', height: '256px', minHeight: '256px' }}>
        <ResponsiveContainer width="100%" height={256}>
          <PieChart>
            <Pie
              data={pieData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={90}
              paddingAngle={2}
              label={(props: any) => `${props.name}: ${(props.win_rate ?? 0).toFixed(0)}%`}
              labelLine={{ stroke: 'rgba(255,255,255,0.3)', strokeWidth: 1 }}
            >
              {pieData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={getWinRateColor(entry.win_rate)}
                  stroke="var(--bg)"
                  strokeWidth={2}
                />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: '#1A1A1C',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '12px',
                color: '#FFF',
                fontSize: '12px',
                padding: '10px',
              }}
              formatter={((value: number, name: string, props: any) => {
                const { payload } = props
                return [
                  <div key={name} className="space-y-1">
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Win Rate:</span>
                      <span className="font-semibold">{(payload.win_rate ?? 0).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Trades:</span>
                      <span>{value}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>W/L:</span>
                      <span>
                        {payload.wins}/{payload.losses}
                      </span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Avg Win:</span>
                      <span style={{ color: 'var(--green)' }}>${(payload.avg_win ?? 0).toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span style={{ color: 'var(--text-3)' }}>Avg Loss:</span>
                      <span style={{ color: 'var(--red)' }}>${(payload.avg_loss ?? 0).toFixed(2)}</span>
                    </div>
                  </div>,
                ]
              }) as any}
            />
            <Legend
              wrapperStyle={{ fontSize: '11px', color: '#8E8E93' }}
              formatter={(value) => value}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Strategy breakdown */}
      <div className="space-y-2">
        {data.strategies.map((strategy) => (
          <div
            key={strategy.strategy}
            className="flex items-center justify-between p-3 rounded-lg"
            style={{ background: 'var(--card)' }}
          >
            <div className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded"
                style={{ background: getWinRateColor(strategy.win_rate) }}
              />
              <span className="text-[12px]" style={{ color: 'var(--text)' }}>
                {strategy.strategy.replace('_', ' ')}
              </span>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                {strategy.total_trades} trades
              </span>
              <span
                className="text-[13px] font-mono font-semibold"
                style={{ color: getWinRateColor(strategy.win_rate) }}
              >
                {(strategy.win_rate ?? 0).toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
