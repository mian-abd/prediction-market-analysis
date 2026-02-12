/**
 * Portfolio Page
 *
 * Paper trading dashboard with:
 * - Equity curve showing cumulative P&L
 * - Open positions table
 * - Closed positions history
 * - Performance metrics
 */

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Briefcase, Loader2, AlertCircle } from 'lucide-react'
import apiClient from '../api/client'
import EquityCurve from '../components/charts/EquityCurve'
import DrawdownChart from '../components/charts/DrawdownChart'
import WinRateChart from '../components/charts/WinRateChart'
import PositionHeatmap from '../components/charts/PositionHeatmap'
import PerformanceAttribution from '../components/charts/PerformanceAttribution'

interface PortfolioSummary {
  open_positions: number
  closed_positions: number
  total_realized_pnl: number
  win_rate: number
  total_exposure: number
  by_strategy: Array<{
    strategy: string
    trades: number
    total_pnl: number
  }>
}

interface Position {
  id: number
  market_id: number
  question: string
  platform: string
  side: string
  entry_price: number
  quantity: number
  entry_time: string
  exit_time: string | null
  exit_price: number | null
  realized_pnl: number | null
  unrealized_pnl: number | null
  current_price: number | null
  strategy: string
}

export default function Portfolio() {
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | 'all'>('30d')
  const [positionStatus, setPositionStatus] = useState<'open' | 'closed' | 'all'>('open')

  // Fetch portfolio summary
  const { data: summary, isLoading: summaryLoading } = useQuery<PortfolioSummary>({
    queryKey: ['portfolio-summary'],
    queryFn: async () => {
      const response = await apiClient.get('/portfolio/summary')
      return response.data
    },
    refetchInterval: 15_000,
  })

  // Fetch positions
  const { data: positionsData, isLoading: positionsLoading } = useQuery<{ positions: Position[] }>({
    queryKey: ['portfolio-positions', positionStatus],
    queryFn: async () => {
      const response = await apiClient.get(`/portfolio/positions?status=${positionStatus}`)
      return response.data
    },
    refetchInterval: 15_000,
  })

  const positions = positionsData?.positions || []

  if (summaryLoading) {
    return (
      <div className="flex items-center justify-center h-80">
        <Loader2 className="h-6 w-6 animate-spin" style={{ color: 'var(--text-3)' }} />
      </div>
    )
  }

  return (
    <div className="space-y-6 fade-up">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background: 'var(--accent-dim)' }}
          >
            <Briefcase className="h-5 w-5" style={{ color: 'var(--accent)' }} />
          </div>
          <div>
            <h1 className="text-[22px] font-bold" style={{ color: 'var(--text)' }}>
              Portfolio
            </h1>
            <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
              Paper trading performance and positions
            </p>
          </div>
        </div>
      </div>

      {/* Summary stats */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="card p-4">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
              Total P&L
            </p>
            <p
              className="text-[20px] font-mono font-bold"
              style={{ color: summary.total_realized_pnl >= 0 ? 'var(--green)' : 'var(--red)' }}
            >
              ${summary.total_realized_pnl.toFixed(2)}
            </p>
          </div>

          <div className="card p-4">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
              Win Rate
            </p>
            <p className="text-[20px] font-mono font-bold" style={{ color: 'var(--accent)' }}>
              {summary.win_rate.toFixed(1)}%
            </p>
          </div>

          <div className="card p-4">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
              Open Positions
            </p>
            <p className="text-[20px] font-semibold" style={{ color: 'var(--text)' }}>
              {summary.open_positions}
            </p>
          </div>

          <div className="card p-4">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
              Closed Trades
            </p>
            <p className="text-[20px] font-semibold" style={{ color: 'var(--text)' }}>
              {summary.closed_positions}
            </p>
          </div>

          <div className="card p-4">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
              Exposure
            </p>
            <p className="text-[20px] font-mono font-bold" style={{ color: 'var(--text)' }}>
              ${summary.total_exposure.toFixed(2)}
            </p>
          </div>
        </div>
      )}

      {/* Equity Curve */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-[16px] font-semibold" style={{ color: 'var(--text)' }}>
            Equity Curve
          </h2>
          <div className="flex items-center gap-2">
            {(['7d', '30d', '90d', 'all'] as const).map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1.5 rounded-lg text-[12px] font-medium transition-colors ${
                  timeRange === range ? 'btn' : 'btn-ghost'
                }`}
              >
                {range.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <EquityCurve timeRange={timeRange} showDrawdown={false} autoRefresh={true} />
      </div>

      {/* Drawdown Chart */}
      <div className="card p-6">
        <h2 className="text-[16px] font-semibold mb-5" style={{ color: 'var(--text)' }}>
          Drawdown Analysis
        </h2>
        <DrawdownChart timeRange={timeRange} />
      </div>

      {/* Win Rate by Strategy */}
      <div className="card p-6">
        <h2 className="text-[16px] font-semibold mb-5" style={{ color: 'var(--text)' }}>
          Win Rate by Strategy
        </h2>
        <WinRateChart minTrades={1} />
      </div>

      {/* Position Heatmap */}
      <div className="card p-6">
        <h2 className="text-[16px] font-semibold mb-5" style={{ color: 'var(--text)' }}>
          Position Heatmap
        </h2>
        <PositionHeatmap />
      </div>

      {/* Positions table */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-[16px] font-semibold" style={{ color: 'var(--text)' }}>
            Positions
          </h2>
          <div className="flex items-center gap-2">
            {(['open', 'closed', 'all'] as const).map((status) => (
              <button
                key={status}
                onClick={() => setPositionStatus(status)}
                className={`px-3 py-1.5 rounded-lg text-[12px] font-medium capitalize transition-colors ${
                  positionStatus === status ? 'btn' : 'btn-ghost'
                }`}
              >
                {status}
              </button>
            ))}
          </div>
        </div>

        {positionsLoading ? (
          <div className="flex items-center justify-center h-40">
            <Loader2 className="h-5 w-5 animate-spin" style={{ color: 'var(--text-3)' }} />
          </div>
        ) : positions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-40 gap-2">
            <AlertCircle className="h-5 w-5" style={{ color: 'var(--text-3)' }} />
            <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
              No {positionStatus} positions
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]">
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  <th className="text-left py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    Market
                  </th>
                  <th className="text-left py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    Side
                  </th>
                  <th className="text-right py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    Entry
                  </th>
                  <th className="text-right py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    Current/Exit
                  </th>
                  <th className="text-right py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    Qty
                  </th>
                  <th className="text-right py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    P&L
                  </th>
                  <th className="text-left py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    Strategy
                  </th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos) => {
                  const pnl = pos.realized_pnl ?? pos.unrealized_pnl ?? 0
                  const isProfitable = pnl >= 0

                  return (
                    <tr
                      key={pos.id}
                      style={{ borderBottom: '1px solid var(--border)' }}
                      className="hover:bg-white/[0.02] transition-colors"
                    >
                      <td className="py-3 px-3">
                        <p className="truncate max-w-[300px]" style={{ color: 'var(--text)' }}>
                          {pos.question}
                        </p>
                        <p className="text-[10px] uppercase mt-0.5" style={{ color: 'var(--text-3)' }}>
                          {pos.platform}
                        </p>
                      </td>
                      <td className="py-3 px-3">
                        <span
                          className="px-2 py-1 rounded text-[10px] font-semibold uppercase"
                          style={{
                            background: pos.side === 'yes' ? 'rgba(76,175,112,0.15)' : 'rgba(207,102,121,0.15)',
                            color: pos.side === 'yes' ? 'var(--green)' : 'var(--red)',
                          }}
                        >
                          {pos.side}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right font-mono" style={{ color: 'var(--text)' }}>
                        {pos.entry_price.toFixed(3)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono" style={{ color: 'var(--text)' }}>
                        {(pos.exit_price ?? pos.current_price ?? 0).toFixed(3)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono" style={{ color: 'var(--text)' }}>
                        {pos.quantity.toFixed(0)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono font-semibold">
                        <span style={{ color: isProfitable ? 'var(--green)' : 'var(--red)' }}>
                          {isProfitable ? '+' : ''}${pnl.toFixed(2)}
                        </span>
                      </td>
                      <td className="py-3 px-3">
                        <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                          {pos.strategy.replace('_', ' ')}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Performance Attribution */}
      <div className="card p-6">
        <h2 className="text-[16px] font-semibold mb-5" style={{ color: 'var(--text)' }}>
          Performance Attribution
        </h2>
        <PerformanceAttribution />
      </div>
    </div>
  )
}
