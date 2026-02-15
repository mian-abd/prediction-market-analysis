import { useQuery } from '@tanstack/react-query'
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ScatterChart,
  Scatter,
  ReferenceLine,
} from 'recharts'
import { Target, TrendingUp } from 'lucide-react'
import apiClient from '../../api/client'
import { Skeleton } from '../LoadingSkeleton'

interface BacktestData {
  n_signals_evaluated: number
  hit_rate: number | null
  brier_score: number | null
  baseline_brier: number | null
  brier_improvement_pct: number | null
  simulated_pnl: number
  by_direction: Record<string, { total: number; correct: number; hit_rate: number; pnl: number }>
  by_quality_tier: Record<string, { total: number; correct: number; hit_rate: number; pnl: number }>
  timeline: Array<{
    date: string
    market_id: number
    direction: string
    ensemble_prob: number
    market_price: number
    resolution: number
    correct: boolean
    pnl: number
    cumulative_pnl: number
  }>
}

interface PerformanceData {
  data: Array<{
    date: string
    signals_generated: number
    signals_correct: number
    daily_pnl: number
    cumulative_pnl: number
    cumulative_hit_rate: number
  }>
  summary: {
    total_scored: number
    total_correct: number
    hit_rate: number
    cumulative_pnl: number
  }
}

export default function SignalAccuracyChart() {
  const { data: backtest, isLoading: btLoading } = useQuery<BacktestData>({
    queryKey: ['signal-backtest'],
    queryFn: async () => (await apiClient.get('/predictions/accuracy/backtest')).data,
    refetchInterval: 300_000,
  })

  const { data: performance, isLoading: perfLoading } = useQuery<PerformanceData>({
    queryKey: ['signal-performance'],
    queryFn: async () => (await apiClient.get('/strategies/signal-performance')).data,
    refetchInterval: 300_000,
  })

  const isLoading = btLoading || perfLoading
  const hasBacktest = backtest && backtest.n_signals_evaluated > 0
  const hasPerformance = performance && performance.data.length > 0

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-[300px] w-full" />
      </div>
    )
  }

  if (!hasBacktest && !hasPerformance) {
    return (
      <div className="card p-6">
        <div className="flex items-center gap-2 mb-3">
          <Target className="h-4 w-4" style={{ color: 'var(--accent)' }} />
          <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>Signal Accuracy</p>
        </div>
        <div className="flex flex-col items-center py-8 gap-2">
          <Target className="h-5 w-5" style={{ color: 'var(--text-3)' }} />
          <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
            No resolved signals yet
          </p>
          <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
            Accuracy data appears after markets with signals resolve
          </p>
        </div>
      </div>
    )
  }

  // Build calibration plot data from backtest timeline
  const calibrationBuckets: Record<number, { count: number; correct: number }> = {}
  if (hasBacktest) {
    for (const entry of backtest.timeline) {
      const bucket = Math.round(entry.ensemble_prob * 10) / 10 // round to 0.1
      if (!calibrationBuckets[bucket]) calibrationBuckets[bucket] = { count: 0, correct: 0 }
      calibrationBuckets[bucket].count++
      if (entry.correct) calibrationBuckets[bucket].correct++
    }
  }
  const calibrationData = Object.entries(calibrationBuckets)
    .map(([prob, { count, correct }]) => ({
      predicted: parseFloat(prob),
      actual: count > 0 ? correct / count : 0,
      count,
    }))
    .sort((a, b) => a.predicted - b.predicted)

  // Perfect calibration diagonal (used for visual reference)

  return (
    <div className="space-y-4">
      {/* Summary Stats */}
      {hasBacktest && (
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-4">
            <Target className="h-4 w-4" style={{ color: 'var(--accent)' }} />
            <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>
              Signal Accuracy (Backtest)
            </p>
            <span className="pill pill-accent">{backtest.n_signals_evaluated} signals evaluated</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
            {[
              {
                label: 'Hit Rate',
                value: backtest.hit_rate != null ? `${(backtest.hit_rate * 100).toFixed(1)}%` : 'N/A',
                color: 'var(--green)',
              },
              {
                label: 'Brier Score',
                value: backtest.brier_score != null ? backtest.brier_score.toFixed(4) : 'N/A',
                color: 'var(--accent)',
              },
              {
                label: 'vs Baseline',
                value: backtest.brier_improvement_pct != null
                  ? `${backtest.brier_improvement_pct > 0 ? '+' : ''}${backtest.brier_improvement_pct.toFixed(1)}%`
                  : 'N/A',
                color: (backtest.brier_improvement_pct ?? 0) > 0 ? 'var(--green)' : 'var(--red)',
              },
              {
                label: 'Simulated P&L',
                value: `$${backtest.simulated_pnl.toFixed(0)}`,
                color: backtest.simulated_pnl >= 0 ? 'var(--green)' : 'var(--red)',
              },
            ].map((s) => (
              <div key={s.label} className="text-center py-3 rounded-xl" style={{ background: 'rgba(255,255,255,0.03)' }}>
                <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>{s.label}</p>
                <p className="text-[18px] font-bold" style={{ color: s.color }}>{s.value}</p>
              </div>
            ))}
          </div>

          {/* Direction breakdown */}
          {Object.keys(backtest.by_direction).length > 0 && (
            <div className="flex gap-4 flex-wrap">
              {Object.entries(backtest.by_direction).map(([dir, stats]) => (
                <div key={dir} className="flex items-center gap-2 px-3 py-1.5 rounded-lg" style={{ background: 'rgba(255,255,255,0.03)' }}>
                  <span className="text-[11px] font-mono capitalize" style={{ color: 'var(--text-2)' }}>{dir.replace('_', ' ')}</span>
                  <span className="text-[11px] font-bold" style={{ color: 'var(--green)' }}>
                    {(stats.hit_rate * 100).toFixed(0)}% ({stats.correct}/{stats.total})
                  </span>
                  <span className="text-[10px] font-mono" style={{ color: stats.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    ${stats.pnl.toFixed(0)}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Quality tier breakdown */}
          {Object.keys(backtest.by_quality_tier).length > 0 && (
            <div className="flex gap-3 flex-wrap mt-3">
              {Object.entries(backtest.by_quality_tier).map(([tier, stats]) => (
                <div key={tier} className="flex items-center gap-2 px-3 py-1.5 rounded-lg" style={{ background: 'rgba(255,255,255,0.03)' }}>
                  <span className={`pill ${tier === 'high' ? 'pill-green' : tier === 'medium' ? 'pill-accent' : tier === 'speculative' ? 'pill-red' : ''}`}>
                    {tier}
                  </span>
                  <span className="text-[11px]" style={{ color: 'var(--text-2)' }}>
                    {(stats.hit_rate * 100).toFixed(0)}% hit &middot; ${stats.pnl.toFixed(0)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Calibration Plot */}
      {calibrationData.length > 2 && (
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-4">
            <Target className="h-4 w-4" style={{ color: 'var(--blue)' }} />
            <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>Calibration Plot</p>
            <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
              Predicted vs actual resolution rate
            </span>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="predicted"
                type="number"
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                stroke="var(--text-3)"
                fontSize={10}
                label={{ value: 'Predicted Probability', position: 'insideBottom', offset: -10, style: { fill: 'var(--text-3)', fontSize: 10 } }}
              />
              <YAxis
                dataKey="actual"
                type="number"
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                stroke="var(--text-3)"
                fontSize={10}
                label={{ value: 'Actual Resolution Rate', angle: -90, position: 'insideLeft', style: { fill: 'var(--text-3)', fontSize: 10 } }}
              />
              <ReferenceLine
                segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                stroke="var(--text-3)"
                strokeDasharray="5 5"
                strokeOpacity={0.5}
              />
              <Tooltip
                contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: '8px', fontSize: '11px' }}
                formatter={(value: unknown, name?: string) => [
                  `${((value as number) * 100).toFixed(1)}%`,
                  name === 'actual' ? 'Actual' : 'Predicted'
                ]}
              />
              <Scatter data={calibrationData} fill="var(--accent)" fillOpacity={0.8} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Cumulative P&L Chart */}
      {hasPerformance && (
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="h-4 w-4" style={{ color: 'var(--green)' }} />
            <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>Cumulative Signal P&L</p>
            <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
              Per $100 notional per signal
            </span>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={performance.data} margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="date"
                stroke="var(--text-3)"
                fontSize={10}
                tickFormatter={(v) => v.slice(5)} // MM-DD
              />
              <YAxis
                yAxisId="pnl"
                stroke="var(--text-3)"
                fontSize={10}
                tickFormatter={(v) => `$${v}`}
              />
              <YAxis
                yAxisId="count"
                orientation="right"
                stroke="var(--text-3)"
                fontSize={10}
              />
              <Tooltip
                contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: '8px', fontSize: '11px' }}
              />
              <Legend wrapperStyle={{ fontSize: '10px' }} />
              <Bar
                yAxisId="count"
                dataKey="signals_generated"
                fill="rgba(196,162,77,0.2)"
                name="Signals"
              />
              <Line
                yAxisId="pnl"
                dataKey="cumulative_pnl"
                stroke="var(--green)"
                strokeWidth={2}
                dot={false}
                name="Cumulative P&L"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
