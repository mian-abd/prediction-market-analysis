/**
 * Auto-Trading Control Panel
 *
 * Two side-by-side strategy cards (ML Ensemble + Elo Sports) with:
 * - Enable/disable toggle
 * - Live status (exposure, P&L, open positions)
 * - Inline settings form for configuration
 */

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Bot, Settings, ToggleLeft, ToggleRight } from 'lucide-react'
import apiClient from '../api/client'
import { Skeleton } from './LoadingSkeleton'

interface AutoConfig {
  strategy: string
  is_enabled: boolean
  min_quality_tier: string
  min_confidence: number
  min_net_ev: number
  bankroll: number
  max_kelly_fraction: number
  max_position_usd: number
  max_total_exposure_usd: number
  max_loss_per_day_usd: number
  max_daily_trades: number
  stop_loss_pct: number
  close_on_signal_expiry: boolean
  updated_at: string | null
}

interface AutoStatus {
  enabled_strategies: string[]
  configs: AutoConfig[]
  open_positions: Record<string, number>
  total_exposure: number
  today_pnl: number
  pnl_by_strategy: Record<string, number>
  exposure_by_strategy: Record<string, number>
  recent_trades: any[]
}

export default function AutoTradingPanel() {
  const queryClient = useQueryClient()

  const { data: status, isLoading } = useQuery<AutoStatus>({
    queryKey: ['auto-trading-status'],
    queryFn: async () => (await apiClient.get('/auto-trading/status')).data,
    refetchInterval: 15_000,
  })

  const toggleMutation = useMutation({
    mutationFn: async ({ strategy, enabled }: { strategy: string; enabled: boolean }) => {
      await apiClient.post(`/auto-trading/toggle/${strategy}?enabled=${enabled}`)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auto-trading-status'] })
    },
  })

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[1, 2].map(i => (
          <div key={i} className="card p-5 space-y-3">
            <Skeleton className="h-5 w-32" />
            <Skeleton className="h-8 w-20" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
          </div>
        ))}
      </div>
    )
  }

  if (!status) return null

  const ensembleConfig = status.configs.find(c => c.strategy === 'ensemble')
  const eloConfig = status.configs.find(c => c.strategy === 'elo')

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <StrategyCard
        title="ML Ensemble"
        strategyKey="ensemble"
        config={ensembleConfig}
        openPositions={status.open_positions.auto_ensemble ?? 0}
        totalExposure={status.exposure_by_strategy?.auto_ensemble ?? 0}
        todayPnl={status.pnl_by_strategy?.auto_ensemble ?? 0}
        onToggle={(enabled) => toggleMutation.mutate({ strategy: 'ensemble', enabled })}
        toggleLoading={toggleMutation.isPending}
      />
      <StrategyCard
        title="Elo Sports"
        strategyKey="elo"
        config={eloConfig}
        openPositions={status.open_positions.auto_elo ?? 0}
        totalExposure={status.exposure_by_strategy?.auto_elo ?? 0}
        todayPnl={status.pnl_by_strategy?.auto_elo ?? 0}
        onToggle={(enabled) => toggleMutation.mutate({ strategy: 'elo', enabled })}
        toggleLoading={toggleMutation.isPending}
      />
    </div>
  )
}

function StrategyCard({
  title,
  strategyKey,
  config,
  openPositions,
  totalExposure,
  todayPnl,
  onToggle,
  toggleLoading,
}: {
  title: string
  strategyKey: string
  config?: AutoConfig
  openPositions: number
  totalExposure: number
  todayPnl: number
  onToggle: (enabled: boolean) => void
  toggleLoading: boolean
}) {
  const [showConfig, setShowConfig] = useState(false)

  if (!config) return null

  const isEnabled = config.is_enabled

  return (
    <div className="card p-5">
      {/* Header + Toggle */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Bot className="h-4 w-4" style={{ color: 'var(--accent)' }} />
          <span className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>
            {title}
          </span>
        </div>
        <button
          onClick={() => onToggle(!isEnabled)}
          disabled={toggleLoading}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold transition-colors"
          style={{
            background: isEnabled ? 'rgba(76,175,112,0.15)' : 'rgba(255,255,255,0.05)',
            color: isEnabled ? 'var(--green)' : 'var(--text-3)',
          }}
        >
          {isEnabled ? (
            <ToggleRight className="h-4 w-4" />
          ) : (
            <ToggleLeft className="h-4 w-4" />
          )}
          {isEnabled ? 'ENABLED' : 'DISABLED'}
        </button>
      </div>

      {/* Live Stats */}
      <div className="space-y-2 mb-4">
        <StatRow label="Bankroll" value={`$${config.bankroll.toLocaleString()}`} />
        <StatRow label="Min Confidence" value={`${(config.min_confidence * 100).toFixed(0)}%`} />
        <StatRow label="Min EV" value={`${(config.min_net_ev * 100).toFixed(0)}%`} />
        <StatRow label="Stop Loss" value={`${(config.stop_loss_pct * 100).toFixed(0)}%`} />
        <StatRow
          label="Exposure"
          value={`$${totalExposure.toFixed(0)} / $${config.max_total_exposure_usd.toFixed(0)}`}
        />
        <StatRow
          label="Today P&L"
          value={`${todayPnl >= 0 ? '+' : ''}$${todayPnl.toFixed(2)}`}
          color={todayPnl >= 0 ? 'var(--green)' : 'var(--red)'}
        />
        <StatRow label="Open Positions" value={`${openPositions}`} />
      </div>

      {/* Configure button */}
      <button
        onClick={() => setShowConfig(!showConfig)}
        className="flex items-center gap-1.5 text-[11px] font-medium transition-colors"
        style={{ color: 'var(--accent)' }}
      >
        <Settings className="h-3.5 w-3.5" />
        {showConfig ? 'Hide Settings' : 'Configure'}
      </button>

      {/* Inline config form */}
      {showConfig && (
        <ConfigForm strategyKey={strategyKey} config={config} />
      )}
    </div>
  )
}

function StatRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>{label}</span>
      <span className="text-[12px] font-mono font-medium" style={{ color: color || 'var(--text)' }}>
        {value}
      </span>
    </div>
  )
}

function ConfigForm({ strategyKey, config }: { strategyKey: string; config: AutoConfig }) {
  const queryClient = useQueryClient()
  const [values, setValues] = useState({
    bankroll: config.bankroll,
    min_confidence: config.min_confidence,
    min_net_ev: config.min_net_ev,
    stop_loss_pct: config.stop_loss_pct,
    max_total_exposure_usd: config.max_total_exposure_usd,
    max_position_usd: config.max_position_usd,
    max_daily_trades: config.max_daily_trades,
    max_loss_per_day_usd: config.max_loss_per_day_usd,
  })

  const saveMutation = useMutation({
    mutationFn: async (data: Record<string, number>) => {
      await apiClient.put(`/auto-trading/config/${strategyKey}`, data)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auto-trading-status'] })
    },
  })

  const handleSave = () => {
    saveMutation.mutate(values)
  }

  return (
    <div className="mt-3 pt-3 space-y-2" style={{ borderTop: '1px solid var(--border)' }}>
      {[
        { key: 'bankroll', label: 'Bankroll ($)', step: 100 },
        { key: 'min_confidence', label: 'Min Confidence', step: 0.05 },
        { key: 'min_net_ev', label: 'Min Net EV', step: 0.01 },
        { key: 'stop_loss_pct', label: 'Stop Loss %', step: 0.05 },
        { key: 'max_total_exposure_usd', label: 'Max Exposure ($)', step: 100 },
        { key: 'max_position_usd', label: 'Max Position ($)', step: 10 },
        { key: 'max_daily_trades', label: 'Max Daily Trades', step: 1 },
        { key: 'max_loss_per_day_usd', label: 'Max Daily Loss ($)', step: 5 },
      ].map(({ key, label, step }) => (
        <div key={key} className="flex items-center justify-between gap-3">
          <label className="text-[11px] flex-shrink-0" style={{ color: 'var(--text-3)' }}>
            {label}
          </label>
          <input
            type="number"
            step={step}
            value={values[key as keyof typeof values]}
            onChange={(e) => setValues(prev => ({ ...prev, [key]: parseFloat(e.target.value) || 0 }))}
            className="input w-24 text-right text-[11px] py-1 px-2"
          />
        </div>
      ))}
      <button
        onClick={handleSave}
        disabled={saveMutation.isPending}
        className="btn w-full text-[11px] py-1.5 mt-2"
      >
        {saveMutation.isPending ? 'Saving...' : saveMutation.isSuccess ? 'Saved' : 'Save Changes'}
      </button>
    </div>
  )
}
