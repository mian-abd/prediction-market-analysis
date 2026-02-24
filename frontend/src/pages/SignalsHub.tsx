import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  Zap,
  ArrowLeftRight,
  Trophy,
  ChevronRight,
  AlertTriangle,
  ArrowUpRight,
  ArrowDownRight,
  Shield,
  ShieldAlert,
  TrendingUp,
  CheckCircle,
  XCircle,
  BarChart3,
  Brain,
  Target,
  Newspaper,
  Clock,
  Activity,
  Users,
  Network,
  Layers,
} from 'lucide-react'
import apiClient from '../api/client'
import { Skeleton } from '../components/LoadingSkeleton'

type SignalTab = 'all' | 'ml' | 'arbitrage' | 'elo' | 'strategies'

interface EnsembleEdge {
  id: number
  market_id: number
  market_question: string
  market_slug: string | null
  detected_at: string | null
  direction: string
  ensemble_prob: number
  market_price: number
  raw_edge_pct: number
  net_ev_pct: number
  kelly_fraction: number
  confidence: number
  quality_tier: string
  model_predictions: Record<string, number> | null
}

interface EloEdge {
  id: number
  market_id: number
  market_question: string
  market_slug: string | null
  sport: string
  player_a: string
  player_b: string
  surface: string | null
  elo_prob_a: number
  market_price_yes: number
  raw_edge_pct: number
  net_edge_pct: number
  kelly_fraction: number
  elo_confidence: number
}

interface ArbOpp {
  id: number
  strategy_type: string
  detected_at: string | null
  market_ids: string | null
  net_profit_pct: number
  estimated_profit_usd: number
}

interface NewStrategySignal {
  id: number
  market_id: number
  market_question: string
  market_slug: string | null
  strategy: string
  detected_at: string | null
  direction: string
  implied_prob: number | null
  market_price: number
  raw_edge_pct: number
  net_ev_pct: number
  kelly_fraction: number
  confidence: number
  quality_tier: string | null
  metadata: Record<string, unknown> | null
}

interface SignalsResponse {
  ensemble_edges: EnsembleEdge[]
  elo_edges: EloEdge[]
  arbitrage_opportunities: ArbOpp[]
  new_strategy_signals: NewStrategySignal[]
  strategy_breakdown: Record<string, number>
  summary: {
    total_signals: number
    ensemble_count: number
    elo_count: number
    arbitrage_count: number
    new_strategy_count: number
    high_confidence_count: number
    avg_kelly_fraction: number
    strategies_active: string[]
  }
}

const TIER_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  high: { bg: 'rgba(76,175,112,0.1)', text: 'var(--green)', border: 'rgba(76,175,112,0.3)' },
  medium: { bg: 'rgba(196,162,77,0.1)', text: 'var(--accent)', border: 'rgba(196,162,77,0.3)' },
  low: { bg: 'rgba(255,255,255,0.03)', text: 'var(--text-3)', border: 'var(--border)' },
  speculative: { bg: 'rgba(207,102,121,0.08)', text: 'var(--red)', border: 'rgba(207,102,121,0.25)' },
}

const STRATEGY_META: Record<string, { label: string; icon: typeof Zap; color: string }> = {
  llm_forecast: { label: 'LLM', icon: Brain, color: 'var(--purple, #a78bfa)' },
  longshot_bias: { label: 'Longshot', icon: Target, color: 'var(--accent)' },
  news_catalyst: { label: 'News', icon: Newspaper, color: 'var(--blue)' },
  resolution_convergence: { label: 'Theta', icon: Clock, color: 'var(--green)' },
  orderflow: { label: 'Flow', icon: Activity, color: '#f59e0b' },
  smart_money: { label: 'Whale', icon: Users, color: '#ec4899' },
  market_clustering: { label: 'Cluster', icon: Network, color: '#06b6d4' },
  consensus: { label: 'Consensus', icon: Shield, color: 'var(--green)' },
}

function getModelAgreement(preds: Record<string, number> | null, direction: string): { agree: boolean; count: number; total: number } {
  if (!preds) return { agree: false, count: 0, total: 0 }
  const models = Object.entries(preds)
  const total = models.length
  const threshold = 0.5
  const count = models.filter(([, prob]) =>
    direction === 'buy_yes' ? prob > threshold : prob < threshold
  ).length
  return { agree: count === total && total >= 2, count, total }
}

function timeAgo(iso: string | null): string {
  if (!iso) return ''
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

export default function SignalsHub() {
  const [activeTab, setActiveTab] = useState<SignalTab>('all')

  const { data, isLoading } = useQuery<SignalsResponse>({
    queryKey: ['strategy-signals'],
    queryFn: async () => (await apiClient.get('/strategies/signals?limit=100')).data,
    refetchInterval: 60_000,
    retry: 1,
  })

  const { data: perfData, isLoading: perfLoading } = useQuery<{
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
  }>({
    queryKey: ['signal-performance'],
    queryFn: async () => (await apiClient.get('/strategies/signal-performance')).data,
    refetchInterval: 300_000,
    retry: 1,
  })

  const ensemble = data?.ensemble_edges ?? []
  const elo = data?.elo_edges ?? []
  const arb = data?.arbitrage_opportunities ?? []
  const newSignals = data?.new_strategy_signals ?? []
  const summary = data?.summary

  const tabs: { key: SignalTab; label: string; count: number; icon: typeof Zap }[] = [
    { key: 'all', label: 'All Signals', count: summary?.total_signals ?? 0, icon: TrendingUp },
    { key: 'ml', label: 'ML Ensemble', count: summary?.ensemble_count ?? 0, icon: Zap },
    { key: 'elo', label: 'Elo Sports', count: summary?.elo_count ?? 0, icon: Trophy },
    { key: 'strategies', label: 'New Strategies', count: summary?.new_strategy_count ?? 0, icon: Layers },
    { key: 'arbitrage', label: 'Arbitrage', count: summary?.arbitrage_count ?? 0, icon: ArrowLeftRight },
  ]

  return (
    <div className="space-y-6 fade-up">
      {/* Header */}
      <div>
        <h1 className="text-[24px] font-bold" style={{ color: 'var(--text)' }}>
          Signals
        </h1>
        <p className="text-[13px] mt-1" style={{ color: 'var(--text-2)' }}>
          Active trading opportunities across all strategies
        </p>
      </div>

      {/* Experimental Warning */}
      <div
        className="flex items-start gap-3 p-4 rounded-xl"
        style={{ background: 'rgba(207,102,121,0.06)', border: '1px solid rgba(207,102,121,0.15)' }}
      >
        <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-0.5" style={{ color: 'var(--red)' }} />
        <div>
          <p className="text-[13px] font-medium" style={{ color: 'var(--text)' }}>
            Experimental Signals
          </p>
          <p className="text-[12px] mt-0.5" style={{ color: 'var(--text-3)' }}>
            Signals are for research only. Edges &gt;15% are marked speculative — such edges on liquid markets are likely noise.
          </p>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          { label: 'Total Signals', value: summary?.total_signals ?? 0, color: 'var(--text)' },
          { label: 'High Confidence', value: summary?.high_confidence_count ?? 0, color: 'var(--green)' },
          { label: 'Avg Kelly', value: `${((summary?.avg_kelly_fraction ?? 0) * 100).toFixed(1)}%`, color: 'var(--accent)' },
          { label: 'Strategies', value: (summary?.strategies_active?.length ?? 0) + [ensemble.length > 0, elo.length > 0, arb.length > 0].filter(Boolean).length, color: 'var(--blue)' },
        ].map((item) => (
          <div key={item.label} className="card p-4">
            <p className="text-[10px] font-medium uppercase tracking-wide mb-1" style={{ color: 'var(--text-3)' }}>
              {item.label}
            </p>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <p className="text-[24px] font-bold" style={{ color: item.color }}>
                {item.value}
              </p>
            )}
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="tab-bar">
        {tabs.map(({ key, label, count, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`tab ${activeTab === key ? 'tab-active' : ''}`}
          >
            <Icon className="h-3.5 w-3.5" />
            {label} ({count})
          </button>
        ))}
      </div>

      {/* Signal List */}
      <div className="space-y-2">
        {isLoading ? (
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="card p-4 space-y-2">
              <div className="flex gap-2"><Skeleton className="h-5 w-16" /><Skeleton className="h-5 w-12" /></div>
              <Skeleton className="h-5 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </div>
          ))
        ) : (
          <>
            {/* ML Ensemble Signals */}
            {(activeTab === 'all' || activeTab === 'ml') && ensemble.map((s) => {
              const isBuyYes = s.direction === 'buy_yes'
              const DirIcon = isBuyYes ? ArrowUpRight : ArrowDownRight
              const dirColor = isBuyYes ? 'var(--green)' : 'var(--red)'
              const tier = TIER_COLORS[s.quality_tier] ?? TIER_COLORS.low
              const agreement = getModelAgreement(s.model_predictions, s.direction)
              const isSpeculative = s.raw_edge_pct > 15

              return (
                <Link
                  key={`ml-${s.id}`}
                  to={`/markets/${s.market_id}`}
                  className="card card-hover p-4 block group relative"
                  style={{ textDecoration: 'none', borderLeft: `3px solid ${tier.text}` }}
                >
                  {/* Top row: badges */}
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <span className="signal-badge">
                      <Zap className="h-3 w-3" /> ML
                    </span>
                    <span
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold"
                      style={{ background: tier.bg, color: tier.text, border: `1px solid ${tier.border}` }}
                    >
                      {s.quality_tier}
                    </span>
                    <span
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold"
                      style={{ background: isBuyYes ? 'rgba(76,175,112,0.1)' : 'rgba(207,102,121,0.1)', color: dirColor }}
                    >
                      <DirIcon className="h-3 w-3" />
                      {isBuyYes ? 'BUY YES' : 'BUY NO'}
                    </span>
                    {agreement.agree && (
                      <span
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold"
                        style={{ background: 'rgba(76,175,112,0.1)', color: 'var(--green)', border: '1px solid rgba(76,175,112,0.2)' }}
                      >
                        <Shield className="h-3 w-3" />
                        {agreement.count}/{agreement.total} models agree
                      </span>
                    )}
                    {isSpeculative && (
                      <span
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold"
                        style={{ background: 'rgba(207,102,121,0.08)', color: 'var(--red)' }}
                      >
                        <ShieldAlert className="h-3 w-3" />
                        Edge &gt;15%
                      </span>
                    )}
                    {s.detected_at && (
                      <span className="text-[10px] ml-auto" style={{ color: 'var(--text-3)' }}>
                        {timeAgo(s.detected_at)}
                      </span>
                    )}
                  </div>

                  {/* Market question */}
                  <p className="text-[14px] font-medium truncate mb-2" style={{ color: 'var(--text)' }}>
                    {s.market_question}
                  </p>

                  {/* Metrics row */}
                  <div className="flex items-center gap-4 flex-wrap">
                    <div className="text-center">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Net EV</p>
                      <p className="text-[15px] font-bold font-mono" style={{ color: 'var(--green)' }}>
                        +{s.net_ev_pct.toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Kelly</p>
                      <p className="text-[15px] font-bold font-mono" style={{ color: 'var(--accent)' }}>
                        {(s.kelly_fraction * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Confidence</p>
                      <p className="text-[15px] font-bold font-mono" style={{ color: 'var(--text)' }}>
                        {(s.confidence * 100).toFixed(0)}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Model</p>
                      <p className="text-[13px] font-mono" style={{ color: 'var(--text-2)' }}>
                        {(s.ensemble_prob * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Market</p>
                      <p className="text-[13px] font-mono" style={{ color: 'var(--text-2)' }}>
                        {(s.market_price * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="ml-auto flex-shrink-0">
                      <ChevronRight
                        className="h-4 w-4 opacity-0 group-hover:opacity-100 transition-opacity"
                        style={{ color: 'var(--text-3)' }}
                      />
                    </div>
                  </div>
                </Link>
              )
            })}

            {/* Elo Sports Signals */}
            {(activeTab === 'all' || activeTab === 'elo') && elo.map((s) => {
              const isBuyYes = s.elo_prob_a > s.market_price_yes
              const dirColor = isBuyYes ? 'var(--green)' : 'var(--red)'
              return (
                <Link
                  key={`elo-${s.id}`}
                  to={`/markets/${s.market_id}`}
                  className="card card-hover p-4 block group relative"
                  style={{ textDecoration: 'none', borderLeft: '3px solid var(--blue)' }}
                >
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <span className="signal-badge" style={{ background: 'rgba(94,180,239,0.12)', color: 'var(--blue)' }}>
                      <Trophy className="h-3 w-3" /> ELO
                    </span>
                    <span className="pill">{s.sport}</span>
                    {s.surface && <span className="pill">{s.surface}</span>}
                  </div>

                  <p className="text-[14px] font-medium truncate mb-1" style={{ color: 'var(--text)' }}>
                    {s.player_a} vs {s.player_b}
                  </p>
                  <p className="text-[12px] truncate mb-2" style={{ color: 'var(--text-3)' }}>
                    {s.market_question}
                  </p>

                  <div className="flex items-center gap-4 flex-wrap">
                    <div className="text-center">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Net Edge</p>
                      <p className="text-[15px] font-bold font-mono" style={{ color: dirColor }}>
                        +{s.net_edge_pct.toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Kelly</p>
                      <p className="text-[15px] font-bold font-mono" style={{ color: 'var(--accent)' }}>
                        {(s.kelly_fraction * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Elo Prob</p>
                      <p className="text-[13px] font-mono" style={{ color: 'var(--text-2)' }}>
                        {(s.elo_prob_a * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Market</p>
                      <p className="text-[13px] font-mono" style={{ color: 'var(--text-2)' }}>
                        {(s.market_price_yes * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Confidence</p>
                      <p className="text-[13px] font-mono" style={{ color: 'var(--text-2)' }}>
                        {(s.elo_confidence * 100).toFixed(0)}%
                      </p>
                    </div>
                    <div className="ml-auto flex-shrink-0">
                      <ChevronRight
                        className="h-4 w-4 opacity-0 group-hover:opacity-100 transition-opacity"
                        style={{ color: 'var(--text-3)' }}
                      />
                    </div>
                  </div>
                </Link>
              )
            })}

            {/* New Strategy Signals */}
            {(activeTab === 'all' || activeTab === 'strategies') && newSignals.map((s) => {
              const meta = STRATEGY_META[s.strategy] ?? { label: s.strategy, icon: Zap, color: 'var(--text-3)' }
              const StratIcon = meta.icon
              const isBuyYes = s.direction === 'buy_yes'
              const dirColor = isBuyYes ? 'var(--green)' : 'var(--red)'
              const DirIcon = isBuyYes ? ArrowUpRight : ArrowDownRight
              const tier = TIER_COLORS[s.quality_tier ?? 'medium'] ?? TIER_COLORS.medium

              return (
                <Link
                  key={`strat-${s.id}`}
                  to={`/markets/${s.market_id}`}
                  className="card card-hover p-4 block group relative"
                  style={{ textDecoration: 'none', borderLeft: `3px solid ${meta.color}` }}
                >
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <span
                      className="signal-badge"
                      style={{ background: `${meta.color}18`, color: meta.color }}
                    >
                      <StratIcon className="h-3 w-3" /> {meta.label}
                    </span>
                    {s.quality_tier && (
                      <span
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold"
                        style={{ background: tier.bg, color: tier.text, border: `1px solid ${tier.border}` }}
                      >
                        {s.quality_tier}
                      </span>
                    )}
                    <span
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold"
                      style={{ background: isBuyYes ? 'rgba(76,175,112,0.1)' : 'rgba(207,102,121,0.1)', color: dirColor }}
                    >
                      <DirIcon className="h-3 w-3" />
                      {isBuyYes ? 'BUY YES' : 'BUY NO'}
                    </span>
                    {s.detected_at && (
                      <span className="text-[10px] ml-auto" style={{ color: 'var(--text-3)' }}>
                        {timeAgo(s.detected_at)}
                      </span>
                    )}
                  </div>

                  <p className="text-[14px] font-medium truncate mb-2" style={{ color: 'var(--text)' }}>
                    {s.market_question}
                  </p>

                  <div className="flex items-center gap-4 flex-wrap">
                    <div className="text-center">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Net EV</p>
                      <p className="text-[15px] font-bold font-mono" style={{ color: 'var(--green)' }}>
                        +{s.net_ev_pct.toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Kelly</p>
                      <p className="text-[15px] font-bold font-mono" style={{ color: 'var(--accent)' }}>
                        {(s.kelly_fraction * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Confidence</p>
                      <p className="text-[15px] font-bold font-mono" style={{ color: 'var(--text)' }}>
                        {((s.confidence ?? 0) * 100).toFixed(0)}%
                      </p>
                    </div>
                    {s.implied_prob != null && (
                      <div className="text-center">
                        <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Model</p>
                        <p className="text-[13px] font-mono" style={{ color: 'var(--text-2)' }}>
                          {(s.implied_prob * 100).toFixed(1)}%
                        </p>
                      </div>
                    )}
                    <div className="text-center">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Market</p>
                      <p className="text-[13px] font-mono" style={{ color: 'var(--text-2)' }}>
                        {(s.market_price * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="ml-auto flex-shrink-0">
                      <ChevronRight
                        className="h-4 w-4 opacity-0 group-hover:opacity-100 transition-opacity"
                        style={{ color: 'var(--text-3)' }}
                      />
                    </div>
                  </div>
                </Link>
              )
            })}

            {/* Arbitrage Signals */}
            {(activeTab === 'all' || activeTab === 'arbitrage') && arb.map((a) => (
              <div
                key={`arb-${a.id}`}
                className="card p-4"
                style={{ borderLeft: '3px solid var(--blue)' }}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="signal-badge" style={{ background: 'rgba(94,180,239,0.12)', color: 'var(--blue)' }}>
                        <ArrowLeftRight className="h-3 w-3" /> ARB
                      </span>
                      <span className="pill">{a.strategy_type || 'cross-platform'}</span>
                      {a.detected_at && (
                        <span className="text-[10px] ml-auto" style={{ color: 'var(--text-3)' }}>
                          {timeAgo(a.detected_at)}
                        </span>
                      )}
                    </div>
                    <p className="text-[12px] mt-1" style={{ color: 'var(--text-3)' }}>
                      Est. profit: ${a.estimated_profit_usd.toFixed(2)} per $100
                    </p>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <p className="text-[16px] font-bold font-mono" style={{ color: 'var(--green)' }}>
                      +{a.net_profit_pct.toFixed(1)}%
                    </p>
                    <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                      net profit
                    </p>
                  </div>
                </div>
              </div>
            ))}

            {/* Empty State */}
            {(summary?.total_signals ?? 0) === 0 && !isLoading && (
              <div className="flex flex-col items-center justify-center h-48 gap-3">
                <Zap className="h-6 w-6" style={{ color: 'var(--text-3)' }} />
                <div className="text-center">
                  <p className="text-[14px] font-medium" style={{ color: 'var(--text-2)' }}>
                    No active signals
                  </p>
                  <p className="text-[12px] mt-1" style={{ color: 'var(--text-3)' }}>
                    Signals appear when the ensemble scanner detects fee-adjusted edges.
                  </p>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Signal Scorecard — Historical Performance Proof */}
      <div className="card p-5">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="h-5 w-5" style={{ color: 'var(--accent)' }} />
          <h2 className="text-[16px] font-semibold" style={{ color: 'var(--text)' }}>
            Signal Scorecard
          </h2>
          <span className="text-[11px] ml-2" style={{ color: 'var(--text-3)' }}>
            Resolved signals vs actual outcomes
          </span>
        </div>

        {perfLoading ? (
          <div className="space-y-3">
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} className="h-10 rounded-lg" />
            ))}
          </div>
        ) : !perfData?.summary || perfData.summary.total_scored === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 gap-2">
            <CheckCircle className="h-5 w-5" style={{ color: 'var(--text-3)' }} />
            <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
              No resolved signals yet — results appear after markets close.
            </p>
          </div>
        ) : (
          <>
            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
              <div className="p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.02)' }}>
                <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Scored</p>
                <p className="text-[20px] font-mono font-bold" style={{ color: 'var(--text)' }}>
                  {perfData.summary.total_scored}
                </p>
              </div>
              <div className="p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.02)' }}>
                <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Hit Rate</p>
                <p
                  className="text-[20px] font-mono font-bold"
                  style={{ color: (perfData.summary.hit_rate ?? 0) >= 0.5 ? 'var(--green)' : 'var(--red)' }}
                >
                  {((perfData.summary.hit_rate ?? 0) * 100).toFixed(1)}%
                </p>
              </div>
              <div className="p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.02)' }}>
                <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Cumulative P&L</p>
                <p
                  className="text-[20px] font-mono font-bold"
                  style={{ color: (perfData.summary.cumulative_pnl ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}
                >
                  {(perfData.summary.cumulative_pnl ?? 0) >= 0 ? '+' : ''}
                  {(perfData.summary.cumulative_pnl ?? 0).toFixed(2)}%
                </p>
              </div>
              <div className="p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.02)' }}>
                <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Correct</p>
                <p className="text-[20px] font-mono font-bold" style={{ color: 'var(--green)' }}>
                  {perfData.summary.total_correct}/{perfData.summary.total_scored}
                </p>
              </div>
            </div>

            {/* Daily Breakdown Table */}
            <div className="space-y-1.5">
              {/* Header */}
              <div
                className="grid grid-cols-5 gap-2 px-3 py-2 text-[10px] uppercase"
                style={{ color: 'var(--text-3)' }}
              >
                <span>Date</span>
                <span className="text-center">Signals</span>
                <span className="text-center">Correct</span>
                <span className="text-right">Daily P&L</span>
                <span className="text-right">Cumulative</span>
              </div>

              {[...(perfData.data || [])]
                .reverse()
                .slice(0, 30)
                .map((row) => {
                  const hitRate = row.signals_generated > 0 ? row.signals_correct / row.signals_generated : 0
                  const isGoodDay = hitRate >= 0.5

                  return (
                    <div
                      key={row.date}
                      className="grid grid-cols-5 gap-2 px-3 py-2.5 rounded-lg"
                      style={{
                        background: isGoodDay
                          ? 'rgba(76,175,112,0.06)'
                          : 'rgba(207,102,121,0.06)',
                      }}
                    >
                      <span className="text-[12px] font-mono" style={{ color: 'var(--text-2)' }}>
                        {new Date(row.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                      </span>
                      <span className="text-[12px] font-mono text-center" style={{ color: 'var(--text)' }}>
                        {row.signals_generated}
                      </span>
                      <span className="text-[12px] font-mono text-center flex items-center justify-center gap-1">
                        {isGoodDay ? (
                          <CheckCircle className="h-3 w-3" style={{ color: 'var(--green)' }} />
                        ) : (
                          <XCircle className="h-3 w-3" style={{ color: 'var(--red)' }} />
                        )}
                        <span style={{ color: isGoodDay ? 'var(--green)' : 'var(--red)' }}>
                          {row.signals_correct}/{row.signals_generated}
                        </span>
                      </span>
                      <span
                        className="text-[12px] font-mono font-semibold text-right"
                        style={{ color: (row.daily_pnl ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}
                      >
                        {(row.daily_pnl ?? 0) >= 0 ? '+' : ''}{(row.daily_pnl ?? 0).toFixed(2)}%
                      </span>
                      <span
                        className="text-[12px] font-mono text-right"
                        style={{ color: (row.cumulative_pnl ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}
                      >
                        {(row.cumulative_pnl ?? 0) >= 0 ? '+' : ''}{(row.cumulative_pnl ?? 0).toFixed(2)}%
                      </span>
                    </div>
                  )
                })}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
