import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  Store,
  Zap,
  Network,
  Shield,
  ChevronRight,
  TrendingUp,
  TrendingDown,
  Database,
  Activity,
  Clock,
  Briefcase,
  Brain,
} from 'lucide-react'
import apiClient from '../api/client'
import ErrorState from '../components/ErrorState'
import { Skeleton } from '../components/LoadingSkeleton'

interface RiskStatusSingle {
  exposure: { current: number; limit: number; utilization_pct: number }
  daily_pnl: { current: number; limit: number; utilization_pct: number }
  daily_trades: { current: number; limit: number; utilization_pct: number }
  max_position_usd: number
  open_positions: number
  circuit_breaker_active: boolean
}

type RiskStatus = {
  manual: RiskStatusSingle
  auto: RiskStatusSingle
}

interface AutoTradingStatus {
  enabled_strategies: string[]
  total_exposure: number
  today_pnl: number
  open_positions: Record<string, number>
}

interface PortfolioSummaryBrief {
  total_pnl: number
  total_realized_pnl: number
  total_unrealized_pnl: number
  open_positions: number
  win_rate: number
  sharpe_ratio: number | null
}

interface DashboardStats {
  total_active_markets: number
  active_arbitrage_opportunities: number
  markets_by_platform: Record<string, number>
  price_snapshots: number
  cross_platform_matches: number
  last_data_fetch: string | null
}

interface MispricedMarket {
  market_id: number
  question: string
  category: string | null
  price_yes: number
  calibrated_price: number
  delta_pct: number
  direction: string
  edge_estimate: number
}

function getRelativeTime(iso: string | null): string {
  if (!iso) return 'Unknown'
  const ms = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(ms / 60000)
  if (mins < 1) return 'Just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

function getFreshnessClass(iso: string | null): string {
  if (!iso) return 'freshness-stale'
  const ms = Date.now() - new Date(iso).getTime()
  if (ms < 300000) return 'freshness-fresh'
  if (ms < 1800000) return 'freshness-ok'
  return 'freshness-stale'
}

export default function Dashboard() {
  const { data: stats, isLoading, error, refetch } = useQuery<DashboardStats>({
    queryKey: ['dashboard-stats'],
    queryFn: async () => (await apiClient.get('/system/stats')).data,
    refetchInterval: 30_000,
  })

  const { data: signalsData, isLoading: signalsLoading } = useQuery<{ markets: MispricedMarket[] }>({
    queryKey: ['top-signals-preview'],
    queryFn: async () => (await apiClient.get('/predictions/top/mispriced', { params: { limit: 5 } })).data,
    refetchInterval: 60_000,
    retry: 1,
  })

  const { data: riskData } = useQuery<RiskStatus>({
    queryKey: ['risk-status'],
    queryFn: async () => (await apiClient.get('/portfolio/risk-status')).data,
    refetchInterval: 30_000,
    retry: 1,
  })

  const { data: autoStatus } = useQuery<AutoTradingStatus>({
    queryKey: ['auto-trading-status-dash'],
    queryFn: async () => (await apiClient.get('/auto-trading/status')).data,
    refetchInterval: 30_000,
    retry: 1,
  })

  const { data: portfolioSummary } = useQuery<PortfolioSummaryBrief>({
    queryKey: ['portfolio-summary-dash'],
    queryFn: async () => (await apiClient.get('/portfolio/summary')).data,
    refetchInterval: 15_000,
    retry: 1,
  })

  if (error) {
    return <ErrorState title="Dashboard unavailable" message="Cannot reach the API server." onRetry={() => refetch()} />
  }

  const s = stats
  const signals = signalsData?.markets ?? []
  const platforms = s?.markets_by_platform ?? {}
  const totalPlatformMarkets = Object.values(platforms).reduce((a, b) => a + b, 0)

  return (
    <div className="space-y-6 fade-up">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-[26px] font-bold" style={{ color: 'var(--text)' }}>Dashboard</h1>
          <p className="text-[13px] mt-1" style={{ color: 'var(--text-2)' }}>
            Prediction market intelligence overview
          </p>
        </div>
        {isLoading ? (
          <Skeleton className="h-7 w-28" />
        ) : (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg" style={{ background: 'rgba(76,175,112,0.08)' }}>
            <span className="h-2 w-2 rounded-full pulse-dot" style={{ background: 'var(--green)' }} />
            <span className="text-[12px] font-medium" style={{ color: 'var(--green)' }}>Pipeline Active</span>
          </div>
        )}
      </div>

      {/* P&L Ticker */}
      {portfolioSummary && (
        <div className="card p-5">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <p className="text-[10px] uppercase font-semibold tracking-wider mb-1" style={{ color: 'var(--text-3)' }}>
                Portfolio P&L
              </p>
              <div className="flex items-center gap-3">
                {portfolioSummary.total_pnl >= 0 ? (
                  <TrendingUp className="h-6 w-6" style={{ color: 'var(--green)' }} />
                ) : (
                  <TrendingDown className="h-6 w-6" style={{ color: 'var(--red)' }} />
                )}
                <span
                  className="text-[32px] font-mono font-bold"
                  style={{ color: portfolioSummary.total_pnl >= 0 ? 'var(--green)' : 'var(--red)' }}
                >
                  {portfolioSummary.total_pnl >= 0 ? '+' : ''}${portfolioSummary.total_pnl.toFixed(2)}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-6">
              <div className="text-center">
                <p className="text-[10px] uppercase mb-0.5" style={{ color: 'var(--text-3)' }}>Realized</p>
                <p className="text-[16px] font-mono font-medium" style={{
                  color: portfolioSummary.total_realized_pnl >= 0 ? 'var(--green)' : 'var(--red)'
                }}>
                  ${portfolioSummary.total_realized_pnl.toFixed(2)}
                </p>
              </div>
              <div className="text-center">
                <p className="text-[10px] uppercase mb-0.5" style={{ color: 'var(--text-3)' }}>Unrealized</p>
                <p className="text-[16px] font-mono font-medium" style={{
                  color: portfolioSummary.total_unrealized_pnl >= 0 ? 'var(--green)' : 'var(--red)'
                }}>
                  ${portfolioSummary.total_unrealized_pnl.toFixed(2)}
                </p>
              </div>
              <div className="text-center">
                <p className="text-[10px] uppercase mb-0.5" style={{ color: 'var(--text-3)' }}>Win Rate</p>
                <p className="text-[16px] font-mono font-medium" style={{ color: 'var(--accent)' }}>
                  {portfolioSummary.win_rate.toFixed(1)}%
                </p>
              </div>
              {portfolioSummary.sharpe_ratio != null && (
                <div className="text-center">
                  <p className="text-[10px] uppercase mb-0.5" style={{ color: 'var(--text-3)' }}>Sharpe</p>
                  <p className="text-[16px] font-mono font-medium" style={{
                    color: portfolioSummary.sharpe_ratio >= 1 ? 'var(--green)' : portfolioSummary.sharpe_ratio >= 0 ? 'var(--accent)' : 'var(--red)'
                  }}>
                    {portfolioSummary.sharpe_ratio.toFixed(2)}
                  </p>
                </div>
              )}
              <div className="text-center">
                <p className="text-[10px] uppercase mb-0.5" style={{ color: 'var(--text-3)' }}>Positions</p>
                <p className="text-[16px] font-semibold" style={{ color: 'var(--text)' }}>
                  {portfolioSummary.open_positions}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          {
            icon: Store,
            label: 'Active Markets',
            value: s?.total_active_markets ?? 0,
            format: (v: number) => v >= 1000 ? `${(v / 1000).toFixed(1)}K` : v.toLocaleString(),
            color: 'var(--text)',
          },
          {
            icon: Zap,
            label: 'ML Signals',
            value: signals.length,
            format: (v: number) => v.toString(),
            color: 'var(--accent)',
          },
          {
            icon: Database,
            label: 'Price Snapshots',
            value: s?.price_snapshots ?? 0,
            format: (v: number) => v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1000 ? `${(v / 1000).toFixed(0)}K` : v.toLocaleString(),
            color: 'var(--text)',
          },
          {
            icon: Clock,
            label: 'Last Updated',
            value: 0,
            format: () => getRelativeTime(s?.last_data_fetch ?? null),
            color: 'var(--text)',
            isFreshness: true,
          },
        ].map((item) => {
          const Icon = item.icon
          return (
            <div key={item.label} className="card p-4">
              <div className="flex items-center gap-2 mb-2">
                <Icon className="h-4 w-4" style={{ color: 'var(--text-3)' }} />
                <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-3)' }}>
                  {item.label}
                </span>
              </div>
              {isLoading ? (
                <Skeleton className="h-7 w-20" />
              ) : (item as any).isFreshness ? (
                <span className={`freshness ${getFreshnessClass(s?.last_data_fetch ?? null)}`}>
                  <span className="freshness-dot" />
                  <span className="text-[16px] font-bold">{item.format(item.value)}</span>
                </span>
              ) : (
                <p className="text-[22px] font-bold" style={{ color: item.color }}>{item.format(item.value)}</p>
              )}
            </div>
          )
        })}
      </div>

      {/* Two-Column: Top Signals + Platform Coverage */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Top Signals */}
        <div className="lg:col-span-2 card p-5">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4" style={{ color: 'var(--accent)' }} />
              <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>Top ML Signals</p>
              <span className="badge badge-experimental">Experimental</span>
            </div>
            <Link
              to="/signals"
              className="text-[12px] font-medium flex items-center gap-1"
              style={{ color: 'var(--accent)', textDecoration: 'none' }}
            >
              View All <ChevronRight className="h-3.5 w-3.5" />
            </Link>
          </div>

          {signalsLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="flex items-center gap-3 p-3 rounded-xl" style={{ background: 'rgba(255,255,255,0.02)' }}>
                  <Skeleton className="h-10 w-10 rounded-xl" />
                  <div className="flex-1 space-y-1.5">
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-3 w-1/3" />
                  </div>
                  <Skeleton className="h-6 w-16" />
                </div>
              ))}
            </div>
          ) : signals.length > 0 ? (
            <div className="space-y-2">
              {signals.slice(0, 5).map((sig) => {
                const isUnder = sig.direction === 'underpriced'
                return (
                  <Link
                    key={sig.market_id}
                    to={`/markets/${sig.market_id}`}
                    className="flex items-center gap-3 p-3 rounded-xl group transition-colors"
                    style={{ background: 'rgba(255,255,255,0.02)', textDecoration: 'none' }}
                  >
                    <div
                      className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                      style={{
                        background: isUnder ? 'rgba(76,175,112,0.1)' : 'rgba(207,102,121,0.1)',
                      }}
                    >
                      {isUnder ? (
                        <TrendingUp className="h-4 w-4" style={{ color: 'var(--green)' }} />
                      ) : (
                        <TrendingDown className="h-4 w-4" style={{ color: 'var(--red)' }} />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-[13px] font-medium truncate" style={{ color: 'var(--text)' }}>
                        {sig.question}
                      </p>
                      <div className="flex items-center gap-2 mt-0.5">
                        {sig.category && <span className="pill text-[10px]">{sig.category}</span>}
                        <span className="text-[11px] font-medium"
                          style={{ color: isUnder ? 'var(--green)' : 'var(--red)' }}>
                          {isUnder ? 'Underpriced' : 'Overpriced'}
                        </span>
                      </div>
                    </div>
                    <div className="text-right flex-shrink-0">
                      <p className="text-[15px] font-bold font-mono" style={{ color: 'var(--accent)' }}>
                        +{(Math.abs(sig.edge_estimate) * 100).toFixed(1)}%
                      </p>
                      <p className="text-[10px]" style={{ color: 'var(--text-3)' }}>edge</p>
                    </div>
                    <ChevronRight
                      className="h-4 w-4 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                      style={{ color: 'var(--text-3)' }}
                    />
                  </Link>
                )
              })}
            </div>
          ) : (
            <div className="flex flex-col items-center py-8 gap-2">
              <Zap className="h-5 w-5" style={{ color: 'var(--text-3)' }} />
              <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>No active ML signals</p>
              <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>Signals appear when the model detects mispriced markets</p>
            </div>
          )}
        </div>

        {/* Platform Coverage */}
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="h-4 w-4" style={{ color: 'var(--text-3)' }} />
            <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>Platform Coverage</p>
          </div>
          {isLoading ? (
            <div className="space-y-4">
              {[1, 2].map(i => <Skeleton key={i} className="h-12 w-full" />)}
            </div>
          ) : (
            <div className="space-y-4">
              {Object.entries(platforms)
                .sort(([, a], [, b]) => b - a)
                .map(([platform, count]) => {
                  const pct = totalPlatformMarkets > 0 ? (count / totalPlatformMarkets * 100) : 0
                  return (
                    <div key={platform}>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-[13px] font-medium capitalize" style={{ color: 'var(--text)' }}>
                          {platform}
                        </span>
                        <span className="text-[11px] font-mono" style={{ color: 'var(--text-3)' }}>
                          {count >= 1000 ? `${(count / 1000).toFixed(1)}K` : count}
                        </span>
                      </div>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${pct}%`, background: 'var(--accent)' }} />
                      </div>
                    </div>
                  )
                })}

              <div className="pt-3 mt-1" style={{ borderTop: '1px solid var(--border)' }}>
                <div className="flex items-center justify-between">
                  <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>Cross-Platform Matches</span>
                  <span className="text-[13px] font-bold" style={{ color: 'var(--text)' }}>
                    {(s?.cross_platform_matches ?? 0).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Risk Status Card */}
      {riskData && riskData.manual && (
        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4" style={{ color: riskData.manual.circuit_breaker_active ? 'var(--red)' : 'var(--green)' }} />
              <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>Risk Status</p>
              {riskData.manual.circuit_breaker_active && (
                <span className="pill pill-red">Circuit Breaker Active</span>
              )}
            </div>
            {/* Auto Trading Status Indicator */}
            {autoStatus && (
              <div className="flex items-center gap-3 px-3 py-1.5 rounded-lg" style={{ background: 'rgba(255,255,255,0.03)' }}>
                <span className="text-[10px] font-semibold uppercase" style={{ color: 'var(--text-3)' }}>Auto Trading</span>
                {['ensemble', 'elo'].map((strat) => {
                  const isOn = autoStatus.enabled_strategies.includes(strat)
                  return (
                    <span key={strat} className="flex items-center gap-1 text-[11px] font-medium">
                      <span
                        className="h-1.5 w-1.5 rounded-full"
                        style={{ background: isOn ? 'var(--green)' : 'var(--red)' }}
                      />
                      <span style={{ color: isOn ? 'var(--green)' : 'var(--text-3)' }}>
                        {strat === 'ensemble' ? 'ML' : 'Elo'}: {isOn ? 'ON' : 'OFF'}
                      </span>
                    </span>
                  )
                })}
              </div>
            )}
          </div>

          {/* Manual portfolio risk bars */}
          <p className="text-[10px] uppercase font-semibold mb-2" style={{ color: 'var(--text-3)' }}>Manual Portfolio</p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {[
              {
                label: 'Exposure',
                current: `$${riskData.manual.exposure.current.toFixed(0)}`,
                limit: `$${riskData.manual.exposure.limit.toFixed(0)}`,
                pct: riskData.manual.exposure.utilization_pct,
              },
              {
                label: 'Daily P&L',
                current: `$${riskData.manual.daily_pnl.current.toFixed(2)}`,
                limit: `$${riskData.manual.daily_pnl.limit.toFixed(0)}`,
                pct: riskData.manual.daily_pnl.utilization_pct,
              },
              {
                label: 'Daily Trades',
                current: `${riskData.manual.daily_trades.current}`,
                limit: `${riskData.manual.daily_trades.limit}`,
                pct: riskData.manual.daily_trades.utilization_pct,
              },
            ].map((item) => {
              const barColor = item.pct > 80 ? 'var(--red)' : item.pct > 50 ? '#F59E0B' : 'var(--green)'
              return (
                <div key={item.label}>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[11px] font-medium" style={{ color: 'var(--text-2)' }}>{item.label}</span>
                    <span className="text-[11px] font-mono" style={{ color: 'var(--text-3)' }}>
                      {item.current} / {item.limit}
                    </span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${Math.min(100, item.pct)}%`, background: barColor }}
                    />
                  </div>
                  <span className="text-[10px] font-mono" style={{ color: barColor }}>
                    {item.pct.toFixed(0)}%
                  </span>
                </div>
              )
            })}
          </div>

          {/* Auto portfolio summary (compact) */}
          {riskData.auto && (
            <div className="mt-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
              <p className="text-[10px] uppercase font-semibold mb-2" style={{ color: 'var(--accent)' }}>Auto Portfolio</p>
              <div className="flex items-center gap-6">
                <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                  Exposure: <span className="font-mono font-medium" style={{ color: 'var(--text)' }}>
                    ${riskData.auto.exposure.current.toFixed(0)} / ${riskData.auto.exposure.limit.toFixed(0)}
                  </span>
                </span>
                <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                  Open: <span className="font-mono font-medium" style={{ color: 'var(--text)' }}>
                    {riskData.auto.open_positions}
                  </span>
                </span>
                <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                  Day P&L: <span className="font-mono font-medium" style={{
                    color: riskData.auto.daily_pnl.current >= 0 ? 'var(--green)' : 'var(--red)'
                  }}>
                    ${riskData.auto.daily_pnl.current.toFixed(2)}
                  </span>
                </span>
              </div>
            </div>
          )}

          <div className="flex items-center gap-4 mt-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
            <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
              Open: {riskData.manual.open_positions} manual + {riskData.auto?.open_positions ?? 0} auto
            </span>
            <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
              Max position: ${riskData.manual.max_position_usd}
            </span>
          </div>
        </div>
      )}

      {/* Quick Navigation */}
      <div>
        <p className="text-[10px] font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-3)' }}>
          Navigate
        </p>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {[
            { to: '/markets', icon: Store, label: 'Markets', desc: 'Browse & analyze' },
            { to: '/signals', icon: Zap, label: 'Signals Hub', desc: 'All trading edges' },
            { to: '/correlation', icon: Network, label: 'Correlation', desc: 'Risk clusters' },
            { to: '/portfolio', icon: Briefcase, label: 'Portfolio', desc: 'Track positions' },
            { to: '/models', icon: Brain, label: 'ML Models', desc: 'Ensemble details' },
            { to: '/system', icon: Shield, label: 'System', desc: 'Data quality' },
          ].map((link) => {
            const Icon = link.icon
            return (
              <Link
                key={link.to}
                to={link.to}
                className="card card-hover p-4 group flex items-center gap-3"
                style={{ textDecoration: 'none' }}
              >
                <div
                  className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ background: 'var(--accent-dim)' }}
                >
                  <Icon className="h-4 w-4" style={{ color: 'var(--accent)' }} />
                </div>
                <div className="min-w-0">
                  <p className="text-[13px] font-medium" style={{ color: 'var(--text)' }}>{link.label}</p>
                  <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>{link.desc}</p>
                </div>
              </Link>
            )
          })}
        </div>
      </div>
    </div>
  )
}
