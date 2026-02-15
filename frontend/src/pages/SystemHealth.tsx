import { useQuery } from '@tanstack/react-query'
import {
  Shield,
  Database,
  Cpu,
  Clock,
  AlertTriangle,
  CheckCircle2,
  Activity,
} from 'lucide-react'
import apiClient from '../api/client'
import ErrorState from '../components/ErrorState'
import { Skeleton } from '../components/LoadingSkeleton'

interface SystemStats {
  total_active_markets: number
  active_arbitrage_opportunities: number
  markets_by_platform: Record<string, number>
  price_snapshots: number
  cross_platform_matches: number
  last_data_fetch: string | null
}

interface ModelAccuracy {
  trained: boolean
  metrics?: {
    trained_at: string
    n_usable: number
    n_total_resolved: number
    feature_names: string[]
    features_dropped: string[]
    ensemble_brier: number
    baseline_brier: number
    ensemble_auc: number
    ensemble_logloss: number
    models_included: string[]
  }
  error?: string
}

function ProgressBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = Math.min(100, (value / max) * 100)
  return (
    <div className="progress-bar">
      <div className="progress-fill" style={{ width: `${pct}%`, background: color }} />
    </div>
  )
}

function FreshnessDisplay({ lastFetch }: { lastFetch: string | null }) {
  if (!lastFetch) return <span className="freshness freshness-stale"><span className="freshness-dot" /> Unknown</span>
  const ageMs = Date.now() - new Date(lastFetch).getTime()
  const ageMins = Math.floor(ageMs / 60000)
  if (ageMins < 5) return <span className="freshness freshness-fresh"><span className="freshness-dot" /> {ageMins}m ago</span>
  if (ageMins < 30) return <span className="freshness freshness-ok"><span className="freshness-dot" /> {ageMins}m ago</span>
  return <span className="freshness freshness-stale"><span className="freshness-dot" /> {ageMins}m ago</span>
}

export default function SystemHealth() {
  const { data: stats, isLoading: statsLoading, error: statsError, refetch } = useQuery<SystemStats>({
    queryKey: ['system-stats'],
    queryFn: async () => (await apiClient.get('/system/stats')).data,
    refetchInterval: 30_000,
  })

  const { data: modelAccuracy } = useQuery<ModelAccuracy>({
    queryKey: ['model-accuracy'],
    queryFn: async () => (await apiClient.get('/predictions/accuracy')).data,
    refetchInterval: 60_000,
    retry: 1,
  })

  if (statsError) {
    return <ErrorState title="System unavailable" message="Cannot reach the API server." onRetry={() => refetch()} />
  }

  const s = stats
  const totalMarkets = s?.total_active_markets ?? 0
  const platforms = s?.markets_by_platform ?? {}
  const totalPlatformMarkets = Object.values(platforms).reduce((a, b) => a + b, 0)

  return (
    <div className="space-y-6 fade-up">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-[24px] font-bold" style={{ color: 'var(--text)' }}>
            System Health
          </h1>
          <p className="text-[13px] mt-1" style={{ color: 'var(--text-2)' }}>
            Data coverage, freshness, and platform reliability
          </p>
        </div>
        <div className="flex items-center gap-2">
          {statsLoading ? (
            <Skeleton className="h-6 w-28" />
          ) : (
            <>
              <CheckCircle2 className="h-4 w-4" style={{ color: 'var(--green)' }} />
              <span className="text-[13px] font-medium" style={{ color: 'var(--green)' }}>
                Operational
              </span>
            </>
          )}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          { icon: Database, label: 'Active Markets', value: totalMarkets, format: (v: number) => v.toLocaleString() },
          { icon: Activity, label: 'Price Snapshots', value: s?.price_snapshots ?? 0, format: (v: number) => v >= 1e6 ? `${(v/1e6).toFixed(1)}M` : v.toLocaleString() },
          { icon: Cpu, label: 'Cross-Platform', value: s?.cross_platform_matches ?? 0, format: (v: number) => v.toLocaleString() },
          { icon: Clock, label: 'Last Fetch', value: 0, format: () => '' },
        ].map((item, i) => {
          const Icon = item.icon
          return (
            <div key={item.label} className="card p-4">
              <div className="flex items-center gap-2 mb-2">
                <Icon className="h-4 w-4" style={{ color: 'var(--text-3)' }} />
                <span className="text-[11px] font-medium uppercase tracking-wide" style={{ color: 'var(--text-3)' }}>
                  {item.label}
                </span>
              </div>
              {statsLoading ? (
                <Skeleton className="h-7 w-20" />
              ) : i === 3 ? (
                <FreshnessDisplay lastFetch={s?.last_data_fetch ?? null} />
              ) : (
                <p className="text-[20px] font-bold" style={{ color: 'var(--text)' }}>
                  {item.format(item.value)}
                </p>
              )}
            </div>
          )
        })}
      </div>

      {/* Platform Coverage */}
      <div className="card p-5">
        <h2 className="text-[14px] font-semibold mb-4" style={{ color: 'var(--text)' }}>
          Platform Coverage
        </h2>
        {statsLoading ? (
          <div className="space-y-4">
            {[1,2].map(i => <Skeleton key={i} className="h-8 w-full" />)}
          </div>
        ) : (
          <div className="space-y-4">
            {Object.entries(platforms).map(([platform, count]) => {
              const pct = totalPlatformMarkets > 0 ? (count / totalPlatformMarkets * 100) : 0
              return (
                <div key={platform}>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[13px] font-medium capitalize" style={{ color: 'var(--text)' }}>
                      {platform}
                    </span>
                    <span className="text-[12px]" style={{ color: 'var(--text-3)' }}>
                      {count.toLocaleString()} markets ({pct.toFixed(0)}%)
                    </span>
                  </div>
                  <ProgressBar value={count} max={totalPlatformMarkets} color="var(--accent)" />
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Model Status */}
      <div className="card p-5">
        <h2 className="text-[14px] font-semibold mb-4" style={{ color: 'var(--text)' }}>
          Model Status
        </h2>
        <div className="space-y-3">
          {/* ML Ensemble */}
          <div className="p-3 rounded-xl" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-medium" style={{ color: 'var(--text)' }}>ML Ensemble</span>
                {modelAccuracy?.trained ? (
                  <span className="badge badge-high">Active</span>
                ) : (
                  <span className="badge badge-experimental">Not Trained</span>
                )}
              </div>
              {modelAccuracy?.metrics?.feature_names && (
                <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                  {modelAccuracy.metrics.feature_names.length} features active
                </span>
              )}
            </div>
            {modelAccuracy?.metrics ? (
              <div className="grid grid-cols-3 gap-4 mt-2">
                <div>
                  <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Brier</p>
                  <p className="text-[14px] font-bold" style={{ color: 'var(--text)' }}>
                    {modelAccuracy.metrics.ensemble_brier.toFixed(4)}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>vs Baseline</p>
                  <p className="text-[14px] font-bold" style={{ color: 'var(--green)' }}>
                    +{((1 - modelAccuracy.metrics.ensemble_brier / modelAccuracy.metrics.baseline_brier) * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>AUC</p>
                  <p className="text-[14px] font-bold" style={{ color: 'var(--text)' }}>
                    {modelAccuracy.metrics.ensemble_auc.toFixed(3)}
                  </p>
                </div>
              </div>
            ) : (
              <p className="text-[12px] mt-1" style={{ color: 'var(--text-3)' }}>
                {modelAccuracy?.error || 'Loading model metrics...'}
              </p>
            )}
          </div>

          {/* Elo System */}
          <div className="p-3 rounded-xl" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-medium" style={{ color: 'var(--text)' }}>Elo System (Tennis)</span>
                <span className="badge badge-high">Active</span>
              </div>
              <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>Glicko-2</span>
            </div>
            <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
              Surface-specific ratings for ATP players. 63.9% backtest accuracy on 2,973 held-out matches.
            </p>
          </div>
        </div>
      </div>

      {/* Known Limitations */}
      <div className="card p-5">
        <h2 className="text-[14px] font-semibold mb-4" style={{ color: 'var(--text)' }}>
          Known Limitations
        </h2>
        <div className="space-y-2">
          {[
            { severity: 'medium', icon: AlertTriangle, text: `Training data: ${modelAccuracy?.metrics?.n_usable?.toLocaleString() ?? '...'} usable resolved markets from ${modelAccuracy?.metrics?.n_total_resolved?.toLocaleString() ?? '...'} total` },
            { severity: 'medium', icon: AlertTriangle, text: 'Orderbook data only available for Polymarket' },
            { severity: 'medium', icon: AlertTriangle, text: `${modelAccuracy?.metrics?.features_dropped?.length ?? 0} features dropped due to zero variance or low diversity` },
            { severity: 'low', icon: Shield, text: 'Arbitrage detection may lag market by 30s-2m' },
          ].map((item, i) => {
            const Icon = item.icon
            const color = item.severity === 'high' ? 'var(--red)' : item.severity === 'medium' ? 'var(--accent)' : 'var(--green)'
            return (
              <div key={i} className="flex items-start gap-3 py-2">
                <Icon className="h-4 w-4 flex-shrink-0 mt-0.5" style={{ color }} />
                <p className="text-[13px]" style={{ color: 'var(--text-2)' }}>{item.text}</p>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
