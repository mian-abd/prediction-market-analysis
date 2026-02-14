import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  Store,
  ArrowLeftRight,
  Brain,
  Target,
  ChevronRight,
} from 'lucide-react'
import apiClient from '../api/client'
import ErrorState from '../components/ErrorState'
import { StatsGridSkeleton } from '../components/LoadingSkeleton'

interface DashboardStats {
  total_active_markets: number
  active_arbitrage_opportunities: number
  markets_by_platform: Record<string, number>
  price_snapshots: number
  cross_platform_matches: number
  last_data_fetch: string | null
}

export default function Dashboard() {
  const { data, isLoading, error, refetch } = useQuery<DashboardStats>({
    queryKey: ['dashboard-stats'],
    queryFn: async () => {
      const response = await apiClient.get('/system/stats')
      return response.data
    },
    refetchInterval: 30_000,
  })

  if (isLoading) {
    return (
      <div className="space-y-8 fade-up">
        <div>
          <h1 className="text-[26px] font-bold" style={{ color: 'var(--text)' }}>
            Dashboard
          </h1>
          <p className="text-[13px] mt-1" style={{ color: 'var(--text-2)' }}>
            Real-time prediction market overview
          </p>
        </div>
        <StatsGridSkeleton count={4} />
      </div>
    )
  }

  if (error) {
    return (
      <ErrorState
        title="Failed to load dashboard"
        message="Could not connect to the API server."
        onRetry={() => refetch()}
      />
    )
  }

  const s = data!

  return (
    <div className="space-y-8 fade-up">
      {/* Title */}
      <div>
        <h1 className="text-[26px] font-bold" style={{ color: 'var(--text)' }}>
          Dashboard
        </h1>
        <p className="text-[13px] mt-1" style={{ color: 'var(--text-2)' }}>
          Real-time prediction market overview
        </p>
      </div>

      {/* Summary Card */}
      <div className="card p-6">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
          {[
            { label: 'Active Markets', value: s.total_active_markets },
            { label: 'Arbitrage Opps', value: s.active_arbitrage_opportunities },
            { label: 'Price Snapshots', value: s.price_snapshots },
            { label: 'Cross-Platform', value: s.cross_platform_matches },
          ].map((item) => (
            <div key={item.label}>
              <p className="text-[11px] font-medium uppercase tracking-wide mb-2" style={{ color: 'var(--text-3)' }}>
                {item.label}
              </p>
              <p className="text-[24px] font-bold" style={{ color: 'var(--text)' }}>
                {(item.value ?? 0).toLocaleString()}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Platforms */}
      {s.markets_by_platform && Object.keys(s.markets_by_platform).length > 0 && (
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-3)' }}>
            By Platform
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {Object.entries(s.markets_by_platform).map(([platform, count]) => (
              <div key={platform} className="card p-4">
                <p className="text-[20px] font-bold" style={{ color: 'var(--text)' }}>
                  {count.toLocaleString()}
                </p>
                <p className="text-[11px] capitalize mt-1" style={{ color: 'var(--text-3)' }}>
                  {platform}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quick Links */}
      <div>
        <p className="text-[11px] font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-3)' }}>
          Quick Actions
        </p>
        <div className="space-y-2">
          {[
            { to: '/markets', icon: Store, label: 'Browse Markets', desc: 'Explore prediction markets across platforms' },
            { to: '/arbitrage', icon: ArrowLeftRight, label: 'Arbitrage Scanner', desc: 'Find cross-platform arbitrage opportunities' },
            { to: '/models', icon: Brain, label: 'ML Models', desc: 'Calibration model and mispriced markets' },
            { to: '/calibration', icon: Target, label: 'Calibration Curve', desc: 'Market price accuracy and bias analysis' },
          ].map((link) => {
            const Icon = link.icon
            return (
              <Link
                key={link.to}
                to={link.to}
                className="card card-hover flex items-center gap-4 p-4 group"
                style={{ textDecoration: 'none' }}
              >
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                  style={{ background: 'var(--accent-dim)' }}
                >
                  <Icon className="h-[17px] w-[17px]" style={{ color: 'var(--accent)' }} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-[14px] font-medium" style={{ color: 'var(--text)' }}>
                    {link.label}
                  </p>
                  <p className="text-[12px] mt-0.5" style={{ color: 'var(--text-3)' }}>
                    {link.desc}
                  </p>
                </div>
                <ChevronRight
                  className="h-4 w-4 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ color: 'var(--text-3)' }}
                />
              </Link>
            )
          })}
        </div>
      </div>
    </div>
  )
}
