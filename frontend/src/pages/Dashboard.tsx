import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  Store,
  ArrowLeftRight,
  Brain,
  Target,
  TrendingUp,
  TrendingDown,
  Loader2,
  AlertCircle,
} from 'lucide-react'
import apiClient from '../api/client'

interface DashboardStats {
  total_active_markets: number
  active_arbitrage_opportunities: number
  markets_by_platform: Record<string, number>
  price_snapshots: number
  cross_platform_matches: number
  last_data_fetch: string | null
}

function StatCard({
  label,
  value,
  icon: Icon,
  trend,
  href,
  color,
}: {
  label: string
  value: string | number
  icon: React.ComponentType<{ className?: string }>
  trend?: { value: number; up: boolean }
  href?: string
  color: string
}) {
  const content = (
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-5 hover:border-gray-600 transition-colors">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-400 mb-1">{label}</p>
          <p className="text-2xl font-bold text-white">{value}</p>
          {trend && (
            <div className="flex items-center gap-1 mt-2">
              {trend.up ? (
                <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
              ) : (
                <TrendingDown className="h-3.5 w-3.5 text-red-400" />
              )}
              <span
                className={`text-xs font-medium ${
                  trend.up ? 'text-emerald-400' : 'text-red-400'
                }`}
              >
                {trend.value}%
              </span>
            </div>
          )}
        </div>
        <div className={`p-3 rounded-lg ${color}`}>
          <Icon className="h-5 w-5 text-white" />
        </div>
      </div>
    </div>
  )

  if (href) {
    return <Link to={href}>{content}</Link>
  }
  return content
}

export default function Dashboard() {
  const { data, isLoading, error } = useQuery<DashboardStats>({
    queryKey: ['dashboard-stats'],
    queryFn: async () => {
      const response = await apiClient.get('/system/stats')
      return response.data
    },
    refetchInterval: 30_000,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-96 text-gray-400">
        <AlertCircle className="h-12 w-12 mb-4 text-red-400" />
        <p className="text-lg font-medium text-white mb-2">
          Failed to load dashboard
        </p>
        <p className="text-sm">
          Make sure the backend API is running at http://localhost:8000
        </p>
      </div>
    )
  }

  const stats = data!

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-sm text-gray-400 mt-1">
          Real-time prediction market overview
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Active Markets"
          value={stats.total_active_markets.toLocaleString()}
          icon={Store}
          color="bg-blue-600"
          href="/markets"
        />
        <StatCard
          label="Arbitrage Opportunities"
          value={stats.active_arbitrage_opportunities}
          icon={ArrowLeftRight}
          color="bg-emerald-600"
          href="/arbitrage"
        />
        <StatCard
          label="Price Snapshots"
          value={stats.price_snapshots.toLocaleString()}
          icon={Brain}
          color="bg-purple-600"
          href="/calibration"
        />
        <StatCard
          label="Cross-Platform Matches"
          value={stats.cross_platform_matches}
          icon={Target}
          color="bg-amber-600"
        />
      </div>

      {/* Platform Breakdown */}
      <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-4">
          Markets by Platform
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {stats.markets_by_platform &&
            Object.entries(stats.markets_by_platform).map(
              ([platform, count]) => (
                <div
                  key={platform}
                  className="bg-gray-900 rounded-lg p-4 text-center border border-gray-700"
                >
                  <p className="text-xl font-bold text-white">{count}</p>
                  <p className="text-xs text-gray-400 mt-1 capitalize">
                    {platform}
                  </p>
                </div>
              ),
            )}
        </div>
      </div>

      {/* Quick Links */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Link
          to="/markets"
          className="bg-gray-800 border border-gray-700 rounded-xl p-5 hover:border-blue-500/50 transition-colors group"
        >
          <Store className="h-8 w-8 text-blue-400 mb-3 group-hover:text-blue-300" />
          <h3 className="font-semibold text-white">Browse Markets</h3>
          <p className="text-sm text-gray-400 mt-1">
            Explore prediction markets across multiple platforms
          </p>
        </Link>
        <Link
          to="/arbitrage"
          className="bg-gray-800 border border-gray-700 rounded-xl p-5 hover:border-emerald-500/50 transition-colors group"
        >
          <ArrowLeftRight className="h-8 w-8 text-emerald-400 mb-3 group-hover:text-emerald-300" />
          <h3 className="font-semibold text-white">Arbitrage Scanner</h3>
          <p className="text-sm text-gray-400 mt-1">
            Find cross-platform arbitrage opportunities in real time
          </p>
        </Link>
        <Link
          to="/calibration"
          className="bg-gray-800 border border-gray-700 rounded-xl p-5 hover:border-purple-500/50 transition-colors group"
        >
          <Target className="h-8 w-8 text-purple-400 mb-3 group-hover:text-purple-300" />
          <h3 className="font-semibold text-white">Calibration Curve</h3>
          <p className="text-sm text-gray-400 mt-1">
            Analyze market price accuracy and calibration bias
          </p>
        </Link>
      </div>
    </div>
  )
}
