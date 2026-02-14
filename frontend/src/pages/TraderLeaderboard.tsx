import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  Users,
  TrendingUp,
  Target,
  Award,
  UserPlus,
  UserCheck,
} from 'lucide-react'
import apiClient from '../api/client'
import ErrorState from '../components/ErrorState'
import { TraderGridSkeleton, Skeleton } from '../components/LoadingSkeleton'
import FollowTraderModal, { type FollowSettings } from '../components/FollowTraderModal'

interface TraderSummary {
  user_id: string
  display_name: string
  bio: string | null
  total_pnl: number
  roi_pct: number
  win_rate: number
  total_trades: number
  risk_score: number
  follower_count: number
  is_following: boolean
}

type SortField = 'total_pnl' | 'roi_pct' | 'win_rate' | 'total_trades'

export default function TraderLeaderboard() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [sortBy, setSortBy] = useState<SortField>('total_pnl')
  const [followModalTrader, setFollowModalTrader] = useState<TraderSummary | null>(null)

  // Fetch trader leaderboard
  const { data: traders, isLoading, error, refetch } = useQuery<TraderSummary[]>({
    queryKey: ['traders', sortBy],
    queryFn: async () => {
      const response = await apiClient.get('/copy-trading/leaderboard', {
        params: { sort_by: sortBy, limit: 50 },
      })
      return response.data
    },
    refetchInterval: 30_000,
  })

  // Follow trader mutation
  const followMutation = useMutation({
    mutationFn: async (settings: FollowSettings) => {
      await apiClient.post('/copy-trading/follow', settings)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['traders'] })
      queryClient.invalidateQueries({ queryKey: ['following'] })
      setFollowModalTrader(null)
    },
  })

  // Unfollow trader mutation
  const unfollowMutation = useMutation({
    mutationFn: async (traderId: string) => {
      await apiClient.delete(`/copy-trading/follow/${traderId}`)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['traders'] })
      queryClient.invalidateQueries({ queryKey: ['following'] })
    },
  })

  if (error) {
    return (
      <ErrorState
        title="Failed to load traders"
        message="Could not fetch trader data from the API."
        onRetry={() => refetch()}
      />
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
            <Users className="h-5 w-5" style={{ color: 'var(--accent)' }} />
          </div>
          <div>
            <h1 className="text-[22px] font-bold" style={{ color: 'var(--text)' }}>
              Copy Trading
            </h1>
            <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
              Follow top traders and automatically copy their positions
            </p>
          </div>
        </div>
      </div>

      {/* Summary stats */}
      {isLoading ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="card p-4 space-y-2">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-7 w-24" />
            </div>
          ))}
        </div>
      ) : traders && traders.length > 0 ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="card p-4">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
              Top Traders
            </p>
            <p className="text-[20px] font-semibold" style={{ color: 'var(--text)' }}>
              {traders.length}
            </p>
          </div>
          <div className="card p-4">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
              Avg Win Rate
            </p>
            <p className="text-[20px] font-semibold" style={{ color: 'var(--green)' }}>
              {(traders.reduce((sum, t) => sum + t.win_rate, 0) / traders.length).toFixed(1)}%
            </p>
          </div>
          <div className="card p-4">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
              Following
            </p>
            <p className="text-[20px] font-semibold" style={{ color: 'var(--text)' }}>
              {traders.filter((t) => t.is_following).length}
            </p>
          </div>
          <div className="card p-4">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
              Total Followers
            </p>
            <p className="text-[20px] font-semibold" style={{ color: 'var(--text)' }}>
              {traders.reduce((sum, t) => sum + t.follower_count, 0)}
            </p>
          </div>
        </div>
      ) : null}

      {/* Sort controls */}
      <div className="flex items-center justify-between gap-3">
        <p className="text-[12px] font-medium" style={{ color: 'var(--text-2)' }}>
          Sorted by performance
        </p>
        <div className="flex gap-2">
          {([
            ['total_pnl', 'Total P&L'],
            ['roi_pct', 'ROI %'],
            ['win_rate', 'Win Rate'],
            ['total_trades', 'Trades'],
          ] as [SortField, string][]).map(([field, label]) => (
            <button
              key={field}
              onClick={() => setSortBy(field)}
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-colors"
              style={{
                background: sortBy === field ? 'var(--card)' : 'transparent',
                color: sortBy === field ? 'var(--text)' : 'var(--text-3)',
                border: `1px solid ${sortBy === field ? 'var(--border)' : 'transparent'}`,
              }}
            >
              {label}
              {sortBy === field && <span style={{ color: 'var(--accent)' }}> ↓</span>}
            </button>
          ))}
        </div>
      </div>

      {/* Trader grid */}
      {isLoading ? (
        <TraderGridSkeleton count={9} />
      ) : traders && traders.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {traders.map((trader) => (
            <div
              key={trader.user_id}
              className="card card-hover p-5 group cursor-pointer"
              onClick={() => navigate(`/copy-trading/${trader.user_id}`)}
            >
              {/* Header with avatar + name */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div
                    className="w-12 h-12 rounded-full flex items-center justify-center text-[16px] font-bold"
                    style={{
                      background: 'var(--accent-dim)',
                      color: 'var(--accent)',
                    }}
                  >
                    {trader.display_name.charAt(0).toUpperCase()}
                  </div>
                  <div>
                    <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>
                      {trader.display_name}
                    </p>
                    <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                      {trader.total_trades} trades
                    </p>
                  </div>
                </div>

                {/* Follow button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    if (trader.is_following) {
                      unfollowMutation.mutate(trader.user_id)
                    } else {
                      setFollowModalTrader(trader)
                    }
                  }}
                  disabled={followMutation.isPending || unfollowMutation.isPending}
                  className="px-3 py-1.5 rounded-lg text-[11px] font-semibold transition-colors flex items-center gap-1.5"
                  style={{
                    background: trader.is_following ? 'transparent' : 'var(--accent)',
                    color: trader.is_following ? 'var(--text-3)' : '#000',
                    border: trader.is_following ? '1px solid var(--border)' : 'none',
                  }}
                >
                  {trader.is_following ? (
                    <>
                      <UserCheck className="h-3.5 w-3.5" />
                      Following
                    </>
                  ) : (
                    <>
                      <UserPlus className="h-3.5 w-3.5" />
                      Follow
                    </>
                  )}
                </button>
              </div>

              {/* Bio */}
              {trader.bio && (
                <p
                  className="text-[12px] mb-4 line-clamp-2"
                  style={{ color: 'var(--text-3)', minHeight: '2.4em' }}
                >
                  {trader.bio}
                </p>
              )}

              {/* Stats grid */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div>
                  <div className="flex items-center gap-1 mb-0.5">
                    <TrendingUp className="h-3 w-3" style={{ color: 'var(--text-3)' }} />
                    <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
                      Total P&L
                    </p>
                  </div>
                  <p
                    className="text-[16px] font-bold font-mono"
                    style={{ color: trader.total_pnl >= 0 ? 'var(--green)' : 'var(--red)' }}
                  >
                    ${trader.total_pnl.toFixed(0)}
                  </p>
                </div>

                <div>
                  <div className="flex items-center gap-1 mb-0.5">
                    <Award className="h-3 w-3" style={{ color: 'var(--text-3)' }} />
                    <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
                      Win Rate
                    </p>
                  </div>
                  <p className="text-[16px] font-bold font-mono" style={{ color: 'var(--green)' }}>
                    {trader.win_rate.toFixed(1)}%
                  </p>
                </div>

                <div>
                  <div className="flex items-center gap-1 mb-0.5">
                    <Target className="h-3 w-3" style={{ color: 'var(--text-3)' }} />
                    <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
                      ROI
                    </p>
                  </div>
                  <p
                    className="text-[16px] font-bold font-mono"
                    style={{ color: trader.roi_pct >= 0 ? 'var(--green)' : 'var(--red)' }}
                  >
                    {trader.roi_pct.toFixed(1)}%
                  </p>
                </div>

                <div>
                  <div className="flex items-center gap-1 mb-0.5">
                    <Users className="h-3 w-3" style={{ color: 'var(--text-3)' }} />
                    <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
                      Followers
                    </p>
                  </div>
                  <p className="text-[16px] font-bold font-mono" style={{ color: 'var(--text)' }}>
                    {trader.follower_count}
                  </p>
                </div>
              </div>

              {/* Risk badge */}
              <div className="flex items-center justify-between pt-3" style={{ borderTop: '1px solid var(--border)' }}>
                <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                  Risk Score
                </p>
                <span
                  className="px-2 py-1 rounded text-[10px] font-semibold"
                  style={{
                    background:
                      trader.risk_score <= 3
                        ? 'rgba(76,175,112,0.15)'
                        : trader.risk_score <= 6
                        ? 'rgba(196,162,77,0.15)'
                        : 'rgba(207,102,121,0.15)',
                    color:
                      trader.risk_score <= 3
                        ? 'var(--green)'
                        : trader.risk_score <= 6
                        ? 'var(--accent)'
                        : 'var(--red)',
                  }}
                >
                  {trader.risk_score <= 3 ? 'Low' : trader.risk_score <= 6 ? 'Medium' : 'High'} ({trader.risk_score}/10)
                </span>
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {/* Empty state */}
      {!isLoading && traders && traders.length === 0 && (
        <div className="card flex flex-col items-center py-20">
          <div
            className="w-14 h-14 rounded-2xl flex items-center justify-center mb-5"
            style={{ background: 'rgba(255,255,255,0.04)' }}
          >
            <Users className="h-6 w-6" style={{ color: 'var(--text-3)' }} />
          </div>
          <p className="text-[14px] font-medium" style={{ color: 'var(--text-2)' }}>
            No traders found
          </p>
          <p className="text-[12px] max-w-sm" style={{ color: 'var(--text-3)' }}>
            Check back soon for top performing traders
          </p>
        </div>
      )}

      {/* Follow Modal */}
      {followModalTrader && (
        <FollowTraderModal
          traderName={followModalTrader.display_name}
          traderId={followModalTrader.user_id}
          onClose={() => setFollowModalTrader(null)}
          onConfirm={(settings) => followMutation.mutate(settings)}
        />
      )}
    </div>
  )
}
