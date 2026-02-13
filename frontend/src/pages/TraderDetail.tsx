import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft,
  TrendingUp,
  Award,
  Target,
  Users,
  Clock,
  TrendingDown,
  Loader2,
  AlertCircle,
  UserPlus,
  UserCheck,
} from 'lucide-react'
import apiClient from '../api/client'

interface TraderDetail {
  user_id: string
  display_name: string
  bio: string | null
  total_pnl: number
  roi_pct: number
  win_rate: number
  total_trades: number
  winning_trades: number
  avg_trade_duration_hrs: number
  risk_score: number
  max_drawdown: number
  follower_count: number
  is_following: boolean
  created_at: string
}

export default function TraderDetail() {
  const { traderId } = useParams<{ traderId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  // Fetch trader details
  const { data: trader, isLoading, error } = useQuery<TraderDetail>({
    queryKey: ['trader', traderId],
    queryFn: async () => {
      const response = await apiClient.get(`/copy-trading/traders/${traderId}`)
      return response.data
    },
    enabled: !!traderId,
    refetchInterval: 15_000,
  })

  // Follow mutation
  const followMutation = useMutation({
    mutationFn: async () => {
      await apiClient.post('/copy-trading/follow', {
        trader_id: traderId,
        allocation_amount: 1000,
        copy_percentage: 1.0,
        auto_copy: true,
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trader', traderId] })
      queryClient.invalidateQueries({ queryKey: ['traders'] })
    },
  })

  // Unfollow mutation
  const unfollowMutation = useMutation({
    mutationFn: async () => {
      await apiClient.delete(`/copy-trading/follow/${traderId}`)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trader', traderId] })
      queryClient.invalidateQueries({ queryKey: ['traders'] })
    },
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-80">
        <Loader2 className="h-6 w-6 animate-spin" style={{ color: 'var(--text-3)' }} />
      </div>
    )
  }

  if (error || !trader) {
    return (
      <div className="flex flex-col items-center justify-center h-80 gap-3">
        <AlertCircle className="h-8 w-8" style={{ color: 'var(--red)' }} />
        <p className="text-[14px] font-medium">Trader not found</p>
        <button onClick={() => navigate('/copy-trading')} className="btn-ghost">
          Back to traders
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6 fade-up">
      {/* Back button */}
      <button
        onClick={() => navigate('/copy-trading')}
        className="flex items-center gap-2 text-[13px] transition-colors"
        style={{ color: 'var(--text-3)' }}
      >
        <ArrowLeft className="h-4 w-4" />
        Back to traders
      </button>

      {/* Trader header */}
      <div className="card p-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-4">
            <div
              className="w-16 h-16 rounded-full flex items-center justify-center text-[24px] font-bold"
              style={{ background: 'var(--accent-dim)', color: 'var(--accent)' }}
            >
              {trader.display_name.charAt(0).toUpperCase()}
            </div>
            <div>
              <h1 className="text-[24px] font-bold mb-1" style={{ color: 'var(--text)' }}>
                {trader.display_name}
              </h1>
              <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
                Trader since {new Date(trader.created_at).toLocaleDateString()}
              </p>
            </div>
          </div>

          {/* Follow button */}
          <button
            onClick={() => {
              if (trader.is_following) {
                unfollowMutation.mutate()
              } else {
                followMutation.mutate()
              }
            }}
            disabled={followMutation.isPending || unfollowMutation.isPending}
            className="px-5 py-2.5 rounded-xl text-[13px] font-semibold transition-colors flex items-center gap-2"
            style={{
              background: trader.is_following ? 'transparent' : 'var(--accent)',
              color: trader.is_following ? 'var(--text)' : '#000',
              border: trader.is_following ? '1px solid var(--border)' : 'none',
            }}
          >
            {trader.is_following ? (
              <>
                <UserCheck className="h-4 w-4" />
                Following
              </>
            ) : (
              <>
                <UserPlus className="h-4 w-4" />
                Follow Trader
              </>
            )}
          </button>
        </div>

        {/* Bio */}
        {trader.bio && (
          <p className="text-[14px] leading-relaxed" style={{ color: 'var(--text-2)' }}>
            {trader.bio}
          </p>
        )}
      </div>

      {/* Performance stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-1">
            <TrendingUp className="h-4 w-4" style={{ color: 'var(--text-3)' }} />
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Total P&L
            </p>
          </div>
          <p
            className="text-[24px] font-bold font-mono"
            style={{ color: trader.total_pnl >= 0 ? 'var(--green)' : 'var(--red)' }}
          >
            ${trader.total_pnl.toFixed(2)}
          </p>
        </div>

        <div className="card p-4">
          <div className="flex items-center gap-2 mb-1">
            <Target className="h-4 w-4" style={{ color: 'var(--text-3)' }} />
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              ROI
            </p>
          </div>
          <p
            className="text-[24px] font-bold font-mono"
            style={{ color: trader.roi_pct >= 0 ? 'var(--green)' : 'var(--red)' }}
          >
            {trader.roi_pct.toFixed(1)}%
          </p>
        </div>

        <div className="card p-4">
          <div className="flex items-center gap-2 mb-1">
            <Award className="h-4 w-4" style={{ color: 'var(--text-3)' }} />
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Win Rate
            </p>
          </div>
          <p className="text-[24px] font-bold font-mono" style={{ color: 'var(--green)' }}>
            {trader.win_rate.toFixed(1)}%
          </p>
          <p className="text-[11px] mt-1" style={{ color: 'var(--text-3)' }}>
            {trader.winning_trades}/{trader.total_trades} trades
          </p>
        </div>

        <div className="card p-4">
          <div className="flex items-center gap-2 mb-1">
            <Users className="h-4 w-4" style={{ color: 'var(--text-3)' }} />
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Followers
            </p>
          </div>
          <p className="text-[24px] font-bold font-mono" style={{ color: 'var(--text)' }}>
            {trader.follower_count}
          </p>
        </div>

        <div className="card p-4">
          <div className="flex items-center gap-2 mb-1">
            <Clock className="h-4 w-4" style={{ color: 'var(--text-3)' }} />
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Avg Duration
            </p>
          </div>
          <p className="text-[20px] font-semibold" style={{ color: 'var(--text)' }}>
            {trader.avg_trade_duration_hrs < 24
              ? `${trader.avg_trade_duration_hrs.toFixed(1)}h`
              : `${(trader.avg_trade_duration_hrs / 24).toFixed(1)}d`}
          </p>
        </div>

        <div className="card p-4">
          <div className="flex items-center gap-2 mb-1">
            <TrendingDown className="h-4 w-4" style={{ color: 'var(--text-3)' }} />
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Max Drawdown
            </p>
          </div>
          <p className="text-[20px] font-semibold" style={{ color: 'var(--red)' }}>
            {trader.max_drawdown.toFixed(1)}%
          </p>
        </div>

        <div className="card p-4">
          <div className="flex items-center gap-2 mb-1">
            <Target className="h-4 w-4" style={{ color: 'var(--text-3)' }} />
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
              Risk Score
            </p>
          </div>
          <span
            className="inline-block px-3 py-1 rounded-lg text-[16px] font-semibold"
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

        <div className="card p-4">
          <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
            Total Trades
          </p>
          <p className="text-[20px] font-semibold" style={{ color: 'var(--text)' }}>
            {trader.total_trades}
          </p>
        </div>
      </div>

      {/* Copy settings card */}
      {!trader.is_following && (
        <div
          className="card p-6"
          style={{ background: 'linear-gradient(135deg, rgba(196,162,77,0.05), rgba(196,162,77,0.02))' }}
        >
          <h2 className="text-[16px] font-semibold mb-2" style={{ color: 'var(--text)' }}>
            Start Copying {trader.display_name}
          </h2>
          <p className="text-[13px] mb-4" style={{ color: 'var(--text-3)' }}>
            Automatically replicate this trader's positions with customizable settings
          </p>
          <div className="flex gap-3">
            <button
              onClick={() => followMutation.mutate()}
              disabled={followMutation.isPending}
              className="px-5 py-2.5 rounded-xl text-[13px] font-semibold transition-colors flex items-center gap-2"
              style={{ background: 'var(--accent)', color: '#000' }}
            >
              {followMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <UserPlus className="h-4 w-4" />
              )}
              Follow & Auto-Copy
            </button>
          </div>
        </div>
      )}

      {/* Performance summary */}
      <div className="card p-6">
        <h2 className="text-[16px] font-semibold mb-4" style={{ color: 'var(--text)' }}>
          Performance Summary
        </h2>
        <div className="space-y-3">
          <div className="flex items-center justify-between py-2" style={{ borderBottom: '1px solid var(--border)' }}>
            <span className="text-[13px]" style={{ color: 'var(--text-2)' }}>
              Success Rate
            </span>
            <span className="text-[13px] font-semibold" style={{ color: 'var(--green)' }}>
              {trader.win_rate.toFixed(1)}% ({trader.winning_trades}/{trader.total_trades})
            </span>
          </div>
          <div className="flex items-center justify-between py-2" style={{ borderBottom: '1px solid var(--border)' }}>
            <span className="text-[13px]" style={{ color: 'var(--text-2)' }}>
              Average Trade Duration
            </span>
            <span className="text-[13px] font-semibold" style={{ color: 'var(--text)' }}>
              {trader.avg_trade_duration_hrs < 24
                ? `${trader.avg_trade_duration_hrs.toFixed(1)} hours`
                : `${(trader.avg_trade_duration_hrs / 24).toFixed(1)} days`}
            </span>
          </div>
          <div className="flex items-center justify-between py-2" style={{ borderBottom: '1px solid var(--border)' }}>
            <span className="text-[13px]" style={{ color: 'var(--text-2)' }}>
              Risk Level
            </span>
            <span
              className="text-[13px] font-semibold"
              style={{
                color:
                  trader.risk_score <= 3 ? 'var(--green)' : trader.risk_score <= 6 ? 'var(--accent)' : 'var(--red)',
              }}
            >
              {trader.risk_score <= 3 ? 'Low' : trader.risk_score <= 6 ? 'Medium' : 'High'} Risk ({trader.risk_score}
              /10)
            </span>
          </div>
          <div className="flex items-center justify-between py-2">
            <span className="text-[13px]" style={{ color: 'var(--text-2)' }}>
              Copiers
            </span>
            <span className="text-[13px] font-semibold" style={{ color: 'var(--text)' }}>
              {trader.follower_count} followers
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
