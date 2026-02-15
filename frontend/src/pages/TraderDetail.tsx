import { useState } from 'react'
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
  Activity,
  Zap,
  Bell,
} from 'lucide-react'
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts'
import apiClient from '../api/client'
import FollowTraderModal, { type FollowSettings } from '../components/FollowTraderModal'

interface TraderDetailData {
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

interface ActivityItem {
  id: number
  type: string
  data: Record<string, unknown>
  market_name: string | null
  created_at: string
}

interface EquityCurvePoint {
  timestamp: string
  pnl: number
  cumulative_pnl: number
}

interface TraderPosition {
  id: number
  market_id: number
  market_name: string
  side: string
  entry_price: number
  quantity: number
  entry_time: string
  exit_price: number | null
  exit_time: string | null
  realized_pnl: number | null
  strategy: string
}

interface DrawdownPoint {
  timestamp: string
  drawdown: number
  cumulative_pnl: number
  peak: number
}

interface PnlBucket {
  range_start: number
  range_end: number
  count: number
}

export default function TraderDetail() {
  const { traderId } = useParams<{ traderId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [showFollowModal, setShowFollowModal] = useState(false)

  // Fetch trader details
  const { data: trader, isLoading, error } = useQuery<TraderDetailData>({
    queryKey: ['trader', traderId],
    queryFn: async () => {
      const response = await apiClient.get(`/copy-trading/traders/${traderId}`)
      return response.data
    },
    enabled: !!traderId,
    refetchInterval: 15_000,
  })

  // Fetch activity feed
  const { data: activities } = useQuery<ActivityItem[]>({
    queryKey: ['trader-activity', traderId],
    queryFn: async () => {
      const response = await apiClient.get(`/copy-trading/traders/${traderId}/activity`, {
        params: { limit: 20 },
      })
      return response.data
    },
    enabled: !!traderId,
    refetchInterval: 30_000,
  })

  // Fetch equity curve
  const { data: equityCurve } = useQuery<{ data: EquityCurvePoint[]; total_pnl: number }>({
    queryKey: ['trader-equity', traderId],
    queryFn: async () => {
      const response = await apiClient.get(`/copy-trading/traders/${traderId}/equity-curve`)
      return response.data
    },
    enabled: !!traderId,
  })

  // Fetch drawdown data
  const { data: drawdownData } = useQuery<{
    drawdown: DrawdownPoint[]
    pnl_distribution: PnlBucket[]
    max_drawdown: number
  }>({
    queryKey: ['trader-drawdown', traderId],
    queryFn: async () => {
      const response = await apiClient.get(`/copy-trading/traders/${traderId}/drawdown`)
      return response.data
    },
    enabled: !!traderId,
  })

  // Fetch recent positions
  const { data: positionsData } = useQuery<{ positions: TraderPosition[]; count: number }>({
    queryKey: ['trader-positions', traderId],
    queryFn: async () => {
      const response = await apiClient.get(`/copy-trading/traders/${traderId}/positions`, {
        params: { status: 'all', limit: 10 },
      })
      return response.data
    },
    enabled: !!traderId,
  })

  // Follow mutation
  const followMutation = useMutation({
    mutationFn: async (settings: FollowSettings) => {
      const response = await apiClient.post('/copy-trading/follow', settings)
      return response.data
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['trader', traderId] })
      queryClient.invalidateQueries({ queryKey: ['traders'] })
      queryClient.invalidateQueries({ queryKey: ['portfolio'] })
      setShowFollowModal(false)

      // Show success feedback (temporary alert until toast system is added)
      if (data?.mode === 'auto') {
        alert(`✓ Now auto-copying ${trader?.display_name || 'trader'}! Future trades will appear in your Manual & Copy portfolio.`)
      } else {
        alert(`✓ Now following ${trader?.display_name || 'trader'}! You'll be notified of new trades.`)
      }
    },
    onError: (error: any) => {
      alert(`❌ Failed to follow trader: ${error.response?.data?.detail || error.message}`)
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

  const formatPnl = (val: number) => {
    if (Math.abs(val) >= 1000000) return `$${(val / 1000000).toFixed(1)}M`
    if (Math.abs(val) >= 1000) return `$${(val / 1000).toFixed(1)}K`
    return `$${val.toFixed(0)}`
  }

  const equityData = (equityCurve?.data || []).map((d) => ({
    ...d,
    date: new Date(d.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
  }))

  const drawdownChartData = (drawdownData?.drawdown || []).map((d) => ({
    ...d,
    date: new Date(d.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
  }))

  const pnlDistribution = (drawdownData?.pnl_distribution || []).map((d) => ({
    ...d,
    label: `$${d.range_start.toFixed(0)}`,
    isPositive: (d.range_start + d.range_end) / 2 >= 0,
  }))

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
                setShowFollowModal(true)
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
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Total P&L</p>
          </div>
          <p
            className="text-[24px] font-bold font-mono"
            style={{ color: trader.total_pnl >= 0 ? 'var(--green)' : 'var(--red)' }}
          >
            {formatPnl(trader.total_pnl)}
          </p>
        </div>
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-1">
            <Target className="h-4 w-4" style={{ color: 'var(--text-3)' }} />
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>ROI</p>
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
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Win Rate</p>
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
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Followers</p>
          </div>
          <p className="text-[24px] font-bold font-mono" style={{ color: 'var(--text)' }}>
            {trader.follower_count}
          </p>
        </div>
      </div>

      {/* Secondary stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-1">
            <Clock className="h-3.5 w-3.5" style={{ color: 'var(--text-3)' }} />
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Avg Duration</p>
          </div>
          <p className="text-[18px] font-semibold" style={{ color: 'var(--text)' }}>
            {trader.avg_trade_duration_hrs < 24
              ? `${trader.avg_trade_duration_hrs.toFixed(1)}h`
              : `${(trader.avg_trade_duration_hrs / 24).toFixed(1)}d`}
          </p>
        </div>
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-1">
            <TrendingDown className="h-3.5 w-3.5" style={{ color: 'var(--text-3)' }} />
            <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Max Drawdown</p>
          </div>
          <p className="text-[18px] font-semibold" style={{ color: 'var(--red)' }}>
            {formatPnl(trader.max_drawdown)}
          </p>
        </div>
        <div className="card p-4">
          <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>Risk Score</p>
          <span
            className="inline-block px-3 py-1 rounded-lg text-[14px] font-semibold"
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
          <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>Total Trades</p>
          <p className="text-[18px] font-semibold" style={{ color: 'var(--text)' }}>
            {trader.total_trades.toLocaleString()}
          </p>
        </div>
      </div>

      {/* Equity Curve */}
      {equityData.length > 0 && (
        <div className="card p-6">
          <h2 className="text-[16px] font-semibold mb-4" style={{ color: 'var(--text)' }}>
            Equity Curve
          </h2>
          <div style={{ height: 240 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={equityData}>
                <defs>
                  <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--green)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="var(--green)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10, fill: 'var(--text-3)' }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: 'var(--text-3)' }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`}
                />
                <Tooltip
                  contentStyle={{
                    background: 'var(--card)',
                    border: '1px solid var(--border)',
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(value: number | undefined) => [`$${(value ?? 0).toFixed(2)}`, 'Cumulative P&L']}
                />
                <Area
                  type="monotone"
                  dataKey="cumulative_pnl"
                  stroke="var(--green)"
                  strokeWidth={2}
                  fill="url(#pnlGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Drawdown Chart + P&L Distribution */}
      {(drawdownChartData.length > 0 || pnlDistribution.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Drawdown Chart */}
          {drawdownChartData.length > 0 && (
            <div className="card p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-[16px] font-semibold" style={{ color: 'var(--text)' }}>
                  Drawdown
                </h2>
                <span className="text-[12px] font-mono" style={{ color: 'var(--red)' }}>
                  Max: {formatPnl(drawdownData?.max_drawdown ?? 0)}
                </span>
              </div>
              <div style={{ height: 200 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={drawdownChartData}>
                    <defs>
                      <linearGradient id="ddGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--red)" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="var(--red)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 10, fill: 'var(--text-3)' }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: 'var(--text-3)' }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`}
                    />
                    <Tooltip
                      contentStyle={{
                        background: 'var(--card)',
                        border: '1px solid var(--border)',
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                      formatter={(value: number | undefined) => [`$${(value ?? 0).toFixed(2)}`, 'Drawdown']}
                    />
                    <ReferenceLine y={0} stroke="var(--border)" />
                    <Area
                      type="monotone"
                      dataKey="drawdown"
                      stroke="var(--red)"
                      strokeWidth={2}
                      fill="url(#ddGradient)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* P&L Distribution */}
          {pnlDistribution.length > 0 && (
            <div className="card p-6">
              <h2 className="text-[16px] font-semibold mb-4" style={{ color: 'var(--text)' }}>
                P&L Distribution
              </h2>
              <div style={{ height: 200 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={pnlDistribution}>
                    <XAxis
                      dataKey="label"
                      tick={{ fontSize: 9, fill: 'var(--text-3)' }}
                      axisLine={false}
                      tickLine={false}
                      interval="preserveStartEnd"
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: 'var(--text-3)' }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        background: 'var(--card)',
                        border: '1px solid var(--border)',
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                      formatter={(value: number | undefined) => [value ?? 0, 'Trades']}
                      labelFormatter={(label) => `Range: ${label}`}
                    />
                    <ReferenceLine x="$0" stroke="var(--border)" />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                      {pnlDistribution.map((entry, index) => (
                        <Cell
                          key={index}
                          fill={entry.isPositive ? 'var(--green)' : 'var(--red)'}
                          opacity={0.7}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Two-column: Activity Feed + Recent Trades */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Activity Feed */}
        <div className="card p-6">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="h-4 w-4" style={{ color: 'var(--accent)' }} />
            <h2 className="text-[16px] font-semibold" style={{ color: 'var(--text)' }}>
              Activity Feed
            </h2>
          </div>
          {activities && activities.length > 0 ? (
            <div className="space-y-3 max-h-[400px] overflow-y-auto">
              {activities.map((a) => (
                <div
                  key={a.id}
                  className="flex items-start gap-3 p-3 rounded-lg"
                  style={{ background: 'rgba(255,255,255,0.02)' }}
                >
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center mt-0.5"
                    style={{
                      background:
                        a.type === 'open_position'
                          ? 'rgba(76,175,112,0.15)'
                          : a.type === 'close_position'
                          ? 'rgba(207,102,121,0.15)'
                          : 'rgba(196,162,77,0.15)',
                    }}
                  >
                    {a.type === 'open_position' ? (
                      <Zap className="h-4 w-4" style={{ color: 'var(--green)' }} />
                    ) : a.type === 'close_position' ? (
                      <TrendingDown className="h-4 w-4" style={{ color: 'var(--red)' }} />
                    ) : (
                      <Bell className="h-4 w-4" style={{ color: 'var(--accent)' }} />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-[12px] font-medium" style={{ color: 'var(--text)' }}>
                      {a.type === 'open_position'
                        ? `Opened ${(a.data.side as string)?.toUpperCase()} position`
                        : a.type === 'close_position'
                        ? 'Closed position'
                        : a.type}
                    </p>
                    {a.market_name && (
                      <p className="text-[11px] truncate" style={{ color: 'var(--text-3)' }}>
                        {a.market_name}
                      </p>
                    )}
                    {a.data.realized_pnl !== undefined && (
                      <p
                        className="text-[11px] font-mono font-semibold"
                        style={{
                          color:
                            (a.data.realized_pnl as number) >= 0 ? 'var(--green)' : 'var(--red)',
                        }}
                      >
                        P&L: ${(a.data.realized_pnl as number).toFixed(2)}
                      </p>
                    )}
                    <p className="text-[10px] mt-1" style={{ color: 'var(--text-3)' }}>
                      {new Date(a.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center py-10">
              <Activity className="h-6 w-6 mb-2" style={{ color: 'var(--text-3)' }} />
              <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
                No activity yet
              </p>
            </div>
          )}
        </div>

        {/* Recent Trades */}
        <div className="card p-6">
          <h2 className="text-[16px] font-semibold mb-4" style={{ color: 'var(--text)' }}>
            Recent Trades
          </h2>
          {positionsData && positionsData.positions.length > 0 ? (
            <div className="space-y-2 max-h-[400px] overflow-y-auto">
              {positionsData.positions.map((pos) => (
                <div
                  key={pos.id}
                  className="flex items-center justify-between p-3 rounded-lg"
                  style={{ background: 'rgba(255,255,255,0.02)' }}
                >
                  <div className="flex-1 min-w-0 mr-3">
                    <p className="text-[12px] font-medium truncate" style={{ color: 'var(--text)' }}>
                      {pos.market_name}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <span
                        className="text-[10px] font-semibold px-1.5 py-0.5 rounded"
                        style={{
                          background:
                            pos.side === 'yes'
                              ? 'rgba(76,175,112,0.15)'
                              : 'rgba(207,102,121,0.15)',
                          color: pos.side === 'yes' ? 'var(--green)' : 'var(--red)',
                        }}
                      >
                        {pos.side.toUpperCase()}
                      </span>
                      <span className="text-[10px] font-mono" style={{ color: 'var(--text-3)' }}>
                        @ {pos.entry_price.toFixed(2)}
                      </span>
                      <span className="text-[10px]" style={{ color: 'var(--text-3)' }}>
                        {pos.exit_time ? 'Closed' : 'Open'}
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    {pos.realized_pnl !== null ? (
                      <p
                        className="text-[13px] font-semibold font-mono"
                        style={{ color: pos.realized_pnl >= 0 ? 'var(--green)' : 'var(--red)' }}
                      >
                        ${pos.realized_pnl.toFixed(2)}
                      </p>
                    ) : (
                      <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                        Active
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center py-10">
              <TrendingUp className="h-6 w-6 mb-2" style={{ color: 'var(--text-3)' }} />
              <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
                No trading history
              </p>
            </div>
          )}
        </div>
      </div>

      {/* CTA for non-followers */}
      {!trader.is_following && (
        <div
          className="card p-6"
          style={{ background: 'linear-gradient(135deg, rgba(196,162,77,0.05), rgba(196,162,77,0.02))' }}
        >
          <h2 className="text-[16px] font-semibold mb-2" style={{ color: 'var(--text)' }}>
            Start Copying {trader.display_name}
          </h2>
          <p className="text-[13px] mb-4" style={{ color: 'var(--text-3)' }}>
            Choose manual mode to track trades or auto mode to replicate positions automatically
          </p>
          <button
            onClick={() => setShowFollowModal(true)}
            disabled={followMutation.isPending}
            className="px-5 py-2.5 rounded-xl text-[13px] font-semibold transition-colors flex items-center gap-2"
            style={{ background: 'var(--accent)', color: '#000' }}
          >
            {followMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <UserPlus className="h-4 w-4" />
            )}
            Configure & Follow
          </button>
        </div>
      )}

      {/* Follow Modal */}
      {showFollowModal && (
        <FollowTraderModal
          traderName={trader.display_name}
          traderId={trader.user_id}
          onClose={() => setShowFollowModal(false)}
          onConfirm={(settings) => followMutation.mutate(settings)}
        />
      )}
    </div>
  )
}
