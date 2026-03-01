/**
 * Portfolio Page
 *
 * Paper trading dashboard with:
 * - Portfolio type tabs (All / Manual & Copy / Auto Trading)
 * - Equity curve showing cumulative P&L
 * - Open positions table with portfolio badges
 * - Closed positions history
 * - Performance metrics
 * - Auto-trading control panel (when Auto tab selected)
 */

import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query'
import { Briefcase, AlertCircle, Bot, X, Plus, Loader2, UserX, RotateCcw } from 'lucide-react'
import apiClient from '../api/client'
import ErrorState from '../components/ErrorState'
import { TableSkeleton, Skeleton } from '../components/LoadingSkeleton'
import EquityCurve from '../components/charts/EquityCurve'
import DrawdownChart from '../components/charts/DrawdownChart'
import WinRateChart from '../components/charts/WinRateChart'
import PositionHeatmap from '../components/charts/PositionHeatmap'
import PerformanceAttribution from '../components/charts/PerformanceAttribution'
import AutoTradingPanel from '../components/AutoTradingPanel'

interface PortfolioSummary {
  open_positions: number
  closed_positions: number
  total_realized_pnl: number
  total_unrealized_pnl: number
  total_pnl: number
  win_rate: number
  total_exposure: number
  sharpe_ratio: number | null
  by_strategy: Array<{
    strategy: string
    trades: number
    total_pnl: number
  }>
}

interface Position {
  id: number
  market_id: number
  question: string
  platform: string
  side: string
  entry_price: number
  quantity: number
  entry_time: string
  exit_time: string | null
  exit_price: number | null
  realized_pnl: number | null
  unrealized_pnl: number | null
  current_price: number | null
  strategy: string
  portfolio_type: string
}

interface FollowingTrader {
  trader_id: string
  display_name: string
  allocation_amount: number
  copy_percentage: number
  auto_copy: boolean
  followed_at: string
  copied_trades: number
  copy_pnl: number
}

type PortfolioType = 'all' | 'manual' | 'auto'

const PNL_BASELINE_KEY = 'portfolio_pnl_baseline'

type PnlBaseline = { total: number; realized: number; unrealized: number }

function getStoredPnlBaseline(): Partial<Record<'all' | 'manual', PnlBaseline>> {
  try {
    const raw = localStorage.getItem(PNL_BASELINE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as Partial<Record<'all' | 'manual', PnlBaseline>>
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

function setStoredPnlBaseline(scope: 'all' | 'manual', value: PnlBaseline | null) {
  const stored = getStoredPnlBaseline()
  if (value == null) {
    delete stored[scope]
  } else {
    stored[scope] = value
  }
  localStorage.setItem(PNL_BASELINE_KEY, JSON.stringify(stored))
}

export default function Portfolio() {
  const navigate = useNavigate()
  const [portfolioType, setPortfolioType] = useState<PortfolioType>('all')
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | 'all'>('30d')
  const [positionStatus, setPositionStatus] = useState<'open' | 'closed' | 'all'>('open')
  const [pnlView, setPnlView] = useState<'total' | 'realized' | 'open'>('total')
  const [showOpenForm, setShowOpenForm] = useState(false)
  const [closingId, setClosingId] = useState<number | null>(null)
  const [pnlBaselineVersion, setPnlBaselineVersion] = useState(0)

  // Open position form state
  const [formMarketId, setFormMarketId] = useState('')
  const [formSide, setFormSide] = useState<'yes' | 'no'>('yes')
  const [formEntryPrice, setFormEntryPrice] = useState('')
  const [formQuantity, setFormQuantity] = useState('')
  const [formStrategy, setFormStrategy] = useState('manual')
  const [marketSearch, setMarketSearch] = useState('')
  const [showMarketDropdown, setShowMarketDropdown] = useState(false)
  const [selectedMarket, setSelectedMarket] = useState<{ id: number; question: string; platform: string; price_yes: number } | null>(null)

  const queryClient = useQueryClient()

  const invalidatePortfolio = () => {
    queryClient.invalidateQueries({ queryKey: ['portfolio-summary'] })
    queryClient.invalidateQueries({ queryKey: ['portfolio-positions'] })
  }

  const pnlBaseline: PnlBaseline | null =
    portfolioType === 'auto'
      ? null
      : (getStoredPnlBaseline()[portfolioType === 'manual' ? 'manual' : 'all'] ?? null)

  const setPnlToZero = useCallback(() => {
    if (!summary) return
    const baseline: PnlBaseline = {
      total: summary.total_pnl ?? 0,
      realized: summary.total_realized_pnl ?? 0,
      unrealized: summary.total_unrealized_pnl ?? 0,
    }
    const scope = portfolioType === 'manual' ? 'manual' : 'all'
    setStoredPnlBaseline(scope, baseline)
    setPnlBaselineVersion((v) => v + 1)
  }, [summary, portfolioType])

  const clearPnlReset = useCallback(() => {
    const scope = portfolioType === 'manual' ? 'manual' : 'all'
    setStoredPnlBaseline(scope, null)
    setPnlBaselineVersion((v) => v + 1)
  }, [portfolioType])

  // Open position mutation
  const openMutation = useMutation({
    mutationFn: async () => {
      const res = await apiClient.post('/portfolio/positions', {
        market_id: parseInt(formMarketId),
        side: formSide,
        entry_price: parseFloat(formEntryPrice),
        quantity: parseFloat(formQuantity),
        strategy: formStrategy,
      })
      if (res.data.error) throw new Error(res.data.error)
      return res.data
    },
    onSuccess: () => {
      invalidatePortfolio()
      setShowOpenForm(false)
      setFormMarketId('')
      setFormEntryPrice('')
      setFormQuantity('')
      setSelectedMarket(null)
      setMarketSearch('')
    },
  })

  // Close position mutation
  const closeMutation = useMutation({
    mutationFn: async ({ positionId, exitPrice }: { positionId: number; exitPrice: number }) => {
      const res = await apiClient.post(`/portfolio/positions/${positionId}/close`, {
        exit_price: exitPrice,
      })
      if (res.data.error) throw new Error(res.data.error)
      return res.data
    },
    onSuccess: () => {
      invalidatePortfolio()
      setClosingId(null)
    },
  })

  // Copy trading: currently followed traders (for unfollow / unfollow+close)
  const { data: followingData } = useQuery<FollowingTrader[]>({
    queryKey: ['following'],
    queryFn: async () => {
      const response = await apiClient.get('/copy-trading/following')
      return response.data
    },
    staleTime: 60_000,
  })

  const unfollowMutation = useMutation({
    mutationFn: async (traderId: string) => {
      await apiClient.delete(`/copy-trading/follow/${traderId}`)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['following'] })
    },
  })

  const unfollowAndCloseMutation = useMutation({
    mutationFn: async (traderId: string) => {
      const res = await apiClient.post(`/copy-trading/unfollow-and-close/${traderId}`)
      return res.data
    },
    onSuccess: () => {
      invalidatePortfolio()
      queryClient.invalidateQueries({ queryKey: ['following'] })
    },
  })

  // Close all open manual/copy positions in one action
  const closeAllManualMutation = useMutation({
    mutationFn: async () => {
      const res = await apiClient.post('/portfolio/positions/close-all?portfolio_type=manual')
      if (res.data.error) throw new Error(res.data.error)
      return res.data
    },
    onSuccess: () => {
      invalidatePortfolio()
    },
  })

  const portfolioParam = (portfolioType && portfolioType !== 'all') ? `&portfolio_type=${portfolioType}` : ''
  const summaryParam = (portfolioType && portfolioType !== 'all') ? `?portfolio_type=${portfolioType}` : ''

  // Fetch markets for position opening selector
  const { data: marketsData } = useQuery<{ markets: Array<{ id: number; question: string; platform: string; price_yes: number }> }>({
    queryKey: ['markets-search', marketSearch],
    queryFn: async () => {
      const searchParam = marketSearch ? `&search=${encodeURIComponent(marketSearch)}` : ''
      const response = await apiClient.get(`/markets?limit=20${searchParam}`)
      return response.data
    },
    enabled: showMarketDropdown && marketSearch.length >= 2,
    staleTime: 30_000,
  })

  // Fetch portfolio summary
  const { data: summary, isLoading: summaryLoading, error: summaryError, refetch: refetchSummary } = useQuery<PortfolioSummary>({
    queryKey: ['portfolio-summary', portfolioType],
    queryFn: async () => {
      const response = await apiClient.get(`/portfolio/summary${summaryParam}`)
      return response.data
    },
    refetchInterval: 15_000,
  })

  // Fetch positions
  const { data: positionsData, isLoading: positionsLoading } = useQuery<{ positions: Position[] }>({
    queryKey: ['portfolio-positions', positionStatus, portfolioType],
    queryFn: async () => {
      const response = await apiClient.get(`/portfolio/positions?status=${positionStatus}${portfolioParam}`)
      return response.data
    },
    refetchInterval: 15_000,
  })

  const positions = positionsData?.positions || []

  if (summaryError) {
    return (
      <ErrorState
        title="Failed to load portfolio"
        message="Could not fetch portfolio data from the API."
        onRetry={() => refetchSummary()}
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
            <Briefcase className="h-5 w-5" style={{ color: 'var(--accent)' }} />
          </div>
          <div>
            <h1 className="text-[22px] font-bold" style={{ color: 'var(--text)' }}>
              Portfolio
            </h1>
            <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
              Paper trading performance and positions
            </p>
          </div>
        </div>
        <button
          onClick={() => setShowOpenForm(!showOpenForm)}
          className="btn flex items-center gap-1.5 px-4 py-2 rounded-lg text-[12px] font-medium"
        >
          {showOpenForm ? <X className="h-3.5 w-3.5" /> : <Plus className="h-3.5 w-3.5" />}
          {showOpenForm ? 'Cancel' : 'Open Position'}
        </button>
      </div>

      {/* Open Position Form */}
      {showOpenForm && (
        <div className="card p-5">
          <h3 className="text-[14px] font-semibold mb-4" style={{ color: 'var(--text)' }}>
            Open Paper Position
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
            <div className="relative md:col-span-2">
              <label className="text-[10px] uppercase block mb-1" style={{ color: 'var(--text-3)' }}>
                Select Market
              </label>
              <input
                type="text"
                value={selectedMarket ? selectedMarket.question : marketSearch}
                onChange={(e) => {
                  setMarketSearch(e.target.value)
                  setSelectedMarket(null)
                  setFormMarketId('')
                  setShowMarketDropdown(true)
                }}
                onFocus={() => setShowMarketDropdown(true)}
                placeholder="Search markets..."
                className="input w-full px-3 py-2 rounded-lg text-[12px]"
              />
              {showMarketDropdown && marketSearch.length >= 2 && marketsData && marketsData.markets.length > 0 && (
                <div
                  className="absolute z-10 w-full mt-1 rounded-lg overflow-hidden shadow-lg"
                  style={{ background: 'var(--card)', border: '1px solid var(--border)', maxHeight: '300px', overflowY: 'auto' }}
                >
                  {marketsData.markets.map((market) => (
                    <button
                      key={market.id}
                      type="button"
                      onClick={() => {
                        setSelectedMarket(market)
                        setFormMarketId(market.id.toString())
                        setFormEntryPrice(market.price_yes.toFixed(3))
                        setShowMarketDropdown(false)
                        setMarketSearch('')
                      }}
                      className="w-full px-3 py-2 text-left hover:bg-white/5 transition-colors border-b border-white/5 last:border-0"
                    >
                      <p className="text-[12px] font-medium" style={{ color: 'var(--text)' }}>
                        {market.question}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
                          {market.platform}
                        </span>
                        <span className="text-[10px]" style={{ color: 'var(--accent)' }}>
                          Current: {(market.price_yes * 100).toFixed(1)}%
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
              {selectedMarket && (
                <div className="mt-2 p-2 rounded-lg" style={{ background: 'rgba(196, 162, 77, 0.1)', border: '1px solid var(--accent)' }}>
                  <p className="text-[11px] font-medium" style={{ color: 'var(--text)' }}>
                    {selectedMarket.question}
                  </p>
                  <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-3)' }}>
                    {selectedMarket.platform} • ID: {selectedMarket.id} • Current: {(selectedMarket.price_yes * 100).toFixed(1)}%
                  </p>
                </div>
              )}
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div>
              <label className="text-[10px] uppercase block mb-1" style={{ color: 'var(--text-3)' }}>
                Side
              </label>
              <div className="flex gap-2">
                {(['yes', 'no'] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setFormSide(s)}
                    className={`flex-1 px-3 py-2 rounded-lg text-[12px] font-medium uppercase transition-colors ${
                      formSide === s ? 'btn' : 'btn-ghost'
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="text-[10px] uppercase block mb-1" style={{ color: 'var(--text-3)' }}>
                Entry Price
              </label>
              <input
                type="number"
                step="0.001"
                min="0"
                max="1"
                value={formEntryPrice}
                onChange={(e) => setFormEntryPrice(e.target.value)}
                placeholder="0.650"
                className="input w-full px-3 py-2 rounded-lg text-[12px]"
              />
            </div>
            <div>
              <label className="text-[10px] uppercase block mb-1" style={{ color: 'var(--text-3)' }}>
                Quantity
              </label>
              <input
                type="number"
                min="1"
                value={formQuantity}
                onChange={(e) => setFormQuantity(e.target.value)}
                placeholder="100"
                className="input w-full px-3 py-2 rounded-lg text-[12px]"
              />
            </div>
            <div>
              <label className="text-[10px] uppercase block mb-1" style={{ color: 'var(--text-3)' }}>
                Strategy
              </label>
              <select
                value={formStrategy}
                onChange={(e) => setFormStrategy(e.target.value)}
                className="input w-full px-3 py-2 rounded-lg text-[12px]"
              >
                <option value="manual">Manual</option>
                <option value="single_market_arb">Single Market Arb</option>
                <option value="cross_platform_arb">Cross Platform Arb</option>
                <option value="calibration">Calibration</option>
              </select>
            </div>
          </div>
          {openMutation.error && (
            <p className="text-[12px] mt-2" style={{ color: 'var(--red)' }}>
              {(openMutation.error as Error).message}
            </p>
          )}
          <div className="flex justify-end mt-4">
            <button
              onClick={() => openMutation.mutate()}
              disabled={openMutation.isPending || !formMarketId || !formEntryPrice || !formQuantity}
              className="btn px-5 py-2 rounded-lg text-[12px] font-medium disabled:opacity-40"
            >
              {openMutation.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                'Submit Order'
              )}
            </button>
          </div>
        </div>
      )}

      {/* Portfolio Type Tabs */}
      <div className="flex items-center gap-2">
        {([
          { key: 'all' as const, label: 'All' },
          { key: 'manual' as const, label: 'Manual & Copy' },
          { key: 'auto' as const, label: 'Auto Trading', icon: Bot },
        ]).map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setPortfolioType(key)}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-[12px] font-medium transition-colors ${
              portfolioType === key ? 'btn' : 'btn-ghost'
            }`}
          >
            {Icon && <Icon className="h-3.5 w-3.5" />}
            {label}
          </button>
        ))}
      </div>

      {/* Auto Trading Panel (only when Auto tab selected) */}
      {portfolioType === 'auto' && <AutoTradingPanel />}

      {/* Summary stats */}
      {summaryLoading ? (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="card p-4 space-y-2">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-7 w-24" />
            </div>
          ))}
        </div>
      ) : summary ? (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {/* P&L Card — Robinhood/Webull style: prominent number + Total / Realized / Open toggle + Reset to 0 */}
          <div className="card p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
                P&L
              </p>
              {(portfolioType === 'all' || portfolioType === 'manual') && (
                <div className="flex items-center gap-1">
                  {pnlBaseline ? (
                    <button
                      type="button"
                      onClick={clearPnlReset}
                      className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium transition-colors hover:bg-white/10"
                      style={{ color: 'var(--text-3)' }}
                      title="Show actual P&L again"
                    >
                      <RotateCcw className="h-3 w-3" />
                      Clear reset
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={setPnlToZero}
                      disabled={!summary}
                      className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium transition-colors hover:bg-white/10 disabled:opacity-50"
                      style={{ color: 'var(--text-3)' }}
                      title="Reset P&L display to $0 (track from here)"
                    >
                      <RotateCcw className="h-3 w-3" />
                      Reset to $0
                    </button>
                  )}
                </div>
              )}
            </div>
            <div className="flex gap-2 mb-2">
              {(['total', 'realized', 'open'] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setPnlView(mode)}
                  className="px-3 py-1.5 rounded-lg text-[11px] font-semibold uppercase transition-colors"
                  style={{
                    background: pnlView === mode ? 'var(--accent)' : 'rgba(255,255,255,0.06)',
                    color: pnlView === mode ? 'var(--bg)' : 'var(--text-3)',
                  }}
                >
                  {mode === 'total' ? 'Total' : mode === 'realized' ? 'Realized' : 'Open'}
                </button>
              ))}
            </div>
            {(() => {
              const totalRaw = summary.total_pnl ?? summary.total_realized_pnl ?? 0
              const realizedRaw = summary.total_realized_pnl ?? 0
              const openRaw = summary.total_unrealized_pnl ?? 0
              const total = pnlBaseline ? totalRaw - pnlBaseline.total : totalRaw
              const realized = pnlBaseline ? realizedRaw - pnlBaseline.realized : realizedRaw
              const open = pnlBaseline ? openRaw - pnlBaseline.unrealized : openRaw
              const value = pnlView === 'total' ? total : pnlView === 'realized' ? realized : open
              const color = value >= 0 ? 'var(--green)' : 'var(--red)'
              return (
                <>
                  <p className="text-[26px] font-mono font-bold tracking-tight" style={{ color }}>
                    ${value.toFixed(2)}
                  </p>
                  {pnlView === 'total' && (
                    <p className="text-[11px] font-mono mt-1" style={{ color: 'var(--text-3)' }}>
                      Realized ${realized.toFixed(2)} · Open ${open.toFixed(2)}
                    </p>
                  )}
                  {pnlBaseline && (
                    <p className="text-[10px] mt-1" style={{ color: 'var(--text-3)', opacity: 0.8 }}>
                      Display reset to $0
                    </p>
                  )}
                </>
              )
            })()}
          </div>

          <div className="card p-4">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
              Win Rate
            </p>
            <p className="text-[20px] font-mono font-bold" style={{ color: 'var(--accent)' }}>
              {(summary.win_rate ?? 0).toFixed(1)}%
            </p>
          </div>

          <div className="card p-4">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
              Sharpe Ratio
            </p>
            <p className="text-[20px] font-mono font-bold" style={{
              color: summary.sharpe_ratio != null
                ? (summary.sharpe_ratio ?? 0) >= 1 ? 'var(--green)' : (summary.sharpe_ratio ?? 0) >= 0 ? 'var(--accent)' : 'var(--red)'
                : 'var(--text-3)'
            }}>
              {summary.sharpe_ratio != null ? (summary.sharpe_ratio ?? 0).toFixed(2) : '—'}
            </p>
          </div>

          <div className="card p-4">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
              Open Positions
            </p>
            <p className="text-[20px] font-semibold" style={{ color: 'var(--text)' }}>
              {summary.open_positions}
            </p>
          </div>

          <div className="card p-4">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
              Closed Trades
            </p>
            <p className="text-[20px] font-semibold" style={{ color: 'var(--text)' }}>
              {summary.closed_positions}
            </p>
          </div>

          <div className="card p-4">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>
              Exposure
            </p>
            <p className="text-[20px] font-mono font-bold" style={{ color: 'var(--text)' }}>
              ${(summary.total_exposure ?? 0).toFixed(2)}
            </p>
          </div>
        </div>
      ) : null}

      {/* Equity Curve */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-[16px] font-semibold" style={{ color: 'var(--text)' }}>
            Equity Curve
          </h2>
          <div className="flex items-center gap-2">
            {(['7d', '30d', '90d', 'all'] as const).map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1.5 rounded-lg text-[12px] font-medium transition-colors ${
                  timeRange === range ? 'btn' : 'btn-ghost'
                }`}
              >
                {range.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <EquityCurve
          timeRange={timeRange}
          showDrawdown={false}
          autoRefresh={true}
          portfolioType={portfolioType}
          pnlBaseline={pnlBaseline?.total ?? undefined}
        />
      </div>

      {/* Drawdown Chart */}
      <div className="card p-6">
        <h2 className="text-[16px] font-semibold mb-5" style={{ color: 'var(--text)' }}>
          Drawdown Analysis
        </h2>
        <DrawdownChart timeRange={timeRange} portfolioType={portfolioType} />
      </div>

      {/* Win Rate by Strategy */}
      <div className="card p-6">
        <h2 className="text-[16px] font-semibold mb-5" style={{ color: 'var(--text)' }}>
          Win Rate by Strategy
        </h2>
        <WinRateChart minTrades={1} portfolioType={portfolioType} />
      </div>

      {/* Position Heatmap */}
      <div className="card p-6">
        <h2 className="text-[16px] font-semibold mb-5" style={{ color: 'var(--text)' }}>
          Position Heatmap
        </h2>
        <PositionHeatmap portfolioType={portfolioType} />
      </div>

      {/* Copy Trading — Following (always visible on All / Manual so users see unfollow + close-all) */}
      {(portfolioType === 'all' || portfolioType === 'manual') && (
        <div className="card p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <UserX className="h-4 w-4" style={{ color: 'var(--text-3)' }} />
              <h2 className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>
                Copy Trading — Following
              </h2>
            </div>
            <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
              Manage who you follow and quickly exit copied trades
            </p>
          </div>
          {followingData && followingData.length > 0 ? (
            <div className="flex flex-col gap-2">
            {followingData.map((t) => (
              <div
                key={t.trader_id}
                className="flex flex-col md:flex-row md:items-center justify-between gap-2 py-2 px-2 rounded-lg"
                style={{ background: 'rgba(255,255,255,0.02)' }}
              >
                <div>
                  <p className="text-[13px] font-semibold" style={{ color: 'var(--text)' }}>
                    {t.display_name}
                  </p>
                  <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                    Copied trades: {t.copied_trades} • Copy P&L:{' '}
                    <span
                      className="font-mono font-semibold"
                      style={{ color: t.copy_pnl >= 0 ? 'var(--green)' : 'var(--red)' }}
                    >
                      {t.copy_pnl >= 0 ? '+' : '-'}${Math.abs(t.copy_pnl).toFixed(2)}
                    </span>
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => unfollowMutation.mutate(t.trader_id)}
                    disabled={unfollowMutation.isPending || unfollowAndCloseMutation.isPending}
                    className="px-3 py-1.5 rounded-lg text-[11px] font-semibold"
                    style={{
                      border: '1px solid var(--border)',
                      color: 'var(--text-3)',
                      background: 'transparent',
                    }}
                  >
                    Unfollow
                  </button>
                  <button
                    onClick={() => unfollowAndCloseMutation.mutate(t.trader_id)}
                    disabled={unfollowAndCloseMutation.isPending || unfollowMutation.isPending}
                    className="px-3 py-1.5 rounded-lg text-[11px] font-semibold flex items-center gap-1.5"
                    style={{
                      background: 'rgba(207,102,121,0.15)',
                      color: 'var(--red)',
                    }}
                  >
                    <UserX className="h-3 w-3" />
                    Unfollow & Close All
                  </button>
                </div>
              </div>
            ))}
            </div>
          ) : (
            <p className="text-[12px] py-2" style={{ color: 'var(--text-3)' }}>
              Not following any traders.{' '}
              <button
                type="button"
                onClick={() => navigate('/copy-trading')}
                className="font-semibold underline"
                style={{ color: 'var(--accent)' }}
              >
                Go to Copy Trading
              </button>
              {' '}to follow.
            </p>
          )}
        </div>
      )}

      {/* Positions table */}
      <div className="card p-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
          <h2 className="text-[16px] font-semibold" style={{ color: 'var(--text)' }}>
            Positions
          </h2>
          <div className="flex items-center gap-2 flex-wrap">
            {(['open', 'closed', 'all'] as const).map((status) => (
              <button
                key={status}
                onClick={() => setPositionStatus(status)}
                className={`px-3 py-1.5 rounded-lg text-[12px] font-medium capitalize transition-colors ${
                  positionStatus === status ? 'btn' : 'btn-ghost'
                }`}
              >
                {status}
              </button>
            ))}
            {/* Close all manual/copy open positions — visible when viewing open and there are manual positions */}
            {positionStatus === 'open' &&
              (portfolioType === 'all' || portfolioType === 'manual') &&
              positions.some((p) => p.portfolio_type === 'manual') && (
                <button
                  onClick={() => closeAllManualMutation.mutate()}
                  disabled={closeAllManualMutation.isPending}
                  className="px-3 py-1.5 rounded-lg text-[12px] font-semibold flex items-center gap-1.5"
                  style={{
                    background: 'rgba(239,68,68,0.15)',
                    color: 'var(--red)',
                  }}
                >
                  {closeAllManualMutation.isPending ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : null}
                  Close all (Manual & Copy)
                </button>
              )}
          </div>
        </div>

        {positionsLoading ? (
          <TableSkeleton rows={5} columns={8} />
        ) : positions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-40 gap-2">
            <AlertCircle className="h-5 w-5" style={{ color: 'var(--text-3)' }} />
            <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
              No {positionStatus} positions
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]">
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  <th className="text-left py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    Market
                  </th>
                  <th className="text-left py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    Type
                  </th>
                  <th className="text-left py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    Side
                  </th>
                  <th className="text-right py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    Entry
                  </th>
                  <th className="text-right py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    Current/Exit
                  </th>
                  <th className="text-right py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    Qty
                  </th>
                  <th className="text-right py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    P&L
                  </th>
                  <th className="text-left py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    Strategy
                  </th>
                  <th className="text-center py-2 px-3" style={{ color: 'var(--text-3)' }}>
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos) => {
                  const pnl = pos.realized_pnl ?? pos.unrealized_pnl ?? 0
                  const isProfitable = pnl >= 0
                  const isAuto = pos.portfolio_type === 'auto'
                  const isCopy = pos.strategy === 'copy_trade'

                  return (
                    <tr
                      key={pos.id}
                      style={{ borderBottom: '1px solid var(--border)' }}
                      className="hover:bg-white/[0.02] transition-colors"
                    >
                      <td className="py-3 px-3">
                        <p className="truncate max-w-[300px]" style={{ color: 'var(--text)' }}>
                          {pos.question}
                        </p>
                        <p className="text-[10px] uppercase mt-0.5" style={{ color: 'var(--text-3)' }}>
                          {pos.platform}
                        </p>
                      </td>
                      <td className="py-3 px-3">
                        <span
                          className="px-2 py-1 rounded text-[10px] font-semibold uppercase"
                          style={{
                            background: isAuto
                              ? 'rgba(196,162,77,0.15)'
                              : isCopy
                                ? 'rgba(99,102,241,0.15)'
                                : 'rgba(255,255,255,0.06)',
                            color: isAuto
                              ? 'var(--accent)'
                              : isCopy
                                ? '#818CF8'
                                : 'var(--text-3)',
                          }}
                        >
                          {isAuto ? 'AUTO' : isCopy ? 'COPY' : 'MANUAL'}
                        </span>
                      </td>
                      <td className="py-3 px-3">
                        <span
                          className="px-2 py-1 rounded text-[10px] font-semibold uppercase"
                          style={{
                            background: pos.side === 'yes' ? 'rgba(76,175,112,0.15)' : 'rgba(207,102,121,0.15)',
                            color: pos.side === 'yes' ? 'var(--green)' : 'var(--red)',
                          }}
                        >
                          {pos.side}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right font-mono" style={{ color: 'var(--text)' }}>
                        {(pos.entry_price ?? 0).toFixed(3)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono" style={{ color: 'var(--text)' }}>
                        {(pos.exit_price ?? pos.current_price ?? 0).toFixed(3)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono" style={{ color: 'var(--text)' }}>
                        {(pos.quantity ?? 0).toFixed(0)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono font-semibold">
                        <span style={{ color: isProfitable ? 'var(--green)' : 'var(--red)' }}>
                          {isProfitable ? '+' : ''}${pnl.toFixed(2)}
                        </span>
                      </td>
                      <td className="py-3 px-3">
                        <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                          {pos.strategy.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-center">
                        {pos.exit_time === null ? (
                          closingId === pos.id ? (
                            <div className="flex items-center gap-1 justify-center">
                              <Loader2 className="h-3 w-3 animate-spin" style={{ color: 'var(--text-3)' }} />
                            </div>
                          ) : (
                            <button
                              onClick={() => {
                                setClosingId(pos.id)
                                closeMutation.mutate({
                                  positionId: pos.id,
                                  exitPrice: pos.current_price ?? pos.entry_price,
                                })
                              }}
                              className="px-2.5 py-1 rounded text-[10px] font-semibold uppercase transition-colors hover:opacity-80"
                              style={{
                                background: 'rgba(207,102,121,0.15)',
                                color: 'var(--red)',
                              }}
                            >
                              Close
                            </button>
                          )
                        ) : (
                          <span className="text-[10px]" style={{ color: 'var(--text-3)' }}>
                            Closed
                          </span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Performance Attribution */}
      <div className="card p-6">
        <h2 className="text-[16px] font-semibold mb-5" style={{ color: 'var(--text)' }}>
          Performance Attribution
        </h2>
        <PerformanceAttribution portfolioType={portfolioType} />
      </div>
    </div>
  )
}
