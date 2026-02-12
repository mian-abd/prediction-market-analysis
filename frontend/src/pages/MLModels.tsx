import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  Loader2,
  AlertCircle,
  Brain,
  TrendingUp,
  TrendingDown,
  ChevronRight,
} from 'lucide-react'
import apiClient from '../api/client'

interface MispricedMarket {
  market_id: number
  question: string
  category: string | null
  price_yes: number
  calibrated_price: number
  delta_pct: number
  direction: string
  edge_estimate: number
  volume_24h: number | null
}

export default function MLModels() {
  const navigate = useNavigate()
  const [directionFilter, setDirectionFilter] = useState<'all' | 'overpriced' | 'underpriced'>('all')
  const [minEdge, setMinEdge] = useState(0)

  const { data, isLoading, error } = useQuery<{ markets: MispricedMarket[] }>({
    queryKey: ['top-mispriced'],
    queryFn: async () => {
      const response = await apiClient.get('/predictions/top/mispriced', { params: { limit: 30 } })
      return response.data
    },
    refetchInterval: 60_000,
  })

  // All hooks must be called before any early returns
  const markets = data?.markets ?? []

  const filteredMarkets = useMemo(() => {
    let result = [...markets]
    if (directionFilter !== 'all') {
      result = result.filter((m) => m.direction === directionFilter)
    }
    if (minEdge > 0) {
      result = result.filter((m) => m.edge_estimate * 100 >= minEdge)
    }
    return result
  }, [markets, directionFilter, minEdge])

  // Early returns must come after all hooks
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-80">
        <Loader2 className="h-6 w-6 animate-spin" style={{ color: 'var(--text-3)' }} />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-80 gap-3">
        <AlertCircle className="h-8 w-8" style={{ color: 'var(--red)' }} />
        <p className="text-[14px] font-medium">Failed to load ML predictions</p>
        <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>Check API connection.</p>
      </div>
    )
  }

  const overCount = markets.filter((m) => m.direction === 'overpriced').length
  const underCount = markets.filter((m) => m.direction === 'underpriced').length
  const avgEdge = markets.length > 0
    ? markets.reduce((s, m) => s + m.edge_estimate, 0) / markets.length * 100
    : 0

  return (
    <div className="space-y-6 fade-up">
      {/* Title */}
      <div>
        <h1 className="text-[26px] font-bold" style={{ color: 'var(--text)' }}>ML Models</h1>
        <p className="text-[13px] mt-1" style={{ color: 'var(--text-2)' }}>
          Calibration model predictions and mispriced market detection
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Markets Analyzed', value: markets.length.toString(), color: 'var(--text)' },
          { label: 'Avg Edge', value: `${avgEdge.toFixed(2)}%`, color: 'var(--accent)' },
          { label: 'Overpriced', value: overCount.toString(), color: 'var(--red)' },
          { label: 'Underpriced', value: underCount.toString(), color: 'var(--green)' },
        ].map((s) => (
          <div key={s.label} className="card p-4 text-center">
            <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>{s.label}</p>
            <p className="text-[22px] font-bold" style={{ color: s.color }}>{s.value}</p>
          </div>
        ))}
      </div>

      {/* Model Card */}
      <div className="card p-6">
        <div className="flex items-center gap-3 mb-5">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background: 'var(--accent-dim)' }}
          >
            <Brain className="h-[17px] w-[17px]" style={{ color: 'var(--accent)' }} />
          </div>
          <div className="flex-1">
            <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>
              Calibration Model
            </p>
            <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
              Isotonic Regression &middot; Detects overconfidence at extremes
            </p>
          </div>
          <span className="pill pill-green">Active</span>
        </div>
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: 'Type', value: 'Isotonic' },
            { label: 'Avg Bias', value: '+6pp', color: 'var(--accent)' },
            { label: 'Markets', value: String(markets.length) },
            { label: 'Cost', value: '$0.00', color: 'var(--green)' },
          ].map((s) => (
            <div key={s.label} className="text-center py-3 rounded-xl" style={{ background: 'rgba(255,255,255,0.03)' }}>
              <p className="text-[10px] uppercase mb-1" style={{ color: 'var(--text-3)' }}>{s.label}</p>
              <p className="text-[14px] font-bold" style={{ color: s.color ?? 'var(--text)' }}>{s.value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex gap-1.5">
          {(['all', 'overpriced', 'underpriced'] as const).map((dir) => (
            <button
              key={dir}
              onClick={() => setDirectionFilter(dir)}
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-colors"
              style={{
                background: directionFilter === dir ? 'var(--card)' : 'transparent',
                color: directionFilter === dir ? 'var(--text)' : 'var(--text-3)',
                border: `1px solid ${directionFilter === dir ? 'var(--border)' : 'transparent'}`,
              }}
            >
              {dir === 'all' ? 'All' : dir.charAt(0).toUpperCase() + dir.slice(1)}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>Min Edge:</span>
          <input
            type="number"
            value={minEdge}
            onChange={(e) => setMinEdge(parseFloat(e.target.value) || 0)}
            className="input"
            style={{ width: '80px', padding: '6px 10px', fontSize: '12px' }}
            min={0}
            step={0.5}
          />
          <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>%</span>
        </div>
        <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
          {filteredMarkets.length} results
        </span>
      </div>

      {/* Mispriced Markets */}
      <div>
        <p className="text-[11px] font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-3)' }}>
          Top Mispriced Markets
        </p>

        {filteredMarkets.length > 0 ? (
          <div className="space-y-1.5">
            {filteredMarkets.map((m) => {
              const over = m.direction === 'overpriced'
              const confidence = Math.abs(m.delta_pct)
              const confLabel = confidence > 10 ? 'High' : confidence > 5 ? 'Med' : 'Low'
              return (
                <div
                  key={m.market_id}
                  onClick={() => navigate(`/markets/${m.market_id}`)}
                  className="card card-hover flex items-center gap-4 px-5 py-4 cursor-pointer group"
                >
                  {/* Direction indicator */}
                  <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                    style={{
                      background: over ? 'rgba(207,102,121,0.1)' : 'rgba(76,175,112,0.1)',
                    }}
                  >
                    {over ? (
                      <TrendingDown className="h-4 w-4" style={{ color: 'var(--red)' }} />
                    ) : (
                      <TrendingUp className="h-4 w-4" style={{ color: 'var(--green)' }} />
                    )}
                  </div>

                  {/* Question */}
                  <div className="flex-1 min-w-0">
                    <p className="text-[13px] font-medium line-clamp-1" style={{ color: 'var(--text)' }}>
                      {m.question}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      {m.category && <span className="pill">{m.category}</span>}
                      <span className={`pill ${over ? 'pill-red' : 'pill-green'}`}>
                        {over ? 'Overpriced' : 'Underpriced'}
                      </span>
                    </div>
                  </div>

                  {/* Stats */}
                  <div className="flex items-center gap-5 flex-shrink-0">
                    <div className="text-right">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Market</p>
                      <p className="text-[12px] font-mono" style={{ color: 'var(--text-2)' }}>
                        {(m.price_yes * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Calibrated</p>
                      <p className="text-[12px] font-mono font-medium" style={{ color: 'var(--blue)' }}>
                        {(m.calibrated_price * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Edge</p>
                      <p className="text-[12px] font-mono font-medium" style={{ color: 'var(--accent)' }}>
                        {(m.edge_estimate * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Confidence</p>
                      <div className="flex items-center gap-1.5">
                        <div className="w-10 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.min(100, confidence * 10)}%`,
                              background: 'var(--accent)',
                            }}
                          />
                        </div>
                        <span className="text-[10px] font-mono" style={{ color: 'var(--text-3)' }}>
                          {confLabel}
                        </span>
                      </div>
                    </div>
                  </div>

                  <ChevronRight
                    className="h-4 w-4 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ color: 'var(--text-3)' }}
                  />
                </div>
              )
            })}
          </div>
        ) : (
          <div className="card flex flex-col items-center py-16">
            <Brain className="h-6 w-6 mb-3" style={{ color: 'var(--text-3)' }} />
            <p className="text-[13px]" style={{ color: 'var(--text-2)' }}>No mispriced markets detected</p>
            <p className="text-[12px] mt-1" style={{ color: 'var(--text-3)' }}>Try adjusting the filters above</p>
          </div>
        )}
      </div>
    </div>
  )
}
