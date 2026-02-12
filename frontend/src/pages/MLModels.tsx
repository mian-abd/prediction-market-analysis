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
  const { data, isLoading, error } = useQuery<{ markets: MispricedMarket[] }>({
    queryKey: ['top-mispriced'],
    queryFn: async () => {
      const response = await apiClient.get('/predictions/top/mispriced', { params: { limit: 30 } })
      return response.data
    },
    refetchInterval: 60_000,
  })

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

  const markets = data?.markets ?? []

  return (
    <div className="space-y-8 fade-up">
      {/* Title */}
      <div>
        <h1 className="text-[26px] font-bold" style={{ color: 'var(--text)' }}>ML Models</h1>
        <p className="text-[13px] mt-1" style={{ color: 'var(--text-2)' }}>
          Calibration model predictions and mispriced market detection
        </p>
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

      {/* Mispriced Markets */}
      <div>
        <p className="text-[11px] font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-3)' }}>
          Top Mispriced Markets
        </p>

        {markets.length > 0 ? (
          <div className="space-y-1.5">
            {markets.map((m) => {
              const over = m.direction === 'overpriced'
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
            <p className="text-[12px] mt-1" style={{ color: 'var(--text-3)' }}>Calibration model is analyzing data</p>
          </div>
        )}
      </div>
    </div>
  )
}
