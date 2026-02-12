import { useQuery } from '@tanstack/react-query'
import {
  Loader2,
  AlertCircle,
  ArrowLeftRight,
  RefreshCw,
} from 'lucide-react'
import apiClient from '../api/client'

interface ArbitrageOpportunity {
  id: number
  strategy_type: string
  markets: Array<{
    id: number
    question: string
    price_yes: number
    price_no: number
  }>
  gross_spread: number | null
  net_profit_pct: number | null
  estimated_profit_usd: number | null
  detected_at: string | null
}

export default function ArbitrageScanner() {
  const {
    data,
    isLoading,
    error,
    dataUpdatedAt,
  } = useQuery<{ opportunities: ArbitrageOpportunity[]; count: number }>({
    queryKey: ['arbitrage-opportunities'],
    queryFn: async () => {
      const response = await apiClient.get('/arbitrage/opportunities')
      return response.data
    },
    refetchInterval: 15_000,
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
        <p className="text-[14px] font-medium">Failed to load arbitrage data</p>
        <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>Check API connection.</p>
      </div>
    )
  }

  const opps = data?.opportunities ?? []

  return (
    <div className="space-y-8 fade-up">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-[26px] font-bold" style={{ color: 'var(--text)' }}>
            Arbitrage Scanner
          </h1>
          <p className="text-[13px] mt-1" style={{ color: 'var(--text-2)' }}>
            Cross-platform arbitrage opportunities
          </p>
        </div>
        <div className="flex items-center gap-1.5" style={{ color: 'var(--text-3)' }}>
          <RefreshCw className="h-3 w-3" />
          <span className="text-[11px]">{new Date(dataUpdatedAt).toLocaleTimeString()}</span>
        </div>
      </div>

      {/* Summary */}
      <div className="card p-6">
        <div className="grid grid-cols-3 gap-6">
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wide mb-2" style={{ color: 'var(--text-3)' }}>
              Active
            </p>
            <p className="text-[24px] font-bold">{opps.length}</p>
          </div>
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wide mb-2" style={{ color: 'var(--text-3)' }}>
              Avg Net Spread
            </p>
            <p className="text-[24px] font-bold" style={{ color: 'var(--green)' }}>
              {opps.length > 0
                ? ((opps.reduce((s, o) => s + (o.net_profit_pct ?? 0), 0) / opps.length) * 100).toFixed(2)
                : '0.00'}%
            </p>
          </div>
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wide mb-2" style={{ color: 'var(--text-3)' }}>
              Total Est. Profit
            </p>
            <p className="text-[24px] font-bold">
              ${opps
                .reduce((s, o) => s + (o.estimated_profit_usd ?? 0), 0)
                .toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </div>
        </div>
      </div>

      {/* Opportunities or Empty */}
      {opps.length > 0 ? (
        <div className="space-y-2">
          {opps.map((opp) => (
            <div key={opp.id} className="card p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="pill pill-accent">
                  {opp.strategy_type.replace(/_/g, ' ')}
                </span>
                <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                  {opp.detected_at ? new Date(opp.detected_at).toLocaleTimeString() : ''}
                </span>
              </div>

              {opp.markets.map((m) => (
                <div
                  key={m.id}
                  className="flex items-center justify-between py-2 px-3 rounded-lg mb-1"
                  style={{ background: 'rgba(255,255,255,0.03)' }}
                >
                  <span className="text-[12px] truncate max-w-[300px]" style={{ color: 'var(--text)' }}>
                    {m.question}
                  </span>
                  <span className="text-[12px] font-mono ml-3" style={{ color: 'var(--text-2)' }}>
                    {(m.price_yes * 100).toFixed(1)}%
                  </span>
                </div>
              ))}

              <div className="grid grid-cols-3 gap-4 mt-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
                <div>
                  <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Gross</p>
                  <p className="text-[13px] font-mono font-medium" style={{ color: 'var(--accent)' }}>
                    {((opp.gross_spread ?? 0) * 100).toFixed(2)}%
                  </p>
                </div>
                <div>
                  <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Net</p>
                  <p className="text-[13px] font-mono font-medium" style={{ color: 'var(--green)' }}>
                    {((opp.net_profit_pct ?? 0) * 100).toFixed(2)}%
                  </p>
                </div>
                <div>
                  <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Profit</p>
                  <p className="text-[13px] font-mono font-medium">
                    ${(opp.estimated_profit_usd ?? 0).toFixed(2)}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="card flex flex-col items-center py-20">
          <div
            className="w-14 h-14 rounded-2xl flex items-center justify-center mb-5"
            style={{ background: 'rgba(255,255,255,0.04)' }}
          >
            <ArrowLeftRight className="h-6 w-6" style={{ color: 'var(--text-3)' }} />
          </div>
          <p className="text-[14px] font-medium mb-1" style={{ color: 'var(--text-2)' }}>
            No arbitrage opportunities detected
          </p>
          <p className="text-[12px] text-center max-w-sm" style={{ color: 'var(--text-3)' }}>
            Scanner refreshes every 15 seconds. CLOB orderbook integration will enable real spread detection.
          </p>
          <div className="flex items-center gap-2 mt-5">
            <span className="h-1.5 w-1.5 rounded-full pulse-dot" style={{ background: 'var(--accent)' }} />
            <span className="text-[11px]" style={{ color: 'var(--accent)' }}>Scanning</span>
          </div>
        </div>
      )}
    </div>
  )
}
