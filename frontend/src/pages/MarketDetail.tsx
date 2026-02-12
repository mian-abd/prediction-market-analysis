import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  ArrowLeft,
  Loader2,
  AlertCircle,
  Brain,
  Sparkles,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Clock,
  ExternalLink,
} from 'lucide-react'
import apiClient from '../api/client'
import PriceChart from '../components/charts/PriceChart'
import OrderbookDepth from '../components/charts/OrderbookDepth'
import VolumeProfile from '../components/charts/VolumeProfile'
import SentimentGauge from '../components/charts/SentimentGauge'

interface MarketData {
  id: number
  question: string
  description: string | null
  price_yes: number | null
  price_no: number | null
  volume_24h: number | null
  volume_total: number | null
  category: string | null
  platform: string
  end_date: string | null
  last_fetched_at: string | null
  updated_at: string | null
  cross_platform_matches: Array<{
    id: number
    platform: string
    question: string
    price_yes: number
    similarity: number
  }>
}

function getRelativeTime(isoString: string): string {
  const diff = Date.now() - new Date(isoString).getTime()
  const seconds = Math.floor(diff / 1000)
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  return `${hours}h ago`
}

function getFreshnessColor(isoString: string): string {
  const diff = Date.now() - new Date(isoString).getTime()
  const seconds = diff / 1000
  if (seconds < 30) return '#4CAF70'
  if (seconds < 120) return '#C4A24D'
  return '#CF6679'
}

interface Prediction {
  market_id: number
  models: {
    calibration: {
      market_price: number
      calibrated_price: number
      delta: number
      delta_pct: number
      direction: string
      edge_estimate: number
    }
  }
}

interface AnalysisResult {
  market_id: number
  question: string
  analysis: string
}

export default function MarketDetail() {
  const { id } = useParams<{ id: string }>()
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)

  const { data: market, isLoading, error } = useQuery<MarketData>({
    queryKey: ['market', id],
    queryFn: async () => {
      const response = await apiClient.get(`/markets/${id}`)
      return response.data
    },
    refetchInterval: 15_000,
  })

  const { data: prediction } = useQuery<Prediction>({
    queryKey: ['prediction', id],
    queryFn: async () => {
      const response = await apiClient.get(`/predictions/${id}`)
      return response.data
    },
    enabled: !!id,
  })

  const analyzeMutation = useMutation({
    mutationFn: async () => {
      const response = await apiClient.post(`/analyze/${id}`)
      return response.data as AnalysisResult
    },
    onSuccess: (data) => setAnalysis(data),
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-80">
        <Loader2 className="h-6 w-6 animate-spin" style={{ color: 'var(--text-3)' }} />
      </div>
    )
  }

  if (error || !market) {
    return (
      <div className="flex flex-col items-center justify-center h-80 gap-3">
        <AlertCircle className="h-8 w-8" style={{ color: 'var(--red)' }} />
        <p className="text-[14px] font-medium">Market not found</p>
        <Link to="/markets" className="btn-ghost text-[13px]">
          <ArrowLeft className="h-3.5 w-3.5" /> Back to Markets
        </Link>
      </div>
    )
  }

  const cal = prediction?.models?.calibration

  return (
    <div className="space-y-6 fade-up">
      {/* Back */}
      <Link
        to="/markets"
        className="inline-flex items-center gap-1.5 text-[12px] font-medium"
        style={{ color: 'var(--text-3)' }}
      >
        <ArrowLeft className="h-3.5 w-3.5" /> Markets
      </Link>

      {/* Header */}
      <div className="card p-6">
        <div className="flex flex-col lg:flex-row lg:items-start gap-6">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-3">
              <span className="pill">{market.category ?? 'other'}</span>
              <span className="pill pill-accent capitalize">{market.platform}</span>
            </div>
            <h1 className="text-[20px] font-bold leading-snug" style={{ color: 'var(--text)' }}>
              {market.question}
            </h1>
            {market.description && (
              <p className="text-[13px] mt-2 leading-relaxed" style={{ color: 'var(--text-2)' }}>
                {market.description}
              </p>
            )}
          </div>

          {/* Prices */}
          <div className="flex gap-3 flex-shrink-0">
            <div className="text-center px-5 py-4 rounded-2xl" style={{ background: 'rgba(76,175,112,0.08)' }}>
              <p className="text-[10px] font-semibold uppercase mb-1" style={{ color: 'var(--green)' }}>Yes</p>
              <p className="text-[26px] font-bold font-mono" style={{ color: 'var(--green)' }}>
                {((market.price_yes ?? 0) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="text-center px-5 py-4 rounded-2xl" style={{ background: 'rgba(207,102,121,0.08)' }}>
              <p className="text-[10px] font-semibold uppercase mb-1" style={{ color: 'var(--red)' }}>No</p>
              <p className="text-[26px] font-bold font-mono" style={{ color: 'var(--red)' }}>
                {((market.price_no ?? 0) * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6 pt-6" style={{ borderTop: '1px solid var(--border)' }}>
          {[
            { icon: DollarSign, label: '24h Volume', value: `$${(market.volume_24h ?? 0).toLocaleString()}` },
            { icon: DollarSign, label: 'Total Volume', value: `$${(market.volume_total ?? 0).toLocaleString()}` },
            { icon: Clock, label: 'End Date', value: market.end_date ? new Date(market.end_date).toLocaleDateString() : 'N/A' },
            { icon: ExternalLink, label: 'Matches', value: `${market.cross_platform_matches?.length ?? 0}` },
          ].map((stat, i) => {
            const Icon = stat.icon
            return (
              <div key={i} className="flex items-center gap-2.5">
                <Icon className="h-4 w-4 flex-shrink-0" style={{ color: 'var(--text-3)' }} />
                <div>
                  <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>{stat.label}</p>
                  <p className="text-[13px] font-medium" style={{ color: 'var(--text)' }}>{stat.value}</p>
                </div>
              </div>
            )
          })}
        </div>

        {/* Freshness Indicator */}
        {market.last_fetched_at && (
          <div className="flex items-center gap-2 mt-4 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ background: getFreshnessColor(market.last_fetched_at) }}
            />
            <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
              Last updated {getRelativeTime(market.last_fetched_at)}
            </span>
          </div>
        )}
      </div>

      {/* Price Chart */}
      <div className="card p-6">
        <p className="text-[14px] font-semibold mb-5" style={{ color: 'var(--text)' }}>Price History</p>
        <PriceChart
          marketId={market.id}
          interval="5m"
          type="candlestick"
          height={400}
          showVolume={true}
          showCrosshair={true}
          autoRefresh={true}
        />
      </div>

      {/* Orderbook Depth */}
      <div className="card p-6">
        <p className="text-[14px] font-semibold mb-5" style={{ color: 'var(--text)' }}>Orderbook Depth</p>
        <OrderbookDepth
          marketId={market.id}
          maxDepth={10}
          highlightSpread={true}
          autoRefresh={true}
        />
      </div>

      {/* Volume Profile */}
      <div className="card p-6">
        <p className="text-[14px] font-semibold mb-5" style={{ color: 'var(--text)' }}>Volume Profile</p>
        <VolumeProfile
          marketId={market.id}
          bucketSize={0.01}
          lookbackDays={7}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* ML Prediction */}
        <div className="card p-6">
          <div className="flex items-center gap-2.5 mb-5">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{ background: 'var(--accent-dim)' }}
            >
              <Brain className="h-4 w-4" style={{ color: 'var(--accent)' }} />
            </div>
            <p className="text-[14px] font-semibold">ML Prediction</p>
          </div>

          {cal ? (
            <div className="space-y-3">
              {[
                { label: 'Calibrated Price', value: `${(cal.calibrated_price * 100).toFixed(1)}%`, color: 'var(--blue)' },
                { label: 'Market Price', value: `${(cal.market_price * 100).toFixed(1)}%`, color: 'var(--text-2)' },
                { label: 'Delta', value: `${cal.delta_pct > 0 ? '+' : ''}${cal.delta_pct.toFixed(1)}%`, color: cal.delta_pct > 0 ? 'var(--red)' : 'var(--green)' },
                { label: 'Edge', value: `${(cal.edge_estimate * 100).toFixed(2)}%`, color: 'var(--accent)' },
              ].map((row) => (
                <div key={row.label} className="flex items-center justify-between py-1">
                  <span className="text-[12px]" style={{ color: 'var(--text-3)' }}>{row.label}</span>
                  <span className="text-[13px] font-mono font-medium" style={{ color: row.color }}>{row.value}</span>
                </div>
              ))}
              <div className="flex items-center justify-between pt-3" style={{ borderTop: '1px solid var(--border)' }}>
                <span className="text-[12px]" style={{ color: 'var(--text-3)' }}>Direction</span>
                <span className="flex items-center gap-1.5">
                  {cal.direction === 'overpriced' ? (
                    <TrendingDown className="h-3.5 w-3.5" style={{ color: 'var(--red)' }} />
                  ) : (
                    <TrendingUp className="h-3.5 w-3.5" style={{ color: 'var(--green)' }} />
                  )}
                  <span
                    className="text-[12px] font-medium capitalize"
                    style={{ color: cal.direction === 'overpriced' ? 'var(--red)' : 'var(--green)' }}
                  >
                    {cal.direction}
                  </span>
                </span>
              </div>
            </div>
          ) : (
            <p className="text-[12px] py-8 text-center" style={{ color: 'var(--text-3)' }}>
              No prediction available
            </p>
          )}
        </div>

        {/* Claude Analysis */}
        <div className="card p-6">
          <div className="flex items-center gap-2.5 mb-5">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{ background: 'rgba(94,180,239,0.1)' }}
            >
              <Sparkles className="h-4 w-4" style={{ color: 'var(--blue)' }} />
            </div>
            <p className="text-[14px] font-semibold">Claude Analysis</p>
          </div>

          {!analysis && (
            <div className="text-center py-4">
              <p className="text-[12px] mb-5" style={{ color: 'var(--text-2)' }}>
                AI-powered deep analysis of this market including key factors, risks, and recommendations.
              </p>
              <button
                onClick={() => analyzeMutation.mutate()}
                disabled={analyzeMutation.isPending}
                className="btn"
              >
                {analyzeMutation.isPending ? (
                  <><Loader2 className="h-4 w-4 animate-spin" /> Analyzing...</>
                ) : (
                  <><Sparkles className="h-4 w-4" /> Analyze with Claude</>
                )}
              </button>
              <p className="text-[10px] mt-3" style={{ color: 'var(--text-3)' }}>
                ~$0.15 (cached forever)
              </p>
              {analyzeMutation.isError && (
                <p className="text-[12px] mt-2" style={{ color: 'var(--red)' }}>Analysis failed. Try again.</p>
              )}
            </div>
          )}

          {analysis && (
            <p className="text-[13px] leading-relaxed whitespace-pre-wrap" style={{ color: 'var(--text)' }}>
              {typeof analysis.analysis === 'string' ? analysis.analysis : JSON.stringify(analysis.analysis, null, 2)}
            </p>
          )}
        </div>

        {/* Market Sentiment */}
        <div className="card p-6">
          <div className="flex items-center gap-2.5 mb-5">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{ background: 'rgba(76,175,112,0.1)' }}
            >
              <TrendingUp className="h-4 w-4" style={{ color: 'var(--green)' }} />
            </div>
            <p className="text-[14px] font-semibold">Sentiment</p>
          </div>
          <SentimentGauge marketId={market.id} size="small" showBreakdown={true} />
        </div>
      </div>
    </div>
  )
}
