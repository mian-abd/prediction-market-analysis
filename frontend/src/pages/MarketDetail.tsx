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
  ExternalLink,
  AlertTriangle,
  BarChart3,
  BookOpen,
  Newspaper,
  Link2,
} from 'lucide-react'
import apiClient from '../api/client'
import PriceChart from '../components/charts/PriceChart'
import OrderbookDepth from '../components/charts/OrderbookDepth'
import VolumeProfile from '../components/charts/VolumeProfile'
import SentimentGauge from '../components/charts/SentimentGauge'
import NewsSection from '../components/NewsSection'

interface NewsArticle {
  title: string
  url: string
  domain: string
  publish_date: string
  tone: number
}

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
  news?: NewsArticle[]
}

function getRelativeTime(isoString: string): string {
  const diff = Date.now() - new Date(isoString).getTime()
  if (diff < 0) return 'just now'
  const seconds = Math.floor(diff / 1000)
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

function cleanMarketQuestion(question: string): string {
  const legCount = (question.match(/\b(yes|no)\s/gi) || []).length
  if (legCount > 1) {
    const legs = question.split(',').map(s => s.trim())
    return legs.map(l => l.replace(/^(yes|no)\s+/i, '')).join(' · ')
  }
  return question
}

function getTimeToClose(endDate: string | null): string {
  if (!endDate) return 'No end date'
  const diff = new Date(endDate).getTime() - Date.now()
  if (diff < 0) return 'Closed'
  const days = Math.floor(diff / (1000 * 60 * 60 * 24))
  if (days > 30) return `${Math.floor(days / 30)}mo`
  if (days > 0) return `${days}d`
  const hours = Math.floor(diff / (1000 * 60 * 60))
  if (hours > 0) return `${hours}h`
  return '<1h'
}

interface EnsembleData {
  ensemble_probability: number
  market_price: number
  delta: number
  delta_pct: number
  direction: string
  edge_estimate: number
  model_predictions: {
    calibration: { probability: number; weight: number }
    xgboost: { probability: number; weight: number }
    lightgbm: { probability: number; weight: number }
  }
  features_used: number
  ensemble_active: boolean
}

interface EdgeSignalData {
  market_id: number
  direction: string | null
  ensemble_prob: number
  market_price: number
  raw_edge_pct: number
  fee_cost_pct: number
  net_ev_pct: number
  kelly_fraction: number
  confidence: number
  quality_tier: string
  quality_gate: {
    passes: boolean
    reasons: string[]
  }
  model_predictions: Record<string, { probability: number; weight: number }>
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
    ensemble: EnsembleData | null
  }
  edge_signal?: EdgeSignalData | null
}

interface AnalysisResult {
  market_id: number
  question: string
  analysis: string
}

type DetailTab = 'price' | 'signals' | 'orderbook' | 'news' | 'related'

export default function MarketDetail() {
  const { id } = useParams<{ id: string }>()
  const [activeTab, setActiveTab] = useState<DetailTab>('price')
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
      <div className="space-y-4 fade-up">
        <div className="h-5 w-24 shimmer rounded" />
        <div className="card p-6 space-y-4">
          <div className="h-6 w-3/4 shimmer rounded" />
          <div className="flex gap-3">
            <div className="h-20 w-24 shimmer rounded-xl" />
            <div className="h-20 w-24 shimmer rounded-xl" />
          </div>
        </div>
        <div className="card p-6"><div className="h-[300px] shimmer rounded-lg" /></div>
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

  const ens = prediction?.models?.ensemble
  const cal = prediction?.models?.calibration
  const hasSignals = !!(ens || cal)
  const newsCount = market.news?.length ?? 0
  const relatedCount = market.cross_platform_matches?.length ?? 0

  const tabs: { key: DetailTab; label: string; icon: typeof BarChart3; count?: number }[] = [
    { key: 'price', label: 'Price', icon: BarChart3 },
    { key: 'signals', label: 'Signals', icon: Brain, count: hasSignals ? 1 : 0 },
    { key: 'orderbook', label: 'Orderbook', icon: BookOpen },
    { key: 'news', label: 'News', icon: Newspaper, count: newsCount },
    { key: 'related', label: 'Related', icon: Link2, count: relatedCount },
  ]

  return (
    <div className="space-y-5 fade-up">
      {/* Breadcrumb */}
      <Link
        to="/markets"
        className="inline-flex items-center gap-1.5 text-[12px] font-medium"
        style={{ color: 'var(--text-3)' }}
      >
        <ArrowLeft className="h-3.5 w-3.5" /> Markets
      </Link>

      {/* Hero Header */}
      <div className="card p-5">
        <div className="flex flex-col lg:flex-row lg:items-start gap-5">
          <div className="flex-1 min-w-0">
            {/* Pills */}
            <div className="flex items-center gap-2 mb-2">
              <span className="pill pill-accent capitalize">{market.platform}</span>
              <span className="pill">{market.category ?? 'other'}</span>
              <span className="pill pill-blue">{getTimeToClose(market.end_date)}</span>
              {hasSignals && <span className="signal-badge">ML Edge</span>}
            </div>

            {/* Title */}
            <h1 className="text-[20px] font-bold leading-snug" style={{ color: 'var(--text)' }}>
              {cleanMarketQuestion(market.question)}
            </h1>

            {/* Stats Row */}
            <div className="flex items-center gap-5 mt-3">
              <div className="flex items-center gap-1.5">
                <DollarSign className="h-3.5 w-3.5" style={{ color: 'var(--text-3)' }} />
                <span className="text-[12px]" style={{ color: 'var(--text-2)' }}>
                  ${(market.volume_24h ?? 0).toLocaleString()} 24h
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <DollarSign className="h-3.5 w-3.5" style={{ color: 'var(--text-3)' }} />
                <span className="text-[12px]" style={{ color: 'var(--text-2)' }}>
                  ${(market.volume_total ?? 0).toLocaleString()} total
                </span>
              </div>
              {market.last_fetched_at && (
                <span className={`freshness ${
                  (Date.now() - new Date(market.last_fetched_at).getTime()) < 30000 ? 'freshness-fresh' :
                  (Date.now() - new Date(market.last_fetched_at).getTime()) < 120000 ? 'freshness-ok' : 'freshness-stale'
                }`}>
                  <span className="freshness-dot" />
                  {getRelativeTime(market.last_fetched_at)}
                </span>
              )}
            </div>
          </div>

          {/* Price Blocks */}
          <div className="flex gap-3 flex-shrink-0">
            <div className="price-block price-block-yes">
              <span className="price-block-label" style={{ color: 'var(--green)' }}>Yes</span>
              <span className="price-block-value" style={{ color: 'var(--green)' }}>
                {((market.price_yes ?? 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="price-block price-block-no">
              <span className="price-block-label" style={{ color: 'var(--red)' }}>No</span>
              <span className="price-block-value" style={{ color: 'var(--red)' }}>
                {((market.price_no ?? 0) * 100).toFixed(1)}%
              </span>
            </div>
            {ens && (
              <div className="price-block" style={{ background: 'var(--accent-dim)' }}>
                <span className="price-block-label" style={{ color: 'var(--accent)' }}>Edge</span>
                <span className="price-block-value" style={{ color: 'var(--accent)' }}>
                  {ens.delta_pct > 0 ? '+' : ''}{ens.delta_pct.toFixed(1)}%
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="tab-bar">
        {tabs.map(({ key, label, icon: Icon, count }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`tab ${activeTab === key ? 'tab-active' : ''}`}
          >
            <span className="flex items-center gap-1.5">
              <Icon className="h-3.5 w-3.5" />
              {label}
              {count !== undefined && count > 0 && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full" style={{ background: 'var(--accent-dim)', color: 'var(--accent)' }}>
                  {count}
                </span>
              )}
            </span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="fade-in">
        {/* Price Tab */}
        {activeTab === 'price' && (
          <div className="space-y-4">
            <div className="card p-5">
              <div className="flex items-center justify-between mb-4">
                <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>Price History</p>
              </div>
              <PriceChart
                marketId={market.id}
                interval="5m"
                type="candlestick"
                height={380}
                showVolume={true}
                showCrosshair={true}
                autoRefresh={true}
              />
            </div>
            <div className="card p-5">
              <p className="text-[14px] font-semibold mb-4" style={{ color: 'var(--text)' }}>Volume Profile</p>
              <VolumeProfile marketId={market.id} bucketSize={0.01} lookbackDays={7} />
            </div>
          </div>
        )}

        {/* Signals Tab */}
        {activeTab === 'signals' && (
          <div className="space-y-4">
            {/* Experimental Warning */}
            <div
              className="flex items-start gap-3 p-4 rounded-xl"
              style={{ background: 'rgba(207,102,121,0.06)', border: '1px solid rgba(207,102,121,0.15)' }}
            >
              <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-0.5" style={{ color: 'var(--red)' }} />
              <p className="text-[12px]" style={{ color: 'var(--text-2)' }}>
                Experimental predictions — model trained on limited data. Not financial advice.
              </p>
            </div>

            {/* Ensemble Prediction */}
            {ens && (
              <div className="card p-5">
                <div className="flex items-center gap-2.5 mb-4">
                  <Brain className="h-4 w-4" style={{ color: 'var(--accent)' }} />
                  <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>ML Ensemble Prediction</p>
                  <span className="badge badge-experimental">Experimental</span>
                </div>

                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                  <div>
                    <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Model Predicts</p>
                    <p className="text-[20px] font-bold" style={{ color: 'var(--accent)' }}>
                      {(ens.ensemble_probability * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Market Price</p>
                    <p className="text-[20px] font-bold" style={{ color: 'var(--text-2)' }}>
                      {(ens.market_price * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Direction</p>
                    <div className="flex items-center gap-1 mt-1">
                      {ens.direction === 'overpriced' ? (
                        <TrendingDown className="h-4 w-4" style={{ color: 'var(--red)' }} />
                      ) : (
                        <TrendingUp className="h-4 w-4" style={{ color: 'var(--green)' }} />
                      )}
                      <span className="text-[14px] font-semibold capitalize"
                        style={{ color: ens.direction === 'overpriced' ? 'var(--red)' : 'var(--green)' }}>
                        {ens.direction}
                      </span>
                    </div>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Net Edge</p>
                    <p className="text-[20px] font-bold" style={{ color: 'var(--green)' }}>
                      {(ens.edge_estimate * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>

                {/* Model Breakdown */}
                {ens.ensemble_active && ens.model_predictions && (
                  <div className="pt-4" style={{ borderTop: '1px solid var(--border)' }}>
                    <p className="text-[11px] uppercase font-semibold mb-3" style={{ color: 'var(--text-3)' }}>
                      Model Breakdown
                    </p>
                    <div className="grid grid-cols-3 gap-3">
                      {Object.entries(ens.model_predictions).map(([name, model]) => (
                        <div key={name} className="p-3 rounded-lg" style={{ background: 'rgba(255,255,255,0.02)' }}>
                          <p className="text-[11px] capitalize mb-1" style={{ color: 'var(--text-3)' }}>{name}</p>
                          <p className="text-[16px] font-bold font-mono" style={{ color: 'var(--text)' }}>
                            {(model.probability * 100).toFixed(1)}%
                          </p>
                          <span className="text-[10px] font-mono" style={{ color: 'var(--accent)' }}>
                            weight: {(model.weight * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Calibration Only (fallback) */}
            {!ens && cal && (
              <div className="card p-5">
                <div className="flex items-center gap-2.5 mb-4">
                  <Brain className="h-4 w-4" style={{ color: 'var(--blue)' }} />
                  <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>Calibration Model</p>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <div>
                    <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Calibrated</p>
                    <p className="text-[20px] font-bold" style={{ color: 'var(--blue)' }}>
                      {(cal.calibrated_price * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Market</p>
                    <p className="text-[20px] font-bold" style={{ color: 'var(--text-2)' }}>
                      {(cal.market_price * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Delta</p>
                    <p className="text-[20px] font-bold"
                      style={{ color: cal.delta_pct > 0 ? 'var(--red)' : 'var(--green)' }}>
                      {cal.delta_pct > 0 ? '+' : ''}{cal.delta_pct.toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>Edge</p>
                    <p className="text-[20px] font-bold" style={{ color: 'var(--accent)' }}>
                      {(cal.edge_estimate * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Edge Signal & Quality Gates */}
            {prediction?.edge_signal && prediction.edge_signal.direction && (
              <div className="card p-5">
                <div className="flex items-center gap-2.5 mb-4">
                  <TrendingUp className="h-4 w-4" style={{ color: 'var(--accent)' }} />
                  <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>Trading Signal</p>
                  <span className={`pill ${
                    prediction.edge_signal.quality_tier === 'high' ? 'pill-green' :
                    prediction.edge_signal.quality_tier === 'medium' ? 'pill-accent' :
                    prediction.edge_signal.quality_tier === 'speculative' ? 'pill-red' : ''
                  }`}>
                    {prediction.edge_signal.quality_tier}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                  {[
                    { label: 'Direction', value: prediction.edge_signal.direction.replace('_', ' '), color: prediction.edge_signal.direction === 'buy_yes' ? 'var(--green)' : 'var(--red)' },
                    { label: 'Net EV', value: `${prediction.edge_signal.net_ev_pct.toFixed(1)}%`, color: 'var(--accent)' },
                    { label: 'Kelly Size', value: `${(prediction.edge_signal.kelly_fraction * 100).toFixed(2)}%`, color: 'var(--text)' },
                    { label: 'Confidence', value: `${(prediction.edge_signal.confidence * 100).toFixed(0)}%`, color: 'var(--blue)' },
                  ].map((s) => (
                    <div key={s.label} className="text-center py-2 rounded-lg" style={{ background: 'rgba(255,255,255,0.03)' }}>
                      <p className="text-[10px] uppercase mb-0.5" style={{ color: 'var(--text-3)' }}>{s.label}</p>
                      <p className="text-[14px] font-bold capitalize" style={{ color: s.color }}>{s.value}</p>
                    </div>
                  ))}
                </div>

                {/* Quality Gate Checklist */}
                <div className="pt-3" style={{ borderTop: '1px solid var(--border)' }}>
                  <p className="text-[10px] uppercase font-semibold mb-2" style={{ color: 'var(--text-3)' }}>Quality Gates</p>
                  <div className="flex items-center gap-2 mb-1">
                    <span style={{ color: prediction.edge_signal.quality_gate.passes ? 'var(--green)' : 'var(--red)' }}>
                      {prediction.edge_signal.quality_gate.passes ? '✓' : '✗'}
                    </span>
                    <span className="text-[11px]" style={{ color: 'var(--text-2)' }}>
                      {prediction.edge_signal.quality_gate.passes ? 'All gates passed' : 'Some gates failed'}
                    </span>
                  </div>
                  {prediction.edge_signal.quality_gate.reasons.length > 0 && (
                    <div className="ml-5 space-y-0.5">
                      {prediction.edge_signal.quality_gate.reasons.map((reason, i) => (
                        <p key={i} className="text-[10px]" style={{ color: 'var(--red)' }}>
                          {reason}
                        </p>
                      ))}
                    </div>
                  )}

                  <div className="flex gap-3 mt-2 text-[10px]" style={{ color: 'var(--text-3)' }}>
                    <span>Raw edge: {prediction.edge_signal.raw_edge_pct.toFixed(1)}%</span>
                    <span>Fees: {prediction.edge_signal.fee_cost_pct.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            )}

            {/* Claude Analysis */}
            <div className="card p-5">
              <div className="flex items-center gap-2.5 mb-4">
                <Sparkles className="h-4 w-4" style={{ color: 'var(--blue)' }} />
                <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>Claude Analysis</p>
              </div>
              {!analysis ? (
                <div className="text-center py-6">
                  <p className="text-[12px] mb-4" style={{ color: 'var(--text-2)' }}>
                    AI-powered deep analysis: key factors, risks, and recommendations.
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
                  <p className="text-[10px] mt-2" style={{ color: 'var(--text-3)' }}>~$0.15 (cached forever)</p>
                  {analyzeMutation.isError && (
                    <p className="text-[12px] mt-2" style={{ color: 'var(--red)' }}>Analysis failed. Try again.</p>
                  )}
                </div>
              ) : (
                <p className="text-[13px] leading-relaxed whitespace-pre-wrap" style={{ color: 'var(--text)' }}>
                  {typeof analysis.analysis === 'string' ? analysis.analysis : JSON.stringify(analysis.analysis, null, 2)}
                </p>
              )}
            </div>

            {/* Sentiment */}
            <div className="card p-5">
              <div className="flex items-center gap-2.5 mb-4">
                <TrendingUp className="h-4 w-4" style={{ color: 'var(--green)' }} />
                <p className="text-[14px] font-semibold" style={{ color: 'var(--text)' }}>Sentiment</p>
              </div>
              <SentimentGauge marketId={market.id} size="small" showBreakdown={true} />
            </div>

            {/* No Signals */}
            {!hasSignals && (
              <div className="flex flex-col items-center justify-center h-40 gap-2">
                <Brain className="h-5 w-5" style={{ color: 'var(--text-3)' }} />
                <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
                  No ML prediction available for this market
                </p>
              </div>
            )}
          </div>
        )}

        {/* Orderbook Tab */}
        {activeTab === 'orderbook' && (
          <div className="card p-5">
            <p className="text-[14px] font-semibold mb-4" style={{ color: 'var(--text)' }}>Orderbook Depth</p>
            <OrderbookDepth
              marketId={market.id}
              maxDepth={10}
              highlightSpread={true}
              autoRefresh={true}
            />
          </div>
        )}

        {/* News Tab */}
        {activeTab === 'news' && (
          <div className="card p-5">
            {newsCount > 0 ? (
              <>
                <p className="text-[14px] font-semibold mb-4" style={{ color: 'var(--text)' }}>
                  Recent News ({newsCount} articles)
                </p>
                <NewsSection articles={market.news || []} />
              </>
            ) : (
              <div className="flex flex-col items-center justify-center h-40 gap-2">
                <Newspaper className="h-5 w-5" style={{ color: 'var(--text-3)' }} />
                <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
                  No news articles found for this market
                </p>
              </div>
            )}
          </div>
        )}

        {/* Related Markets Tab */}
        {activeTab === 'related' && (
          <div className="space-y-4">
            {/* Cross-Platform Matches */}
            {relatedCount > 0 ? (
              <div className="card p-5">
                <p className="text-[14px] font-semibold mb-4" style={{ color: 'var(--text)' }}>
                  Cross-Platform Matches
                </p>
                <div className="space-y-2">
                  {market.cross_platform_matches.map((match) => (
                    <Link
                      key={match.id}
                      to={`/markets/${match.id}`}
                      className="flex items-center justify-between p-3 rounded-xl card-hover"
                      style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border)', textDecoration: 'none' }}
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="pill pill-accent capitalize text-[10px]">{match.platform}</span>
                          <span className="badge badge-info">{(match.similarity * 100).toFixed(0)}% match</span>
                        </div>
                        <p className="text-[13px] truncate" style={{ color: 'var(--text)' }}>
                          {match.question}
                        </p>
                      </div>
                      <div className="text-right flex-shrink-0 ml-4">
                        <p className="text-[16px] font-bold" style={{ color: 'var(--green)' }}>
                          {(match.price_yes * 100).toFixed(1)}%
                        </p>
                        <p className="text-[10px]" style={{ color: 'var(--text-3)' }}>YES</p>
                      </div>
                    </Link>
                  ))}
                </div>
              </div>
            ) : (
              <div className="card p-5">
                <div className="flex flex-col items-center justify-center h-40 gap-2">
                  <Link2 className="h-5 w-5" style={{ color: 'var(--text-3)' }} />
                  <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
                    No cross-platform matches found
                  </p>
                  <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                    This market may be unique to {market.platform}
                  </p>
                </div>
              </div>
            )}

            {/* Explore Correlation Link */}
            <Link
              to="/correlation"
              className="card card-hover p-4 flex items-center gap-3 block"
              style={{ textDecoration: 'none' }}
            >
              <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'var(--accent-dim)' }}>
                <ExternalLink className="h-4 w-4" style={{ color: 'var(--accent)' }} />
              </div>
              <div>
                <p className="text-[13px] font-medium" style={{ color: 'var(--text)' }}>Explore Correlations</p>
                <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                  See which markets move together and identify cluster risk
                </p>
              </div>
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}
