/**
 * Market Sentiment Gauge
 *
 * Speedometer-style gauge showing market sentiment:
 * - Composite score from price momentum, volume trend, OBI
 * - Color zones: red (bearish) → yellow (neutral) → green (bullish)
 * - Displays sentiment breakdown
 */

import { useEffect, useState } from 'react'
import { Loader2, TrendingUp, TrendingDown, Minus, Gauge } from 'lucide-react'
import apiClient from '../../api/client'
import EmptyState from '../EmptyState'

interface SentimentData {
  sentiment_score: number // 0-100
  price_momentum: number // -1 to 1
  volume_trend: number // -1 to 1
  orderbook_imbalance: number // -1 to 1
  current_price: number
  price_change_24h: number
}

interface SentimentGaugeProps {
  marketId: number
  size?: 'small' | 'large'
  showBreakdown?: boolean
}

export default function SentimentGauge({
  marketId,
  size = 'large',
  showBreakdown = true,
}: SentimentGaugeProps) {
  const [data, setData] = useState<SentimentData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchSentiment = async () => {
    setLoading(true)
    setError(null)

    try {
      // Fetch market data
      const marketResponse = await apiClient.get(`/markets/${marketId}`)
      const market = marketResponse.data

      // Fetch price history for momentum calculation
      const priceHistoryResponse = await apiClient.get(
        `/markets/${marketId}/price-history?interval=1h&limit=24`
      )
      const priceHistory = priceHistoryResponse.data.data || []

      // Fetch orderbook for OBI
      let obi = 0
      try {
        const orderbookResponse = await apiClient.get(`/markets/${marketId}/orderbook`)
        obi = orderbookResponse.data.obi || 0
      } catch {
        // Orderbook may not be available for all markets
      }

      // Calculate price momentum (24h price change)
      const currentPrice = market.price_yes || 0.5
      let priceMomentum = 0
      let priceChangePercent = 0
      if (priceHistory.length >= 2) {
        const oldestPrice = priceHistory[0].close
        const priceChange = currentPrice - oldestPrice
        priceMomentum = Math.max(-1, Math.min(1, priceChange * 10)) // Scale to [-1, 1] for sentiment
        priceChangePercent = oldestPrice > 0 ? (currentPrice - oldestPrice) / oldestPrice : 0
      }

      // Calculate volume trend (recent vs older)
      let volumeTrend = 0
      if (priceHistory.length >= 12) {
        const recentVolume = priceHistory.slice(-6).reduce((sum: number, c: any) => sum + (c.volume || 0), 0)
        const olderVolume = priceHistory.slice(0, 6).reduce((sum: number, c: any) => sum + (c.volume || 0), 0)
        if (olderVolume > 0) {
          const volumeChange = (recentVolume - olderVolume) / olderVolume
          volumeTrend = Math.max(-1, Math.min(1, volumeChange))
        }
      }

      // Calculate composite sentiment score (0-100)
      // 40% price momentum, 30% volume trend, 30% OBI
      const sentimentScore =
        50 + // Base (neutral)
        (priceMomentum * 0.4 * 50) + // Price contributes ±20
        (volumeTrend * 0.3 * 50) + // Volume contributes ±15
        (obi * 0.3 * 50) // OBI contributes ±15

      const clampedScore = Math.max(0, Math.min(100, sentimentScore))

      setData({
        sentiment_score: clampedScore,
        price_momentum: priceMomentum,
        volume_trend: volumeTrend,
        orderbook_imbalance: obi,
        current_price: currentPrice,
        price_change_24h: priceChangePercent, // Actual ratio (0.05 = 5%)
      })
    } catch (err: any) {
      setError('Failed to calculate sentiment')
      console.error(err)
    }
  }

  useEffect(() => {
    fetchSentiment()
  }, [marketId])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader2 className="h-5 w-5 animate-spin" style={{ color: 'var(--text-3)' }} />
      </div>
    )
  }

  if (error || !data) {
    return (
      <EmptyState
        icon={Gauge}
        title={error || 'No sentiment data'}
        message="Unable to calculate market sentiment at this time."
      />
    )
  }

  const score = data.sentiment_score
  const getSentimentColor = (s: number): string => {
    if (s < 33) return '#CF6679' // Bearish (red)
    if (s < 66) return '#C4A24D' // Neutral (yellow/accent)
    return '#4CAF70' // Bullish (green)
  }

  const getSentimentLabel = (s: number): string => {
    if (s < 33) return 'Bearish'
    if (s < 66) return 'Neutral'
    return 'Bullish'
  }

  const color = getSentimentColor(score)
  const label = getSentimentLabel(score)

  // Gauge dimensions
  const gaugeSize = size === 'small' ? 120 : 160
  const radius = gaugeSize / 2 - 20
  const cx = gaugeSize / 2
  const cy = gaugeSize / 2

  // Calculate needle angle (0° = bearish, 180° = bullish)
  const needleAngle = 180 - (score / 100) * 180

  return (
    <div className="space-y-4">
      {/* Gauge */}
      <div className="flex items-center justify-center">
        <svg width={gaugeSize} height={gaugeSize / 1.3} viewBox={`0 0 ${gaugeSize} ${gaugeSize / 1.3}`}>
          {/* Background arc zones */}
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#CF6679" />
              <stop offset="50%" stopColor="#C4A24D" />
              <stop offset="100%" stopColor="#4CAF70" />
            </linearGradient>
          </defs>

          {/* Gauge arc */}
          <path
            d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
            fill="none"
            stroke="url(#gaugeGradient)"
            strokeWidth="12"
            strokeLinecap="round"
          />

          {/* Needle */}
          <g transform={`rotate(${needleAngle} ${cx} ${cy})`}>
            <line
              x1={cx}
              y1={cy}
              x2={cx}
              y2={cy - radius + 5}
              stroke={color}
              strokeWidth="3"
              strokeLinecap="round"
            />
            <circle cx={cx} cy={cy} r="5" fill={color} />
          </g>

          {/* Score text */}
          <text
            x={cx}
            y={cy + 20}
            textAnchor="middle"
            style={{ fill: color, fontSize: size === 'small' ? '24px' : '32px', fontWeight: 'bold' }}
          >
            {score.toFixed(0)}
          </text>
          <text
            x={cx}
            y={cy + 38}
            textAnchor="middle"
            style={{ fill: color, fontSize: '12px', fontWeight: '600' }}
          >
            {label}
          </text>
        </svg>
      </div>

      {/* Breakdown */}
      {showBreakdown && (
        <div className="space-y-2">
          <div className="flex items-center justify-between py-1.5 px-3 rounded-lg" style={{ background: 'var(--card)' }}>
            <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>Price Momentum</span>
            <div className="flex items-center gap-1.5">
              {data.price_momentum > 0 ? (
                <TrendingUp className="h-3 w-3" style={{ color: 'var(--green)' }} />
              ) : data.price_momentum < 0 ? (
                <TrendingDown className="h-3 w-3" style={{ color: 'var(--red)' }} />
              ) : (
                <Minus className="h-3 w-3" style={{ color: 'var(--text-3)' }} />
              )}
              <span className="text-[11px] font-mono" style={{ color: 'var(--text)' }}>
                {data.price_change_24h > 0 ? '+' : ''}{(data.price_change_24h * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          <div className="flex items-center justify-between py-1.5 px-3 rounded-lg" style={{ background: 'var(--card)' }}>
            <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>Volume Trend</span>
            <div className="flex items-center gap-1.5">
              {data.volume_trend > 0 ? (
                <TrendingUp className="h-3 w-3" style={{ color: 'var(--green)' }} />
              ) : data.volume_trend < 0 ? (
                <TrendingDown className="h-3 w-3" style={{ color: 'var(--red)' }} />
              ) : (
                <Minus className="h-3 w-3" style={{ color: 'var(--text-3)' }} />
              )}
              <span className="text-[11px] font-mono" style={{ color: 'var(--text)' }}>
                {data.volume_trend > 0 ? '+' : ''}{(data.volume_trend * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          <div className="flex items-center justify-between py-1.5 px-3 rounded-lg" style={{ background: 'var(--card)' }}>
            <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>Orderbook Imbalance</span>
            <div className="flex items-center gap-1.5">
              {data.orderbook_imbalance > 0 ? (
                <TrendingUp className="h-3 w-3" style={{ color: 'var(--green)' }} />
              ) : data.orderbook_imbalance < 0 ? (
                <TrendingDown className="h-3 w-3" style={{ color: 'var(--red)' }} />
              ) : (
                <Minus className="h-3 w-3" style={{ color: 'var(--text-3)' }} />
              )}
              <span className="text-[11px] font-mono" style={{ color: 'var(--text)' }}>
                {(data.orderbook_imbalance * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
