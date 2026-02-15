/**
 * Orderbook Depth Visualization
 *
 * Displays bid/ask depth with cumulative visualization.
 * Features:
 * - Dual area chart (green bids, red asks)
 * - Spread highlighted in middle
 * - Order Book Imbalance (OBI) indicator
 * - Hover shows price/size details
 */

import { useEffect, useState } from 'react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { Loader2, TrendingUp, TrendingDown, BookOpen } from 'lucide-react'
import apiClient from '../../api/client'
import EmptyState from '../EmptyState'

interface OrderbookLevel {
  price: number
  size: number
  cumulative: number
}

interface OrderbookData {
  market_id: number
  timestamp: string
  best_bid: number
  best_ask: number
  spread: number
  obi: number
  bids: OrderbookLevel[]
  asks: OrderbookLevel[]
  bid_depth_total: number
  ask_depth_total: number
}

interface OrderbookDepthProps {
  marketId: number
  maxDepth?: number
  highlightSpread?: boolean
  autoRefresh?: boolean
}

export default function OrderbookDepth({
  marketId,
  maxDepth = 10,
  highlightSpread = true,
  autoRefresh = true,
}: OrderbookDepthProps) {
  const [data, setData] = useState<OrderbookData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchOrderbook = async () => {
    try {
      const response = await apiClient.get(`/markets/${marketId}/orderbook`)
      setData(response.data)
      setError(null)
    } catch (err: any) {
      if (err.response?.status === 404) {
        setError('No orderbook data available')
      } else {
        setError('Failed to load orderbook')
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchOrderbook()

    if (autoRefresh) {
      const interval = setInterval(fetchOrderbook, 60_000) // Refresh every 60 seconds
      return () => clearInterval(interval)
    }
  }, [marketId, autoRefresh])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-5 w-5 animate-spin" style={{ color: 'var(--text-3)' }} />
      </div>
    )
  }

  if (error || !data) {
    return (
      <EmptyState
        icon={BookOpen}
        title={error || 'No orderbook data'}
        message="Orderbook snapshots are collected periodically for Polymarket markets. Data may not be available for all markets yet."
      />
    )
  }

  if (data.bids.length === 0 && data.asks.length === 0) {
    return (
      <EmptyState
        icon={BookOpen}
        title="Empty orderbook"
        message="No bids or asks available for this market at this time."
      />
    )
  }

  // Prepare chart data: combine bids and asks
  // Bids: show cumulative depth going left from best bid
  // Asks: show cumulative depth going right from best ask
  const chartData: any[] = []

  // Add bids (reversed, so highest bid is closest to spread)
  const bidsToShow = data.bids.slice(0, maxDepth).reverse()
  bidsToShow.forEach((bid) => {
    chartData.push({
      price: bid.price,
      bidDepth: bid.cumulative,
      askDepth: null,
      side: 'bid',
    })
  })

  // Add asks
  const asksToShow = data.asks.slice(0, maxDepth)
  asksToShow.forEach((ask) => {
    chartData.push({
      price: ask.price,
      bidDepth: null,
      askDepth: ask.cumulative,
      side: 'ask',
    })
  })

  // Sort by price (bids descending, then asks ascending)
  chartData.sort((a, b) => a.price - b.price)

  // Calculate OBI percentage
  const obiPct = data.obi !== null ? (data.obi * 100).toFixed(1) : null
  const obiBullish = data.obi > 0

  return (
    <div className="space-y-4">
      {/* Header with stats */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div>
            <p className="text-[10px] uppercase mb-0.5" style={{ color: 'var(--text-3)' }}>
              Best Bid
            </p>
            <p className="text-[15px] font-mono font-semibold" style={{ color: 'var(--green)' }}>
              {data.best_bid ? data.best_bid.toFixed(3) : '-'}
            </p>
          </div>
          <div>
            <p className="text-[10px] uppercase mb-0.5" style={{ color: 'var(--text-3)' }}>
              Spread
            </p>
            <p className="text-[15px] font-mono font-semibold" style={{ color: 'var(--text)' }}>
              {data.spread ? (data.spread * 100).toFixed(2) + '%' : '-'}
            </p>
          </div>
          <div>
            <p className="text-[10px] uppercase mb-0.5" style={{ color: 'var(--text-3)' }}>
              Best Ask
            </p>
            <p className="text-[15px] font-mono font-semibold" style={{ color: 'var(--red)' }}>
              {data.best_ask ? data.best_ask.toFixed(3) : '-'}
            </p>
          </div>
        </div>

        {/* Order Book Imbalance */}
        {obiPct !== null && (
          <div className="flex items-center gap-2">
            {obiBullish ? (
              <TrendingUp className="h-4 w-4" style={{ color: 'var(--green)' }} />
            ) : (
              <TrendingDown className="h-4 w-4" style={{ color: 'var(--red)' }} />
            )}
            <div>
              <p className="text-[10px] uppercase" style={{ color: 'var(--text-3)' }}>
                OBI
              </p>
              <p
                className="text-[13px] font-mono font-medium"
                style={{ color: obiBullish ? 'var(--green)' : 'var(--red)' }}
              >
                {obiPct}%
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Depth chart */}
      <div style={{ width: '100%', height: '256px', minHeight: '256px' }}>
        <ResponsiveContainer width="100%" height={256}>
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="bidGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#4CAF70" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#4CAF70" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="askGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#CF6679" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#CF6679" stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />

            <XAxis
              dataKey="price"
              stroke="rgba(255,255,255,0.06)"
              tick={{ fill: '#48484A', fontSize: 11 }}
              tickFormatter={(value) => value.toFixed(3)}
            />

            <YAxis
              stroke="rgba(255,255,255,0.06)"
              tick={{ fill: '#48484A', fontSize: 11 }}
              label={{
                value: 'Cumulative Depth',
                angle: -90,
                position: 'insideLeft',
                style: { fill: '#48484A', fontSize: 11 },
              }}
            />

            <Tooltip
              contentStyle={{
                backgroundColor: '#1A1A1C',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '12px',
                color: '#FFF',
                fontSize: '12px',
                padding: '8px 12px',
              }}
              formatter={(value, name) => {
                if (value == null) return null
                return [Number(value).toFixed(2), name === 'bidDepth' ? 'Bid Depth' : 'Ask Depth']
              }}
              labelFormatter={(price) => `Price: ${Number(price).toFixed(3)}`}
            />

            {/* Spread reference line */}
            {highlightSpread && data.best_bid && data.best_ask && (
              <>
                <ReferenceLine
                  x={data.best_bid}
                  stroke="#4CAF70"
                  strokeDasharray="3 3"
                  strokeWidth={1}
                />
                <ReferenceLine
                  x={data.best_ask}
                  stroke="#CF6679"
                  strokeDasharray="3 3"
                  strokeWidth={1}
                />
              </>
            )}

            {/* Bids (green area) */}
            <Area
              type="stepAfter"
              dataKey="bidDepth"
              stroke="#4CAF70"
              fill="url(#bidGradient)"
              strokeWidth={2}
              name="Bids"
              connectNulls={false}
            />

            {/* Asks (red area) */}
            <Area
              type="stepBefore"
              dataKey="askDepth"
              stroke="#CF6679"
              fill="url(#askGradient)"
              strokeWidth={2}
              name="Asks"
              connectNulls={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Depth totals */}
      <div className="flex items-center justify-between text-[12px]">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded" style={{ background: '#4CAF70' }} />
          <span style={{ color: 'var(--text-3)' }}>Total Bid Depth:</span>
          <span className="font-mono font-medium" style={{ color: 'var(--green)' }}>
            {data.bid_depth_total ? data.bid_depth_total.toFixed(2) : '0'}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded" style={{ background: '#CF6679' }} />
          <span style={{ color: 'var(--text-3)' }}>Total Ask Depth:</span>
          <span className="font-mono font-medium" style={{ color: 'var(--red)' }}>
            {data.ask_depth_total ? data.ask_depth_total.toFixed(2) : '0'}
          </span>
        </div>
      </div>
    </div>
  )
}
