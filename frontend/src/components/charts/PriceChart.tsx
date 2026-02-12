import { useEffect, useRef, useState } from 'react'
import {
  createChart,
  ColorType,
  CrosshairMode,
  CandlestickSeries,
  LineSeries,
  AreaSeries,
  HistogramSeries
} from 'lightweight-charts'
import type { IChartApi, ISeriesApi, CandlestickData, Time, LineData, HistogramData } from 'lightweight-charts'

export type ChartType = 'candlestick' | 'line' | 'area'
export type TimeInterval = '1m' | '5m' | '15m' | '1h' | '4h' | '1d'

interface PriceDataPoint {
  timestamp: number // Unix timestamp in seconds
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface PriceChartProps {
  marketId: number
  interval?: TimeInterval
  type?: ChartType
  height?: number
  showVolume?: boolean
  showCrosshair?: boolean
  autoRefresh?: boolean // Auto-refresh every 60 seconds
  data?: PriceDataPoint[] // Optional: provide data directly (for testing)
}

export default function PriceChart({
  marketId,
  interval = '5m',
  type = 'candlestick',
  height = 400,
  showVolume = true,
  showCrosshair = true,
  autoRefresh = true,
  data: providedData,
}: PriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const mainSeriesRef = useRef<ISeriesApi<'Candlestick' | 'Line' | 'Area'> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)

  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [priceData, setPriceData] = useState<PriceDataPoint[]>([])
  const [containerMounted, setContainerMounted] = useState(false)

  // Callback ref to detect when container is actually mounted in DOM
  const containerCallbackRef = (node: HTMLDivElement | null) => {
    chartContainerRef.current = node
    if (node && !containerMounted) {
      console.log('[PriceChart] Container mounted in DOM')
      setContainerMounted(true)
    }
  }

  // Fetch price data from API
  useEffect(() => {
    if (providedData) {
      console.log('[PriceChart] Using provided data:', providedData.length, 'points')
      setPriceData(providedData)
      setIsLoading(false)
      return
    }

    const fetchPriceData = async () => {
      try {
        setIsLoading(true)
        setError(null)

        // Fetch price snapshots for this market
        const url = `/api/v1/markets/${marketId}/price-history?interval=${interval}&limit=500`
        console.log('[PriceChart] Fetching from:', url)
        const response = await fetch(url)

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const result = await response.json()
        console.log('[PriceChart] Received data:', result.data?.length || 0, 'points', result)
        setPriceData(result.data || [])
      } catch (err) {
        console.error('[PriceChart] Failed to fetch price data:', err)
        setError(err instanceof Error ? err.message : 'Failed to load chart data')
      } finally {
        setIsLoading(false)
      }
    }

    fetchPriceData()

    // Auto-refresh if enabled
    if (autoRefresh) {
      const intervalId = setInterval(fetchPriceData, 60000) // Every 60 seconds
      return () => clearInterval(intervalId)
    }
  }, [marketId, interval, autoRefresh, providedData])

  // Initialize chart
  useEffect(() => {
    if (!containerMounted || !chartContainerRef.current) {
      console.log('[PriceChart] Waiting for container to mount...')
      return
    }

    const containerWidth = chartContainerRef.current.clientWidth
    console.log('[PriceChart] Initializing chart - container width:', containerWidth, 'height:', height)

    // Don't initialize if container has no width (not laid out yet)
    if (containerWidth === 0) {
      console.warn('[PriceChart] Container width is 0, cannot initialize')
      return
    }

    console.log('[PriceChart] Creating chart instance...')

    try {
      // Create chart instance
      const chart = createChart(chartContainerRef.current, {
        width: containerWidth,
        height: showVolume ? height : height - 100,
        layout: {
          background: { type: ColorType.Solid, color: 'transparent' },
          textColor: '#9CA3AF', // Tailwind gray-400
        },
        grid: {
          vertLines: { color: '#1F2937' }, // Tailwind gray-800
          horzLines: { color: '#1F2937' },
        },
        crosshair: {
          mode: showCrosshair ? CrosshairMode.Normal : CrosshairMode.Hidden,
          vertLine: {
            width: 1,
            color: '#4B5563',
            style: 0,
            labelBackgroundColor: '#C4A24D', // Accent color
          },
          horzLine: {
            width: 1,
            color: '#4B5563',
            style: 0,
            labelBackgroundColor: '#C4A24D',
          },
        },
        rightPriceScale: {
          borderColor: '#374151',
          scaleMargins: {
            top: 0.1,
            bottom: showVolume ? 0.3 : 0.1,
          },
        },
        timeScale: {
          borderColor: '#374151',
          timeVisible: true,
          secondsVisible: interval === '1m',
        },
        handleScroll: {
          mouseWheel: true,
          pressedMouseMove: true,
          horzTouchDrag: true,
          vertTouchDrag: true,
        },
        handleScale: {
          axisPressedMouseMove: true,
          mouseWheel: true,
          pinch: true,
        },
      })

      console.log('[PriceChart] Chart created:', chart)
      console.log('[PriceChart] Chart methods:', Object.keys(chart))
      chartRef.current = chart

      // Create main price series based on type (v5 API)
      if (type === 'candlestick') {
        console.log('[PriceChart] Adding candlestick series...')
        mainSeriesRef.current = chart.addSeries(CandlestickSeries, {
          upColor: '#10B981', // Tailwind green-500
          downColor: '#EF4444', // Tailwind red-500
          borderUpColor: '#10B981',
          borderDownColor: '#EF4444',
          wickUpColor: '#10B981',
          wickDownColor: '#EF4444',
        })
      } else if (type === 'line') {
        mainSeriesRef.current = chart.addSeries(LineSeries, {
          color: '#C4A24D', // Accent color
          lineWidth: 2,
          crosshairMarkerVisible: true,
          crosshairMarkerRadius: 4,
          crosshairMarkerBorderColor: '#C4A24D',
          crosshairMarkerBackgroundColor: '#000000',
        })
      } else if (type === 'area') {
        mainSeriesRef.current = chart.addSeries(AreaSeries, {
          topColor: 'rgba(196, 162, 77, 0.4)', // Accent with alpha
          bottomColor: 'rgba(196, 162, 77, 0.0)',
          lineColor: '#C4A24D',
          lineWidth: 2,
          crosshairMarkerVisible: true,
          crosshairMarkerRadius: 4,
        })
      }

      // Create volume series if enabled (v5 API)
      if (showVolume) {
        volumeSeriesRef.current = chart.addSeries(HistogramSeries, {
          color: '#4B5563', // Tailwind gray-600
          priceFormat: {
            type: 'volume',
          },
          priceScaleId: '', // Use separate price scale
          scaleMargins: {
            top: 0.8,
            bottom: 0,
          },
        })
      }

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }

      window.addEventListener('resize', handleResize)

      console.log('[PriceChart] Chart initialization complete')
    } catch (err) {
      console.error('[PriceChart] Error during chart initialization:', err)
      setError(err instanceof Error ? err.message : 'Failed to initialize chart')
    }

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [type, height, showVolume, showCrosshair, interval, containerMounted])

  // Update chart data
  useEffect(() => {
    console.log('[PriceChart] Update effect - mainSeries:', !!mainSeriesRef.current, 'priceData:', priceData.length)

    if (!mainSeriesRef.current || !priceData.length) {
      console.log('[PriceChart] Skipping update - no series or no data')
      return
    }

    try {
      console.log('[PriceChart] Updating chart with', priceData.length, 'data points')
      if (type === 'candlestick') {
        const candlestickData: CandlestickData<Time>[] = priceData.map((d) => ({
          time: d.timestamp as Time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        }))
        console.log('[PriceChart] Setting candlestick data, first point:', candlestickData[0])
        mainSeriesRef.current.setData(candlestickData)
      } else {
        // For line and area, use close price
        const lineData: LineData<Time>[] = priceData.map((d) => ({
          time: d.timestamp as Time,
          value: d.close,
        }))
        console.log('[PriceChart] Setting line data, first point:', lineData[0])
        mainSeriesRef.current.setData(lineData)
      }

      // Update volume data
      if (showVolume && volumeSeriesRef.current) {
        const volumeData: HistogramData<Time>[] = priceData.map((d, i) => {
          const prevClose = i > 0 ? priceData[i - 1].close : d.open
          const isUp = d.close >= prevClose

          return {
            time: d.timestamp as Time,
            value: d.volume,
            color: isUp ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)', // Green/Red with alpha
          }
        })
        console.log('[PriceChart] Setting volume data')
        volumeSeriesRef.current.setData(volumeData)
      }

      // Fit content to visible range
      if (chartRef.current) {
        chartRef.current.timeScale().fitContent()
        console.log('[PriceChart] Fitted content to chart')
      }
    } catch (err) {
      console.error('[PriceChart] Failed to update chart data:', err)
    }
  }, [priceData, type, showVolume])

  console.log('[PriceChart] Render - isLoading:', isLoading, 'error:', error, 'priceData.length:', priceData.length)

  if (isLoading) {
    return (
      <div
        className="flex items-center justify-center"
        style={{ height: `${height}px`, background: 'var(--card)', borderRadius: '12px' }}
      >
        <div className="text-center">
          <div
            className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent"
            style={{ color: 'var(--accent)' }}
          />
          <p className="mt-4 text-sm" style={{ color: 'var(--text-3)' }}>
            Loading chart data...
          </p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div
        className="flex items-center justify-center"
        style={{ height: `${height}px`, background: 'var(--card)', borderRadius: '12px' }}
      >
        <div className="text-center">
          <p className="text-sm" style={{ color: 'var(--red)' }}>
            Failed to load chart
          </p>
          <p className="mt-2 text-xs" style={{ color: 'var(--text-3)' }}>
            {error}
          </p>
        </div>
      </div>
    )
  }

  if (!priceData.length) {
    return (
      <div
        className="flex items-center justify-center"
        style={{ height: `${height}px`, background: 'var(--card)', borderRadius: '12px' }}
      >
        <div className="text-center">
          <p className="text-sm" style={{ color: 'var(--text-3)' }}>
            No price data available
          </p>
          <p className="mt-2 text-xs" style={{ color: 'var(--text-3)' }}>
            Try selecting a different interval
          </p>
        </div>
      </div>
    )
  }

  console.log('[PriceChart] Rendering chart container')

  return (
    <div
      style={{
        background: 'var(--card)',
        borderRadius: '12px',
        padding: '16px',
        border: '1px solid var(--border)',
      }}
    >
      <div ref={containerCallbackRef} style={{ height: `${height}px`, width: '100%', minHeight: `${height}px` }} />
    </div>
  )
}
