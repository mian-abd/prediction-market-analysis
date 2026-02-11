import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
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
} from 'lucide-react'
import apiClient from '../api/client'

interface Market {
  id: string
  question: string
  description: string
  price_yes: number
  price_no: number
  volume_24h: number
  total_volume: number
  category: string
  platform: string
  status: string
  created_at: string
  close_date: string
  price_history: Array<{
    timestamp: string
    price_yes: number
    price_no: number
  }>
}

interface Prediction {
  market_id: string
  predicted_probability: number
  confidence: number
  model_name: string
  features_used: string[]
  edge: number
}

interface Analysis {
  market_id: string
  summary: string
  key_factors: string[]
  recommendation: string
  confidence_assessment: string
}

export default function MarketDetail() {
  const { id } = useParams<{ id: string }>()
  const [analysis, setAnalysis] = useState<Analysis | null>(null)

  const {
    data: market,
    isLoading,
    error,
  } = useQuery<Market>({
    queryKey: ['market', id],
    queryFn: async () => {
      const response = await apiClient.get(`/markets/${id}`)
      return response.data
    },
    refetchInterval: 15_000,
  })

  const { data: prediction, isLoading: predictionLoading } =
    useQuery<Prediction>({
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
      return response.data as Analysis
    },
    onSuccess: (data) => {
      setAnalysis(data)
    },
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    )
  }

  if (error || !market) {
    return (
      <div className="flex flex-col items-center justify-center h-96 text-gray-400">
        <AlertCircle className="h-12 w-12 mb-4 text-red-400" />
        <p className="text-lg font-medium text-white mb-2">
          Market not found
        </p>
        <Link to="/markets" className="text-blue-400 hover:underline text-sm">
          Back to Markets
        </Link>
      </div>
    )
  }

  const chartData = (market.price_history ?? []).map((point) => ({
    ...point,
    price_yes_pct: point.price_yes * 100,
    price_no_pct: point.price_no * 100,
    date: new Date(point.timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    }),
  }))

  return (
    <div className="space-y-6">
      {/* Back link */}
      <Link
        to="/markets"
        className="inline-flex items-center gap-1.5 text-sm text-gray-400 hover:text-white transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Markets
      </Link>

      {/* Market Header */}
      <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
        <div className="flex flex-col lg:flex-row lg:items-start gap-6">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-300">
                {market.category}
              </span>
              <span className="px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-400 capitalize">
                {market.platform}
              </span>
              <span
                className={`px-2 py-0.5 rounded text-xs font-medium ${
                  market.status === 'active'
                    ? 'bg-emerald-900/50 text-emerald-400'
                    : 'bg-gray-700 text-gray-400'
                }`}
              >
                {market.status}
              </span>
            </div>
            <h1 className="text-xl font-bold text-white mb-2">
              {market.question}
            </h1>
            {market.description && (
              <p className="text-sm text-gray-400 leading-relaxed">
                {market.description}
              </p>
            )}
          </div>

          {/* Price Cards */}
          <div className="flex gap-3 lg:flex-shrink-0">
            <div className="bg-emerald-900/20 border border-emerald-800/50 rounded-lg p-4 text-center min-w-[100px]">
              <p className="text-xs text-emerald-400 mb-1">YES</p>
              <p className="text-2xl font-bold text-emerald-400 font-mono">
                {(market.price_yes * 100).toFixed(1)}%
              </p>
            </div>
            <div className="bg-red-900/20 border border-red-800/50 rounded-lg p-4 text-center min-w-[100px]">
              <p className="text-xs text-red-400 mb-1">NO</p>
              <p className="text-2xl font-bold text-red-400 font-mono">
                {(market.price_no * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6 pt-6 border-t border-gray-700">
          <div className="flex items-center gap-2">
            <DollarSign className="h-4 w-4 text-gray-500" />
            <div>
              <p className="text-xs text-gray-500">24h Volume</p>
              <p className="text-sm font-medium text-white">
                ${market.volume_24h?.toLocaleString() ?? '0'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <DollarSign className="h-4 w-4 text-gray-500" />
            <div>
              <p className="text-xs text-gray-500">Total Volume</p>
              <p className="text-sm font-medium text-white">
                ${market.total_volume?.toLocaleString() ?? '0'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-gray-500" />
            <div>
              <p className="text-xs text-gray-500">Created</p>
              <p className="text-sm font-medium text-white">
                {market.created_at
                  ? new Date(market.created_at).toLocaleDateString()
                  : 'N/A'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-gray-500" />
            <div>
              <p className="text-xs text-gray-500">Closes</p>
              <p className="text-sm font-medium text-white">
                {market.close_date
                  ? new Date(market.close_date).toLocaleDateString()
                  : 'N/A'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Price Chart */}
      {chartData.length > 0 && (
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4">
            Price History
          </h2>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={chartData}
                margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
              >
                <defs>
                  <linearGradient
                    id="colorYes"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="5%"
                      stopColor="#10b981"
                      stopOpacity={0.3}
                    />
                    <stop
                      offset="95%"
                      stopColor="#10b981"
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="date"
                  stroke="#6b7280"
                  tick={{ fill: '#9ca3af', fontSize: 12 }}
                />
                <YAxis
                  domain={[0, 100]}
                  stroke="#6b7280"
                  tick={{ fill: '#9ca3af', fontSize: 12 }}
                  tickFormatter={(v: number) => `${v}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#f9fafb',
                  }}
                  formatter={(value: number | undefined) => [`${(value ?? 0).toFixed(1)}%`]}
                />
                <Area
                  type="monotone"
                  dataKey="price_yes_pct"
                  stroke="#10b981"
                  fill="url(#colorYes)"
                  strokeWidth={2}
                  name="Yes Price"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ML Prediction Panel */}
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="h-5 w-5 text-purple-400" />
            <h2 className="text-lg font-semibold text-white">ML Prediction</h2>
          </div>

          {predictionLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-purple-400" />
            </div>
          ) : prediction ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">
                  Predicted Probability
                </span>
                <span className="text-lg font-bold text-white font-mono">
                  {(prediction.predicted_probability * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Model Confidence</span>
                <span className="text-sm font-medium text-white">
                  {(prediction.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Edge vs Market</span>
                <span
                  className={`flex items-center gap-1 text-sm font-medium ${
                    prediction.edge > 0 ? 'text-emerald-400' : 'text-red-400'
                  }`}
                >
                  {prediction.edge > 0 ? (
                    <TrendingUp className="h-3.5 w-3.5" />
                  ) : (
                    <TrendingDown className="h-3.5 w-3.5" />
                  )}
                  {(prediction.edge * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Model</span>
                <span className="text-xs text-gray-300 bg-gray-700 px-2 py-0.5 rounded">
                  {prediction.model_name}
                </span>
              </div>
              {prediction.features_used && (
                <div>
                  <p className="text-sm text-gray-400 mb-2">Features Used</p>
                  <div className="flex flex-wrap gap-1.5">
                    {prediction.features_used.map((f) => (
                      <span
                        key={f}
                        className="px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-300"
                      >
                        {f}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-gray-500 py-4">
              No prediction available for this market.
            </p>
          )}
        </div>

        {/* Claude Analysis Panel */}
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="h-5 w-5 text-amber-400" />
            <h2 className="text-lg font-semibold text-white">
              Claude Analysis
            </h2>
          </div>

          {!analysis && (
            <div className="text-center py-6">
              <p className="text-sm text-gray-400 mb-4">
                Get an AI-powered analysis of this market including key factors,
                risks, and recommendations.
              </p>
              <button
                onClick={() => analyzeMutation.mutate()}
                disabled={analyzeMutation.isPending}
                className="inline-flex items-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:cursor-not-allowed rounded-lg text-sm font-medium text-white transition-colors"
              >
                {analyzeMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    Analyze with Claude
                  </>
                )}
              </button>
              {analyzeMutation.isError && (
                <p className="text-sm text-red-400 mt-3">
                  Analysis failed. Please try again.
                </p>
              )}
            </div>
          )}

          {analysis && (
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-400 mb-1">Summary</p>
                <p className="text-sm text-white leading-relaxed">
                  {analysis.summary}
                </p>
              </div>
              {analysis.key_factors && analysis.key_factors.length > 0 && (
                <div>
                  <p className="text-sm text-gray-400 mb-2">Key Factors</p>
                  <ul className="space-y-1.5">
                    {analysis.key_factors.map((factor, i) => (
                      <li
                        key={i}
                        className="flex items-start gap-2 text-sm text-gray-300"
                      >
                        <span className="text-blue-400 mt-0.5">--</span>
                        {factor}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              <div>
                <p className="text-sm text-gray-400 mb-1">Recommendation</p>
                <p className="text-sm text-white font-medium">
                  {analysis.recommendation}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400 mb-1">
                  Confidence Assessment
                </p>
                <p className="text-sm text-gray-300">
                  {analysis.confidence_assessment}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
