import { useQuery } from '@tanstack/react-query'
import {
  Loader2,
  AlertCircle,
  ArrowLeftRight,
  RefreshCw,
} from 'lucide-react'
import apiClient from '../api/client'

interface ArbitrageOpportunity {
  id: string
  strategy_type: string
  markets: Array<{
    id: string
    question: string
    platform: string
    price_yes: number
  }>
  gross_spread: number
  net_spread: number
  estimated_profit: number
  confidence: number
  detected_at: string
}

const strategyColors: Record<string, { bg: string; text: string; border: string }> = {
  cross_platform: {
    bg: 'bg-blue-900/20',
    text: 'text-blue-400',
    border: 'border-blue-800/40',
  },
  mirror_contract: {
    bg: 'bg-purple-900/20',
    text: 'text-purple-400',
    border: 'border-purple-800/40',
  },
  multi_leg: {
    bg: 'bg-amber-900/20',
    text: 'text-amber-400',
    border: 'border-amber-800/40',
  },
  calendar_spread: {
    bg: 'bg-emerald-900/20',
    text: 'text-emerald-400',
    border: 'border-emerald-800/40',
  },
}

function getStrategyStyle(type: string) {
  return (
    strategyColors[type] ?? {
      bg: 'bg-gray-900/20',
      text: 'text-gray-400',
      border: 'border-gray-800/40',
    }
  )
}

export default function ArbitrageScanner() {
  const {
    data: opportunities,
    isLoading,
    error,
    dataUpdatedAt,
  } = useQuery<ArbitrageOpportunity[]>({
    queryKey: ['arbitrage-opportunities'],
    queryFn: async () => {
      const response = await apiClient.get('/arbitrage/opportunities')
      return response.data
    },
    refetchInterval: 15_000,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-96 text-gray-400">
        <AlertCircle className="h-12 w-12 mb-4 text-red-400" />
        <p className="text-lg font-medium text-white mb-2">
          Failed to load arbitrage data
        </p>
        <p className="text-sm">Check API connection and try again.</p>
      </div>
    )
  }

  const opps = opportunities ?? []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Arbitrage Scanner</h1>
          <p className="text-sm text-gray-400 mt-1">
            Cross-platform arbitrage opportunities detected in real time
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <RefreshCw className="h-3.5 w-3.5" />
          Updated {new Date(dataUpdatedAt).toLocaleTimeString()}
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
          <p className="text-sm text-gray-400">Active Opportunities</p>
          <p className="text-2xl font-bold text-white mt-1">{opps.length}</p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
          <p className="text-sm text-gray-400">Avg Net Spread</p>
          <p className="text-2xl font-bold text-emerald-400 mt-1">
            {opps.length > 0
              ? (
                  (opps.reduce((s, o) => s + o.net_spread, 0) / opps.length) *
                  100
                ).toFixed(2)
              : '0.00'}
            %
          </p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
          <p className="text-sm text-gray-400">Total Est. Profit</p>
          <p className="text-2xl font-bold text-white mt-1">
            $
            {opps
              .reduce((s, o) => s + o.estimated_profit, 0)
              .toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}
          </p>
        </div>
      </div>

      {/* Table */}
      <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700 text-left">
                <th className="px-4 py-3 text-gray-400 font-medium">
                  Strategy
                </th>
                <th className="px-4 py-3 text-gray-400 font-medium">
                  Markets
                </th>
                <th className="px-4 py-3 text-gray-400 font-medium text-right">
                  Gross Spread
                </th>
                <th className="px-4 py-3 text-gray-400 font-medium text-right">
                  Net Spread
                </th>
                <th className="px-4 py-3 text-gray-400 font-medium text-right">
                  Est. Profit
                </th>
                <th className="px-4 py-3 text-gray-400 font-medium text-right">
                  Confidence
                </th>
                <th className="px-4 py-3 text-gray-400 font-medium text-right">
                  Detected
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700/50">
              {opps.map((opp) => {
                const style = getStrategyStyle(opp.strategy_type)
                return (
                  <tr
                    key={opp.id}
                    className="hover:bg-gray-700/30 transition-colors"
                  >
                    <td className="px-4 py-3">
                      <span
                        className={`inline-block px-2.5 py-1 rounded text-xs font-medium ${style.bg} ${style.text} border ${style.border}`}
                      >
                        <ArrowLeftRight className="h-3 w-3 inline mr-1" />
                        {opp.strategy_type.replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="space-y-1">
                        {opp.markets.map((m) => (
                          <div key={m.id} className="flex items-center gap-2">
                            <span className="text-xs text-gray-500 capitalize w-16 flex-shrink-0">
                              {m.platform}
                            </span>
                            <span className="text-white text-xs truncate max-w-[200px]">
                              {m.question}
                            </span>
                            <span className="font-mono text-xs text-gray-400 ml-auto">
                              {(m.price_yes * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-amber-400">
                      {(opp.gross_spread * 100).toFixed(2)}%
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-emerald-400 font-medium">
                      {(opp.net_spread * 100).toFixed(2)}%
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-white font-medium">
                      $
                      {opp.estimated_profit.toLocaleString(undefined, {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span
                        className={`font-mono text-xs ${
                          opp.confidence >= 0.8
                            ? 'text-emerald-400'
                            : opp.confidence >= 0.5
                              ? 'text-amber-400'
                              : 'text-red-400'
                        }`}
                      >
                        {(opp.confidence * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right text-xs text-gray-500">
                      {new Date(opp.detected_at).toLocaleTimeString()}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {opps.length === 0 && (
          <div className="py-12 text-center text-gray-500">
            <ArrowLeftRight className="h-8 w-8 mx-auto mb-3 opacity-50" />
            <p>No arbitrage opportunities detected at the moment.</p>
            <p className="text-xs mt-1">Scanner refreshes every 15 seconds.</p>
          </div>
        )}
      </div>
    </div>
  )
}
