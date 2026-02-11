import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  Search,
  Filter,
  ArrowUpDown,
  ChevronUp,
  ChevronDown,
  Loader2,
  AlertCircle,
} from 'lucide-react'
import apiClient from '../api/client'

interface Market {
  id: string
  question: string
  price_yes: number
  price_no: number
  volume_24h: number
  category: string
  platform: string
  status: string
}

type SortField = 'volume_24h' | 'price_yes' | 'question'
type SortDir = 'asc' | 'desc'

export default function MarketBrowser() {
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [categoryFilter, setCategoryFilter] = useState<string>('all')
  const [sortField, setSortField] = useState<SortField>('volume_24h')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  const { data: markets, isLoading, error } = useQuery<Market[]>({
    queryKey: ['markets'],
    queryFn: async () => {
      const response = await apiClient.get('/markets')
      return response.data
    },
    refetchInterval: 30_000,
  })

  const categories = useMemo(() => {
    if (!markets) return []
    const cats = new Set(markets.map((m) => m.category))
    return Array.from(cats).sort()
  }, [markets])

  const filtered = useMemo(() => {
    if (!markets) return []

    let result = [...markets]

    // Search filter
    if (search) {
      const lower = search.toLowerCase()
      result = result.filter((m) =>
        m.question.toLowerCase().includes(lower),
      )
    }

    // Category filter
    if (categoryFilter !== 'all') {
      result = result.filter((m) => m.category === categoryFilter)
    }

    // Sort
    result.sort((a, b) => {
      let cmp: number
      if (sortField === 'question') {
        cmp = a.question.localeCompare(b.question)
      } else {
        cmp = (a[sortField] ?? 0) - (b[sortField] ?? 0)
      }
      return sortDir === 'desc' ? -cmp : cmp
    })

    return result
  }, [markets, search, categoryFilter, sortField, sortDir])

  function toggleSort(field: SortField) {
    if (sortField === field) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortField(field)
      setSortDir('desc')
    }
  }

  function SortIcon({ field }: { field: SortField }) {
    if (sortField !== field) {
      return <ArrowUpDown className="h-3.5 w-3.5 text-gray-500" />
    }
    return sortDir === 'asc' ? (
      <ChevronUp className="h-3.5 w-3.5 text-blue-400" />
    ) : (
      <ChevronDown className="h-3.5 w-3.5 text-blue-400" />
    )
  }

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
          Failed to load markets
        </p>
        <p className="text-sm">Check API connection and try again.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Markets</h1>
        <p className="text-sm text-gray-400 mt-1">
          Browse and analyze prediction markets across platforms
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
          <input
            type="text"
            placeholder="Search markets..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          />
        </div>
        <div className="relative">
          <Filter className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500" />
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="pl-10 pr-8 py-2.5 bg-gray-800 border border-gray-700 rounded-lg text-sm text-white appearance-none focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 cursor-pointer"
          >
            <option value="all">All Categories</option>
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Results count */}
      <p className="text-xs text-gray-500">
        {filtered.length} market{filtered.length !== 1 ? 's' : ''} found
      </p>

      {/* Table */}
      <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700 text-left">
                <th
                  className="px-4 py-3 text-gray-400 font-medium cursor-pointer hover:text-white"
                  onClick={() => toggleSort('question')}
                >
                  <div className="flex items-center gap-1.5">
                    Question
                    <SortIcon field="question" />
                  </div>
                </th>
                <th
                  className="px-4 py-3 text-gray-400 font-medium cursor-pointer hover:text-white text-right"
                  onClick={() => toggleSort('price_yes')}
                >
                  <div className="flex items-center justify-end gap-1.5">
                    Yes Price
                    <SortIcon field="price_yes" />
                  </div>
                </th>
                <th
                  className="px-4 py-3 text-gray-400 font-medium cursor-pointer hover:text-white text-right"
                  onClick={() => toggleSort('volume_24h')}
                >
                  <div className="flex items-center justify-end gap-1.5">
                    24h Volume
                    <SortIcon field="volume_24h" />
                  </div>
                </th>
                <th className="px-4 py-3 text-gray-400 font-medium">
                  Category
                </th>
                <th className="px-4 py-3 text-gray-400 font-medium">
                  Platform
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700/50">
              {filtered.map((market) => (
                <tr
                  key={market.id}
                  className="hover:bg-gray-700/30 cursor-pointer transition-colors"
                  onClick={() => navigate(`/markets/${market.id}`)}
                >
                  <td className="px-4 py-3">
                    <p className="text-white font-medium line-clamp-2">
                      {market.question}
                    </p>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span
                      className={`font-mono font-semibold ${
                        market.price_yes >= 0.7
                          ? 'text-emerald-400'
                          : market.price_yes <= 0.3
                            ? 'text-red-400'
                            : 'text-amber-400'
                      }`}
                    >
                      {(market.price_yes * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-gray-300">
                    ${market.volume_24h.toLocaleString()}
                  </td>
                  <td className="px-4 py-3">
                    <span className="inline-block px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-300">
                      {market.category}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-gray-400 capitalize text-xs">
                      {market.platform}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {filtered.length === 0 && (
          <div className="py-12 text-center text-gray-500">
            No markets match your search criteria.
          </div>
        )}
      </div>
    </div>
  )
}
