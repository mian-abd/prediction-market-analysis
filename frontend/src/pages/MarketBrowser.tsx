import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  Search,
  Loader2,
  AlertCircle,
  ChevronRight,
} from 'lucide-react'
import apiClient from '../api/client'

interface Market {
  id: number
  question: string
  price_yes: number | null
  price_no: number | null
  volume_24h: number | null
  category: string | null
  platform: string
}

type SortField = 'volume_24h' | 'price_yes' | 'question'
type SortDir = 'asc' | 'desc'

export default function MarketBrowser() {
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [categoryFilter, setCategoryFilter] = useState<string>('all')
  const [sortField, setSortField] = useState<SortField>('volume_24h')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  const { data, isLoading, error } = useQuery<{ markets: Market[]; total: number }>({
    queryKey: ['markets'],
    queryFn: async () => {
      const response = await apiClient.get('/markets', { params: { limit: 200 } })
      return response.data
    },
    refetchInterval: 30_000,
  })

  const markets = data?.markets ?? []

  const categories = useMemo(() => {
    const cats = new Set(markets.map((m) => m.category ?? 'other'))
    return Array.from(cats).sort()
  }, [markets])

  const filtered = useMemo(() => {
    let result = [...markets]
    if (search) {
      const lower = search.toLowerCase()
      result = result.filter((m) => m.question.toLowerCase().includes(lower))
    }
    if (categoryFilter !== 'all') {
      result = result.filter((m) => (m.category ?? 'other') === categoryFilter)
    }
    result.sort((a, b) => {
      let cmp: number
      if (sortField === 'question') {
        cmp = a.question.localeCompare(b.question)
      } else {
        cmp = ((a[sortField] as number) ?? 0) - ((b[sortField] as number) ?? 0)
      }
      return sortDir === 'desc' ? -cmp : cmp
    })
    return result
  }, [markets, search, categoryFilter, sortField, sortDir])

  function toggleSort(field: SortField) {
    if (sortField === field) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortField(field); setSortDir('desc') }
  }

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
        <p className="text-[14px] font-medium">Failed to load markets</p>
        <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>Check API connection.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6 fade-up">
      {/* Title */}
      <div>
        <h1 className="text-[26px] font-bold" style={{ color: 'var(--text)' }}>Markets</h1>
        <p className="text-[13px] mt-1" style={{ color: 'var(--text-2)' }}>
          Browse and analyze prediction markets across platforms
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4" style={{ color: 'var(--text-3)' }} />
          <input
            type="text"
            placeholder="Search markets..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="input pl-11"
          />
        </div>
        <select
          value={categoryFilter}
          onChange={(e) => setCategoryFilter(e.target.value)}
          className="input"
          style={{ width: 'auto', minWidth: '160px' }}
        >
          <option value="all">All Categories</option>
          {categories.map((cat) => (
            <option key={cat} value={cat}>{cat}</option>
          ))}
        </select>
      </div>

      {/* Sort + Count */}
      <div className="flex items-center justify-between">
        <p className="text-[11px]" style={{ color: 'var(--text-3)' }}>
          {filtered.length} markets{data?.total ? ` of ${data.total.toLocaleString()}` : ''}
        </p>
        <div className="flex gap-1.5">
          {([['question', 'Name'], ['price_yes', 'Price'], ['volume_24h', 'Volume']] as [SortField, string][]).map(
            ([field, label]) => (
              <button
                key={field}
                onClick={() => toggleSort(field)}
                className="px-3 py-1 rounded-lg text-[11px] font-medium transition-colors"
                style={{
                  background: sortField === field ? 'var(--card)' : 'transparent',
                  color: sortField === field ? 'var(--text)' : 'var(--text-3)',
                  border: sortField === field ? '1px solid var(--border)' : '1px solid transparent',
                }}
              >
                {label} {sortField === field && (sortDir === 'desc' ? '↓' : '↑')}
              </button>
            ),
          )}
        </div>
      </div>

      {/* Markets List */}
      <div className="space-y-1.5">
        {filtered.slice(0, 80).map((market) => {
          const p = market.price_yes ?? 0
          return (
            <div
              key={market.id}
              onClick={() => navigate(`/markets/${market.id}`)}
              className="card card-hover flex items-center gap-4 px-5 py-4 cursor-pointer group"
            >
              {/* Price */}
              <div
                className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 text-[13px] font-bold font-mono"
                style={{
                  background: p >= 0.5 ? 'rgba(76,175,112,0.1)' : 'rgba(207,102,121,0.1)',
                  color: p >= 0.5 ? 'var(--green)' : 'var(--red)',
                }}
              >
                {(p * 100).toFixed(0)}%
              </div>

              {/* Question */}
              <div className="flex-1 min-w-0">
                <p className="text-[13px] font-medium line-clamp-1" style={{ color: 'var(--text)' }}>
                  {market.question}
                </p>
                <div className="flex items-center gap-2 mt-1.5">
                  <span className="pill">{market.category ?? 'other'}</span>
                  <span className="pill pill-accent capitalize">{market.platform}</span>
                </div>
              </div>

              {/* Volume */}
              <div className="text-right flex-shrink-0">
                <p className="text-[11px] mb-0.5" style={{ color: 'var(--text-3)' }}>24h Vol</p>
                <p className="text-[13px] font-mono" style={{ color: 'var(--text-2)' }}>
                  ${(market.volume_24h ?? 0).toLocaleString()}
                </p>
              </div>

              <ChevronRight
                className="h-4 w-4 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                style={{ color: 'var(--text-3)' }}
              />
            </div>
          )
        })}
      </div>

      {filtered.length === 0 && (
        <div className="flex flex-col items-center py-16 gap-2">
          <Search className="h-6 w-6 mb-2" style={{ color: 'var(--text-3)' }} />
          <p className="text-[13px]" style={{ color: 'var(--text-2)' }}>No markets found</p>
          <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>Try a different search or filter</p>
        </div>
      )}
    </div>
  )
}
