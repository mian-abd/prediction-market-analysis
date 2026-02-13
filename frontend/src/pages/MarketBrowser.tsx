import { useState, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  Search,
  Loader2,
  AlertCircle,
  ChevronRight,
  ChevronLeft,
  LayoutGrid,
  List,
  Calendar,
} from 'lucide-react'
import apiClient from '../api/client'

interface Market {
  id: number
  question: string
  price_yes: number | null
  price_no: number | null
  volume_24h: number | null
  volume_total: number | null
  category: string | null
  platform: string
  end_date: string | null
  liquidity: number | null
  updated_at: string | null
  last_fetched_at: string | null
  is_resolved: boolean
  is_active: boolean
}

// Helper: Calculate time until market closes
function getTimeToClose(endDate: string | null): string | null {
  if (!endDate) return null
  const now = Date.now()
  const end = new Date(endDate).getTime()
  const diff = end - now

  if (diff < 0) return 'Closed'

  const days = Math.floor(diff / (1000 * 60 * 60 * 24))
  const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60))

  if (days > 30) return `${Math.floor(days / 30)}mo`
  if (days > 0) return `${days}d`
  if (hours > 0) return `${hours}h`
  return '<1h'
}

type SortField = 'volume_24h' | 'price_yes' | 'end_date' | 'liquidity' | 'updated_at'
type StatusFilter = 'active' | 'resolved' | 'all'

const PAGE_SIZE = 24

export default function MarketBrowser() {
  const navigate = useNavigate()
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('all')
  const [platformFilter, setPlatformFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('active')
  const [sortField, setSortField] = useState<SortField>('volume_24h')
  const [sortAsc, setSortAsc] = useState(false)
  const [viewMode, setViewMode] = useState<'list' | 'grid'>('grid')
  const [page, setPage] = useState(0)

  // Debounce search input
  const searchTimeout = useRef<ReturnType<typeof setTimeout> | null>(null)
  const handleSearch = (value: string) => {
    setSearch(value)
    if (searchTimeout.current) clearTimeout(searchTimeout.current)
    searchTimeout.current = setTimeout(() => {
      setDebouncedSearch(value)
      setPage(0)
    }, 300)
  }

  // Fetch categories from server
  const { data: categoriesData } = useQuery<{ category: string; count: number }[]>({
    queryKey: ['categories'],
    queryFn: async () => {
      const response = await apiClient.get('/markets/categories')
      return response.data
    },
  })
  const categories = categoriesData ?? []

  // Fetch markets — backend handles quality filtering (dead prices, combos, expired)
  const { data, isLoading, error } = useQuery<{ markets: Market[]; total: number }>({
    queryKey: ['markets', debouncedSearch, categoryFilter, platformFilter, statusFilter, sortField, sortAsc, page],
    queryFn: async () => {
      const params: Record<string, any> = {
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
        sort_by: sortField,
        sort_dir: sortAsc ? 'asc' : 'desc',
        exclude_combos: true,
      }
      if (debouncedSearch) params.search = debouncedSearch
      if (categoryFilter !== 'all') params.category = categoryFilter
      if (platformFilter !== 'all') params.platform = platformFilter
      if (statusFilter === 'active') {
        params.is_active = true
        params.is_resolved = false
      } else if (statusFilter === 'resolved') {
        params.is_active = false
        params.is_resolved = true
        params.exclude_combos = false
      } else {
        // "all" — show everything, disable quality filters
        params.is_active = false
        params.exclude_combos = false
      }
      const response = await apiClient.get('/markets', { params })
      return response.data
    },
    refetchInterval: 30_000,
    staleTime: 60_000,
    gcTime: 300_000,
    placeholderData: (previousData) => previousData,
  })

  const markets = data?.markets ?? []
  const totalPages = Math.ceil((data?.total ?? 0) / PAGE_SIZE)

  if (isLoading && page === 0) {
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
    <div className="space-y-5 fade-up">
      {/* Title */}
      <div>
        <h1 className="text-[26px] font-bold" style={{ color: 'var(--text)' }}>Markets</h1>
        <p className="text-[13px] mt-1" style={{ color: 'var(--text-2)' }}>
          Browse and analyze prediction markets across platforms
        </p>
      </div>

      {/* Category Tabs */}
      <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-hide">
        <button
          onClick={() => { setCategoryFilter('all'); setPage(0) }}
          className="px-4 py-2 rounded-full text-[12px] font-medium whitespace-nowrap transition-colors flex-shrink-0"
          style={{
            background: categoryFilter === 'all' ? 'var(--accent)' : 'var(--card)',
            color: categoryFilter === 'all' ? '#000' : 'var(--text-2)',
            border: `1px solid ${categoryFilter === 'all' ? 'var(--accent)' : 'var(--border)'}`,
          }}
        >
          All
        </button>
        {categories.map((cat) => (
          <button
            key={cat.category}
            onClick={() => { setCategoryFilter(cat.category); setPage(0) }}
            className="px-4 py-2 rounded-full text-[12px] font-medium whitespace-nowrap transition-colors flex-shrink-0"
            style={{
              background: categoryFilter === cat.category ? 'var(--accent)' : 'var(--card)',
              color: categoryFilter === cat.category ? '#000' : 'var(--text-2)',
              border: `1px solid ${categoryFilter === cat.category ? 'var(--accent)' : 'var(--border)'}`,
            }}
          >
            {cat.category.charAt(0).toUpperCase() + cat.category.slice(1)}
            <span className="ml-1.5 opacity-60">{cat.count.toLocaleString()}</span>
          </button>
        ))}
      </div>

      {/* Search + Filters + View Toggle */}
      <div className="flex flex-col gap-3">
        {/* Search bar */}
        <div className="relative flex-1">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4" style={{ color: 'var(--text-3)' }} />
          <input
            type="text"
            placeholder="Search markets..."
            value={search}
            onChange={(e) => handleSearch(e.target.value)}
            className="input pl-11"
          />
        </div>

        {/* Filters row */}
        <div className="flex flex-wrap gap-3 items-center">
          {/* Status filter */}
          <div className="flex gap-1.5 items-center">
            {(['active', 'resolved', 'all'] as StatusFilter[]).map((s) => (
              <button
                key={s}
                onClick={() => { setStatusFilter(s); setPage(0) }}
                className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-colors capitalize"
                style={{
                  background: statusFilter === s ? 'var(--accent)' : 'var(--card)',
                  color: statusFilter === s ? '#000' : 'var(--text-3)',
                  border: `1px solid ${statusFilter === s ? 'var(--accent)' : 'var(--border)'}`,
                }}
              >
                {s}
              </button>
            ))}
          </div>

          {/* Separator */}
          <div className="w-px h-5" style={{ background: 'var(--border)' }} />

          {/* Platform filter */}
          <div className="flex gap-1.5 items-center">
            {['all', 'polymarket', 'kalshi'].map((p) => (
              <button
                key={p}
                onClick={() => { setPlatformFilter(p); setPage(0) }}
                className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-colors capitalize"
                style={{
                  background: platformFilter === p ? 'var(--card)' : 'transparent',
                  color: platformFilter === p ? 'var(--text)' : 'var(--text-3)',
                  border: `1px solid ${platformFilter === p ? 'var(--border)' : 'transparent'}`,
                }}
              >
                {p === 'all' ? 'All' : p.charAt(0).toUpperCase() + p.slice(1)}
              </button>
            ))}
          </div>

          {/* Spacer */}
          <div className="flex-1" />

          {/* View toggle */}
          <div className="flex gap-1">
            <button
              onClick={() => setViewMode('list')}
              className="p-2 rounded-lg transition-colors"
              style={{
                background: viewMode === 'list' ? 'var(--card)' : 'transparent',
                color: viewMode === 'list' ? 'var(--text)' : 'var(--text-3)',
                border: `1px solid ${viewMode === 'list' ? 'var(--border)' : 'transparent'}`,
              }}
            >
              <List className="h-4 w-4" />
            </button>
            <button
              onClick={() => setViewMode('grid')}
              className="p-2 rounded-lg transition-colors"
              style={{
                background: viewMode === 'grid' ? 'var(--card)' : 'transparent',
                color: viewMode === 'grid' ? 'var(--text)' : 'var(--text-3)',
                border: `1px solid ${viewMode === 'grid' ? 'var(--border)' : 'transparent'}`,
              }}
            >
              <LayoutGrid className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Sort + Count */}
      <div className="flex items-center justify-between gap-3">
        <p className="text-[12px] font-medium" style={{ color: 'var(--text-2)' }}>
          {data?.total?.toLocaleString() ?? 0} markets
        </p>
        <div className="flex gap-1.5 items-center">
          {([
            ['volume_24h', 'Trending'],
            ['updated_at', 'Latest'],
            ['liquidity', 'Liquidity'],
            ['end_date', 'Closes Soon'],
            ['price_yes', 'Price'],
          ] as [SortField, string][]).map(([field, label]) => (
            <button
              key={field}
              onClick={() => {
                if (sortField === field) {
                  setSortAsc(!sortAsc)
                } else {
                  setSortField(field)
                  setSortAsc(false)
                }
                setPage(0)
              }}
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-colors whitespace-nowrap flex items-center gap-1"
              style={{
                background: sortField === field ? 'var(--card)' : 'transparent',
                color: sortField === field ? 'var(--text)' : 'var(--text-3)',
                border: sortField === field ? '1px solid var(--border)' : '1px solid transparent',
              }}
            >
              {label}
              {sortField === field && (
                <span style={{ color: 'var(--accent)' }}>{sortAsc ? '↑' : '↓'}</span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Grid View */}
      {viewMode === 'grid' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {markets.map((market) => {
            const p = market.price_yes ?? 0
            const pNo = market.price_no ?? (1 - p)
            const ttc = getTimeToClose(market.end_date)
            return (
              <div
                key={market.id}
                onClick={() => navigate(`/markets/${market.id}`)}
                className="card card-hover p-5 cursor-pointer group"
                style={{ opacity: market.is_resolved ? 0.6 : 1 }}
              >
                {/* Platform + Category + Time */}
                <div className="flex items-center gap-2 mb-3 flex-wrap">
                  <span className="pill pill-accent capitalize text-[10px]">{market.platform}</span>
                  <span className="pill text-[10px]">{market.category ?? 'other'}</span>
                  {market.is_resolved && (
                    <span className="pill pill-red text-[10px] font-semibold">Resolved</span>
                  )}
                  {!market.is_resolved && ttc && ttc !== 'Closed' && (
                    <span className="pill pill-blue text-[10px] font-semibold">{ttc}</span>
                  )}
                </div>

                {/* Question */}
                <p className="text-[13px] font-medium line-clamp-2 mb-4" style={{ color: 'var(--text)', minHeight: '2.6em' }}>
                  {market.question}
                </p>

                {/* YES / NO prices */}
                <div className="flex gap-2 mb-3">
                  <div className="flex-1 text-center py-2.5 rounded-xl" style={{ background: 'rgba(76,175,112,0.08)' }}>
                    <p className="text-[10px] font-semibold uppercase" style={{ color: 'var(--green)' }}>Yes</p>
                    <p className="text-[20px] font-bold font-mono" style={{ color: 'var(--green)' }}>
                      {(p * 100).toFixed(0)}%
                    </p>
                  </div>
                  <div className="flex-1 text-center py-2.5 rounded-xl" style={{ background: 'rgba(207,102,121,0.08)' }}>
                    <p className="text-[10px] font-semibold uppercase" style={{ color: 'var(--red)' }}>No</p>
                    <p className="text-[20px] font-bold font-mono" style={{ color: 'var(--red)' }}>
                      {(pNo * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between pt-3" style={{ borderTop: '1px solid var(--border)' }}>
                  <span className="text-[11px] font-mono" style={{ color: 'var(--text-3)' }}>
                    ${((market.volume_24h ?? 0) >= 1_000_000
                      ? `${((market.volume_24h ?? 0) / 1_000_000).toFixed(1)}M`
                      : (market.volume_24h ?? 0) >= 1_000
                      ? `${((market.volume_24h ?? 0) / 1_000).toFixed(0)}K`
                      : (market.volume_24h ?? 0).toLocaleString()
                    )} Vol
                  </span>
                  {market.end_date && (
                    <span className="flex items-center gap-1 text-[11px]" style={{ color: 'var(--text-3)' }}>
                      <Calendar className="h-3 w-3" />
                      {new Date(market.end_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                    </span>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* List View */}
      {viewMode === 'list' && (
        <div className="space-y-1.5">
          {markets.map((market) => {
            const p = market.price_yes ?? 0
            return (
              <div
                key={market.id}
                onClick={() => navigate(`/markets/${market.id}`)}
                className="card card-hover flex items-center gap-4 px-5 py-4 cursor-pointer group"
                style={{ opacity: market.is_resolved ? 0.6 : 1 }}
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
                    {market.end_date && (
                      <span className="text-[10px]" style={{ color: 'var(--text-3)' }}>
                        {new Date(market.end_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                      </span>
                    )}
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
      )}

      {/* Empty State */}
      {markets.length === 0 && !isLoading && (
        <div className="flex flex-col items-center py-16 gap-2">
          <Search className="h-6 w-6 mb-2" style={{ color: 'var(--text-3)' }} />
          <p className="text-[13px]" style={{ color: 'var(--text-2)' }}>No markets found</p>
          <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>Try a different search or filter</p>
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-4 pt-2">
          <button
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            className="flex items-center gap-1 px-4 py-2 rounded-lg text-[12px] font-medium transition-colors"
            style={{
              background: page > 0 ? 'var(--card)' : 'transparent',
              color: page > 0 ? 'var(--text)' : 'var(--text-3)',
              border: '1px solid var(--border)',
              opacity: page === 0 ? 0.4 : 1,
            }}
          >
            <ChevronLeft className="h-3.5 w-3.5" /> Previous
          </button>
          <span className="text-[12px] font-mono" style={{ color: 'var(--text-3)' }}>
            {page + 1} / {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => p + 1)}
            disabled={(page + 1) >= totalPages}
            className="flex items-center gap-1 px-4 py-2 rounded-lg text-[12px] font-medium transition-colors"
            style={{
              background: (page + 1) < totalPages ? 'var(--card)' : 'transparent',
              color: (page + 1) < totalPages ? 'var(--text)' : 'var(--text-3)',
              border: '1px solid var(--border)',
              opacity: (page + 1) >= totalPages ? 0.4 : 1,
            }}
          >
            Next <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>
      )}
    </div>
  )
}
