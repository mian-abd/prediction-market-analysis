import { Newspaper, ExternalLink, TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface NewsArticle {
  title: string
  url: string
  domain: string
  publish_date: string
  tone: number // -10 to +10
  social_image?: string
}

interface NewsSectionProps {
  articles: NewsArticle[]
  isLoading?: boolean
}

export default function NewsSection({ articles, isLoading = false }: NewsSectionProps) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="text-center">
          <div
            className="inline-block h-6 w-6 animate-spin rounded-full border-3 border-solid border-current border-r-transparent"
            style={{ color: 'var(--accent)' }}
          />
          <p className="mt-3 text-[12px]" style={{ color: 'var(--text-3)' }}>
            Loading news...
          </p>
        </div>
      </div>
    )
  }

  if (!articles || articles.length === 0) {
    return (
      <div className="text-center py-8">
        <Newspaper className="h-8 w-8 mx-auto mb-3" style={{ color: 'var(--text-3)' }} />
        <p className="text-[13px]" style={{ color: 'var(--text-3)' }}>
          No recent news found
        </p>
      </div>
    )
  }

  // Helper to get sentiment icon and color
  const getSentimentIndicator = (tone: number) => {
    if (tone > 2) {
      return { Icon: TrendingUp, color: 'var(--green)', label: 'Positive' }
    } else if (tone < -2) {
      return { Icon: TrendingDown, color: 'var(--red)', label: 'Negative' }
    } else {
      return { Icon: Minus, color: 'var(--text-3)', label: 'Neutral' }
    }
  }

  // Parse GDELT date format: YYYYMMDDHHMMSS → human readable
  const formatDate = (gdeltDate: string) => {
    if (!gdeltDate || gdeltDate.length < 8) return 'Recently'

    const year = gdeltDate.slice(0, 4)
    const month = gdeltDate.slice(4, 6)
    const day = gdeltDate.slice(6, 8)

    const date = new Date(`${year}-${month}-${day}`)
    const now = new Date()
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24))

    if (diffDays === 0) return 'Today'
    if (diffDays === 1) return 'Yesterday'
    if (diffDays < 7) return `${diffDays} days ago`

    return `${month}/${day}/${year.slice(2)}`
  }

  return (
    <div className="space-y-3">
      {articles.map((article, i) => {
        const sentiment = getSentimentIndicator(article.tone)
        const SentimentIcon = sentiment.Icon

        return (
          <a
            key={i}
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            className="block p-4 rounded-xl transition-all duration-200"
            style={{
              background: 'var(--card)',
              border: '1px solid var(--border)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'var(--accent)'
              e.currentTarget.style.transform = 'translateY(-1px)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'var(--border)'
              e.currentTarget.style.transform = 'translateY(0)'
            }}
          >
            <div className="flex items-start gap-3">
              {/* Sentiment indicator */}
              <div
                className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center"
                style={{ background: `${sentiment.color}15` }}
              >
                <SentimentIcon className="h-4 w-4" style={{ color: sentiment.color }} />
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <h4
                  className="text-[13px] font-medium line-clamp-2 leading-snug"
                  style={{ color: 'var(--text)' }}
                >
                  {article.title}
                </h4>

                <div className="flex items-center gap-3 mt-2">
                  <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                    {article.domain}
                  </span>
                  <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                    •
                  </span>
                  <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                    {formatDate(article.publish_date)}
                  </span>
                  <span className="text-[11px]" style={{ color: 'var(--text-3)' }}>
                    •
                  </span>
                  <span className="text-[10px] uppercase font-medium" style={{ color: sentiment.color }}>
                    {sentiment.label}
                  </span>
                </div>
              </div>

              {/* External link icon */}
              <ExternalLink className="h-3.5 w-3.5 flex-shrink-0" style={{ color: 'var(--text-3)' }} />
            </div>
          </a>
        )
      })}
    </div>
  )
}
