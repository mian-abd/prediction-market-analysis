/**
 * Reusable Error State Component
 *
 * Provides consistent error handling across the app with:
 * - Clear error message
 * - Actionable context
 * - Retry button
 */

import { AlertCircle } from 'lucide-react'

interface ErrorStateProps {
  title?: string
  message?: string
  onRetry?: () => void
  showBackendHint?: boolean
}

export default function ErrorState({
  title = 'Failed to load data',
  message = 'Something went wrong while loading this content.',
  onRetry,
  showBackendHint = true,
}: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center justify-center h-80 gap-3">
      <AlertCircle className="h-8 w-8" style={{ color: 'var(--red)' }} />
      <div className="text-center space-y-1">
        <p className="text-[14px] font-medium" style={{ color: 'var(--text)' }}>
          {title}
        </p>
        <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
          {message}
        </p>
        {showBackendHint && (
          <p className="text-[11px] pt-1" style={{ color: 'var(--text-3)' }}>
            Check that the backend is running at{' '}
            <code className="px-1 py-0.5 rounded" style={{ background: 'var(--card)' }}>
              localhost:8000
            </code>
          </p>
        )}
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-4 py-2 rounded-lg text-[13px] font-medium transition-colors mt-2"
          style={{
            background: 'var(--card)',
            color: 'var(--text)',
            border: '1px solid var(--border)',
          }}
        >
          Try Again
        </button>
      )}
    </div>
  )
}
