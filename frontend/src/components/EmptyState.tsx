/**
 * Reusable Empty State Component
 *
 * Provides consistent empty states across the app with:
 * - Icon
 * - Clear message
 * - Optional CTA
 */

import type { ReactNode, ComponentType } from 'react'

interface EmptyStateProps {
  icon: ComponentType<{ className?: string; style?: React.CSSProperties }>
  title: string
  message?: string
  action?: {
    label: string
    onClick: () => void
  }
  children?: ReactNode
}

export default function EmptyState({
  icon: Icon,
  title,
  message,
  action,
  children,
}: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center h-80 gap-3">
      <Icon className="h-6 w-6 mb-1" style={{ color: 'var(--text-3)' }} />
      <div className="text-center space-y-1 max-w-sm">
        <p className="text-[14px] font-medium" style={{ color: 'var(--text-2)' }}>
          {title}
        </p>
        {message && (
          <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
            {message}
          </p>
        )}
      </div>
      {action && (
        <button
          onClick={action.onClick}
          className="px-4 py-2 rounded-lg text-[13px] font-medium transition-colors mt-2"
          style={{
            background: 'var(--accent)',
            color: '#000',
          }}
        >
          {action.label}
        </button>
      )}
      {children}
    </div>
  )
}
