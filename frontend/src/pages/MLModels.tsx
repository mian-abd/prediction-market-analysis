import { useQuery } from '@tanstack/react-query'
import {
  Loader2,
  AlertCircle,
  Brain,
  CheckCircle2,
  XCircle,
  Clock,
} from 'lucide-react'
import apiClient from '../api/client'

interface MLModel {
  id: string
  name: string
  model_type: string
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  brier_score: number
  training_samples: number
  last_trained: string
  status: string
  features: string[]
}

export default function MLModels() {
  const {
    data: models,
    isLoading,
    error,
  } = useQuery<MLModel[]>({
    queryKey: ['ml-models'],
    queryFn: async () => {
      const response = await apiClient.get('/models')
      return response.data
    },
    refetchInterval: 60_000,
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
          Failed to load models
        </p>
        <p className="text-sm">Check API connection and try again.</p>
      </div>
    )
  }

  const modelList = models ?? []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">ML Models</h1>
        <p className="text-sm text-gray-400 mt-1">
          Machine learning model performance and metrics
        </p>
      </div>

      {/* Model Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {modelList.map((model) => (
          <div
            key={model.id}
            className="bg-gray-800 border border-gray-700 rounded-xl p-6 hover:border-gray-600 transition-colors"
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-900/30 rounded-lg">
                  <Brain className="h-5 w-5 text-purple-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-white">{model.name}</h3>
                  <p className="text-xs text-gray-500 capitalize">
                    {model.model_type}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-1.5">
                {model.status === 'active' ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                ) : model.status === 'training' ? (
                  <Clock className="h-4 w-4 text-amber-400" />
                ) : (
                  <XCircle className="h-4 w-4 text-red-400" />
                )}
                <span
                  className={`text-xs font-medium capitalize ${
                    model.status === 'active'
                      ? 'text-emerald-400'
                      : model.status === 'training'
                        ? 'text-amber-400'
                        : 'text-red-400'
                  }`}
                >
                  {model.status}
                </span>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-3 gap-3 mb-4">
              <div className="bg-gray-900 rounded-lg p-3 text-center">
                <p className="text-xs text-gray-500 mb-1">Accuracy</p>
                <p className="text-lg font-bold text-white font-mono">
                  {(model.accuracy * 100).toFixed(1)}%
                </p>
              </div>
              <div className="bg-gray-900 rounded-lg p-3 text-center">
                <p className="text-xs text-gray-500 mb-1">Precision</p>
                <p className="text-lg font-bold text-white font-mono">
                  {(model.precision * 100).toFixed(1)}%
                </p>
              </div>
              <div className="bg-gray-900 rounded-lg p-3 text-center">
                <p className="text-xs text-gray-500 mb-1">Recall</p>
                <p className="text-lg font-bold text-white font-mono">
                  {(model.recall * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="flex items-center justify-between px-3 py-2 bg-gray-900 rounded-lg">
                <span className="text-xs text-gray-400">F1 Score</span>
                <span className="text-sm font-mono text-white font-medium">
                  {model.f1_score.toFixed(3)}
                </span>
              </div>
              <div className="flex items-center justify-between px-3 py-2 bg-gray-900 rounded-lg">
                <span className="text-xs text-gray-400">Brier Score</span>
                <span className="text-sm font-mono text-white font-medium">
                  {model.brier_score.toFixed(4)}
                </span>
              </div>
            </div>

            {/* Meta info */}
            <div className="flex items-center justify-between text-xs text-gray-500 pt-3 border-t border-gray-700">
              <span>
                {model.training_samples.toLocaleString()} training samples
              </span>
              <span>
                Last trained:{' '}
                {model.last_trained
                  ? new Date(model.last_trained).toLocaleDateString()
                  : 'N/A'}
              </span>
            </div>

            {/* Features */}
            {model.features && model.features.length > 0 && (
              <div className="mt-3 pt-3 border-t border-gray-700">
                <p className="text-xs text-gray-500 mb-2">Features</p>
                <div className="flex flex-wrap gap-1.5">
                  {model.features.map((f) => (
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
        ))}
      </div>

      {modelList.length === 0 && (
        <div className="bg-gray-800 border border-gray-700 rounded-xl py-16 text-center">
          <Brain className="h-12 w-12 mx-auto mb-4 text-gray-600" />
          <p className="text-gray-400">No ML models configured.</p>
          <p className="text-xs text-gray-500 mt-1">
            Models will appear here once training is complete.
          </p>
        </div>
      )}
    </div>
  )
}
