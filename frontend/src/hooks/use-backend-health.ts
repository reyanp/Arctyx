/**
 * React Query hook for checking DataFoundry backend health status
 */

import { useQuery } from '@tanstack/react-query';
import { checkHealth, HealthResponse } from '@/lib/api';

export interface BackendHealthStatus {
  isHealthy: boolean;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  data: HealthResponse | undefined;
  refetch: () => void;
}

/**
 * Hook to check if the DataFoundry backend is healthy and available
 * 
 * @param options.enabled - Whether to automatically fetch on mount (default: true)
 * @param options.refetchInterval - How often to refetch in ms (default: 30000ms / 30s)
 * @returns Backend health status and data
 * 
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { isHealthy, isLoading, data } = useBackendHealth();
 *   
 *   if (isLoading) return <div>Checking connection...</div>;
 *   if (!isHealthy) return <div>Backend unavailable</div>;
 *   
 *   return <div>Connected to backend v{data?.version}</div>;
 * }
 * ```
 */
export function useBackendHealth(options?: {
  enabled?: boolean;
  refetchInterval?: number;
}): BackendHealthStatus {
  const {
    data,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ['backend-health'],
    queryFn: checkHealth,
    enabled: options?.enabled ?? true,
    refetchInterval: options?.refetchInterval ?? 30000, // Refetch every 30 seconds
    retry: 2,
    retryDelay: 1000,
  });

  const isHealthy = !isError && !isLoading && data?.status === 'healthy';

  return {
    isHealthy,
    isLoading,
    isError,
    error: error as Error | null,
    data,
    refetch,
  };
}

