/**
 * Backend Status Indicator Component
 * 
 * Displays the connection status to the DataFoundry backend API
 */

import { useBackendHealth } from '@/hooks/use-backend-health';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Activity, AlertCircle, Loader2 } from 'lucide-react';

interface BackendStatusIndicatorProps {
  variant?: 'badge' | 'icon' | 'full';
}

export function BackendStatusIndicator({ variant = 'badge' }: BackendStatusIndicatorProps) {
  const { isHealthy, isLoading, isError, data } = useBackendHealth();

  // Icon variant - just a small status dot
  if (variant === 'icon') {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center gap-1.5 cursor-help">
            {isLoading ? (
              <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
            ) : isHealthy ? (
              <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            ) : (
              <div className="h-2 w-2 rounded-full bg-red-500" />
            )}
          </div>
        </TooltipTrigger>
        <TooltipContent>
          {isLoading ? (
            <p>Checking backend...</p>
          ) : isHealthy ? (
            <div className="space-y-1">
              <p className="font-medium">Backend Connected</p>
              <p className="text-xs text-muted-foreground">Version: {data?.version}</p>
            </div>
          ) : (
            <p className="text-destructive">Backend Unavailable</p>
          )}
        </TooltipContent>
      </Tooltip>
    );
  }

  // Badge variant - shows status with text
  if (variant === 'badge') {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge 
            variant={isHealthy ? 'default' : isError ? 'destructive' : 'secondary'}
            className="gap-1.5 cursor-help"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin" />
                <span className="text-xs">Checking...</span>
              </>
            ) : isHealthy ? (
              <>
                <Activity className="h-3 w-3" />
                <span className="text-xs">Connected</span>
              </>
            ) : (
              <>
                <AlertCircle className="h-3 w-3" />
                <span className="text-xs">Offline</span>
              </>
            )}
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          {isLoading ? (
            <p>Checking backend connection...</p>
          ) : isHealthy ? (
            <div className="space-y-1">
              <p className="font-medium">DataFoundry Backend</p>
              <p className="text-xs text-muted-foreground">Version: {data?.version}</p>
              <div className="pt-1 text-xs">
                <p className="text-muted-foreground">Available features:</p>
                <ul className="list-disc list-inside text-muted-foreground">
                  {data?.features?.data_labeling && <li>Data Labeling</li>}
                  {data?.features?.model_training && <li>Model Training</li>}
                  {data?.features?.data_generation && <li>Data Generation</li>}
                  {data?.features?.anomaly_detection && <li>Anomaly Detection</li>}
                </ul>
              </div>
            </div>
          ) : (
            <div className="space-y-1">
              <p className="font-medium text-destructive">Backend Unavailable</p>
              <p className="text-xs text-muted-foreground">
                Make sure the Flask API is running on port 5000
              </p>
            </div>
          )}
        </TooltipContent>
      </Tooltip>
    );
  }

  // Full variant - shows detailed status card
  return (
    <div className="flex items-center gap-3 p-3 border rounded-lg bg-card">
      <div className="flex-shrink-0">
        {isLoading ? (
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        ) : isHealthy ? (
          <div className="h-3 w-3 rounded-full bg-green-500 animate-pulse" />
        ) : (
          <AlertCircle className="h-5 w-5 text-destructive" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <p className="text-sm font-medium">
            {isLoading ? 'Checking backend...' : isHealthy ? 'Backend Connected' : 'Backend Unavailable'}
          </p>
          {isHealthy && data?.version && (
            <span className="text-xs text-muted-foreground">v{data.version}</span>
          )}
        </div>
        {isHealthy && data?.features && (
          <p className="text-xs text-muted-foreground mt-1">
            {Object.values(data.features).filter(Boolean).length} features available
          </p>
        )}
        {isError && (
          <p className="text-xs text-destructive mt-1">
            Unable to connect to http://localhost:5000
          </p>
        )}
      </div>
    </div>
  );
}

