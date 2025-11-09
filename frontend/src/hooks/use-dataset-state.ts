/**
 * Hook to manage the current dataset state across pages
 */

import { useState, useEffect } from 'react';
import { DatasetInfoResponse } from '@/lib/api';

export interface DatasetState {
  path: string | null;
  info: DatasetInfoResponse | null;
}

export function useDatasetState() {
  const [datasetState, setDatasetState] = useState<DatasetState>({
    path: null,
    info: null,
  });

  // Load from sessionStorage on mount
  useEffect(() => {
    const storedPath = sessionStorage.getItem('currentDatasetPath');
    const storedInfo = sessionStorage.getItem('currentDatasetInfo');

    if (storedPath && storedInfo) {
      try {
        setDatasetState({
          path: storedPath,
          info: JSON.parse(storedInfo),
        });
      } catch (error) {
        console.error('Failed to parse stored dataset info:', error);
      }
    }
  }, []);

  const setDataset = (path: string, info: DatasetInfoResponse) => {
    sessionStorage.setItem('currentDatasetPath', path);
    sessionStorage.setItem('currentDatasetInfo', JSON.stringify(info));
    setDatasetState({ path, info });
  };

  const clearDataset = () => {
    sessionStorage.removeItem('currentDatasetPath');
    sessionStorage.removeItem('currentDatasetInfo');
    setDatasetState({ path: null, info: null });
  };

  return {
    ...datasetState,
    setDataset,
    clearDataset,
    hasDataset: !!datasetState.path && !!datasetState.info,
  };
}

