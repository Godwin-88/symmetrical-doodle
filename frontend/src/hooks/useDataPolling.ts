/**
 * Custom hook for polling data from backend
 */

import { useEffect, useRef } from 'react';

interface UseDataPollingOptions {
  enabled?: boolean;
  interval?: number;
  onError?: (error: Error) => void;
}

export function useDataPolling(
  fetchFn: () => Promise<void>,
  options: UseDataPollingOptions = {}
) {
  const { enabled = true, interval = 5000, onError } = options;
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    // Initial fetch
    fetchFn().catch((error) => {
      console.error('Data polling error:', error);
      onError?.(error);
    });

    // Set up polling
    intervalRef.current = setInterval(() => {
      fetchFn().catch((error) => {
        console.error('Data polling error:', error);
        onError?.(error);
      });
    }, interval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [fetchFn, enabled, interval, onError]);
}
