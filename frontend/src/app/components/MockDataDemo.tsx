import React, { useState } from 'react';
import { useTradingStore } from '../store/tradingStore';
import { 
  emergencyHalt, 
  tradingControl, 
  forceReconnect, 
  getQuickChart, 
  searchSymbols, 
  getWatchlist 
} from '../../services/api';

/**
 * Demo component to test mock fallback functionality
 * This component demonstrates how the system gracefully handles service outages
 * by providing mock data when backend services are unavailable.
 */
const MockDataDemo: React.FC = () => {
  const [testResults, setTestResults] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  
  const { 
    emergencyHalt: storeEmergencyHalt,
    pauseTrading: storePauseTrading,
    resumeTrading: storeResumeTrading,
    forceReconnect: storeForceReconnect,
    fetchRegimeData,
    fetchGraphFeatures,
    fetchRLState,
    checkHealth
  } = useTradingStore();

  const addResult = (message: string) => {
    setTestResults(prev => [`${new Date().toLocaleTimeString()}: ${message}`, ...prev.slice(0, 9)]);
  };

  const testEmergencyControls = async () => {
    setIsLoading(true);
    addResult('Testing emergency controls...');
    
    try {
      // Test emergency halt
      await storeEmergencyHalt();
      addResult('✅ Emergency halt executed (with mock fallback)');
      
      // Test pause trading
      await storePauseTrading();
      addResult('✅ Trading paused (with mock fallback)');
      
      // Test resume trading
      await storeResumeTrading();
      addResult('✅ Trading resumed (with mock fallback)');
      
      // Test force reconnect
      await storeForceReconnect();
      addResult('✅ Force reconnect executed (with mock fallback)');
      
    } catch (error) {
      addResult(`❌ Emergency controls test failed: ${error}`);
    }
    setIsLoading(false);
  };

  const testQuickActions = async () => {
    setIsLoading(true);
    addResult('Testing quick actions...');
    
    try {
      // Test quick chart
      const chartResult = await getQuickChart({ symbol: 'EURUSD', timeframe: '1H' });
      addResult(`✅ Quick chart: ${chartResult.message}`);
      
      // Test symbol search
      const searchResult = await searchSymbols({ query: 'EUR', limit: 5 });
      addResult(`✅ Symbol search: Found ${searchResult.total_found} results`);
      
      // Test watchlist
      const watchlistResult = await getWatchlist();
      addResult(`✅ Watchlist: ${watchlistResult.items.length} items loaded`);
      
    } catch (error) {
      addResult(`❌ Quick actions test failed: ${error}`);
    }
    setIsLoading(false);
  };

  const testIntelligenceData = async () => {
    setIsLoading(true);
    addResult('Testing intelligence data...');
    
    try {
      // Test regime data
      await fetchRegimeData('EURUSD');
      addResult('✅ Regime data fetched (with mock fallback)');
      
      // Test graph features
      await fetchGraphFeatures('EURUSD');
      addResult('✅ Graph features fetched (with mock fallback)');
      
      // Test RL state
      await fetchRLState(['EURUSD', 'GBPUSD'], ['momentum_alpha']);
      addResult('✅ RL state assembled (with mock fallback)');
      
      // Test health check
      await checkHealth();
      addResult('✅ Health check completed (with mock fallback)');
      
    } catch (error) {
      addResult(`❌ Intelligence data test failed: ${error}`);
    }
    setIsLoading(false);
  };

  const clearResults = () => {
    setTestResults([]);
  };

  return (
    <div className="p-6 bg-gray-900 text-white min-h-screen">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold text-orange-400 mb-6">
          Mock Data Fallback Demo
        </h1>
        
        <div className="mb-6 p-4 bg-gray-800 rounded border border-gray-600">
          <h2 className="text-lg font-semibold mb-2">About This Demo</h2>
          <p className="text-gray-300 text-sm">
            This demo tests the system's resilience when backend services are unavailable. 
            All functions will gracefully fall back to mock data, ensuring the UI remains 
            functional even during service outages. Check the browser console to see the 
            fallback warnings.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <button
            onClick={testEmergencyControls}
            disabled={isLoading}
            className="p-4 bg-red-900 hover:bg-red-800 disabled:opacity-50 border border-red-600 text-red-400 transition-colors"
          >
            Test Emergency Controls
          </button>
          
          <button
            onClick={testQuickActions}
            disabled={isLoading}
            className="p-4 bg-blue-900 hover:bg-blue-800 disabled:opacity-50 border border-blue-600 text-blue-400 transition-colors"
          >
            Test Quick Actions
          </button>
          
          <button
            onClick={testIntelligenceData}
            disabled={isLoading}
            className="p-4 bg-green-900 hover:bg-green-800 disabled:opacity-50 border border-green-600 text-green-400 transition-colors"
          >
            Test Intelligence Data
          </button>
        </div>

        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold">Test Results</h2>
          <button
            onClick={clearResults}
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 border border-gray-500 text-gray-300 text-sm transition-colors"
          >
            Clear Results
          </button>
        </div>

        <div className="bg-black border border-gray-600 rounded p-4 h-96 overflow-y-auto font-mono text-sm">
          {testResults.length === 0 ? (
            <div className="text-gray-500 italic">
              Click the test buttons above to see mock fallback functionality in action...
            </div>
          ) : (
            testResults.map((result, index) => (
              <div key={index} className="mb-1">
                {result}
              </div>
            ))
          )}
        </div>

        {isLoading && (
          <div className="mt-4 text-center">
            <div className="inline-flex items-center text-orange-400">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-orange-400 mr-2"></div>
              Running tests...
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MockDataDemo;