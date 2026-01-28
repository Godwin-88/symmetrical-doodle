import React, { useState } from 'react';
import { X, Cloud, Key, Shield, CheckCircle, AlertCircle, ExternalLink } from 'lucide-react';
import { DataSourceType, getSourceName } from '../../services/multiSourceService';

interface ConnectionModalProps {
  sourceType: DataSourceType;
  isOpen: boolean;
  onClose: () => void;
  onConnect: (credentials?: any) => Promise<void>;
  isLoading: boolean;
}

export function ConnectionModal({ sourceType, isOpen, onClose, onConnect, isLoading }: ConnectionModalProps) {
  const [credentials, setCredentials] = useState<any>({});
  const [step, setStep] = useState<'auth' | 'permissions' | 'success'>('auth');

  if (!isOpen) return null;

  const handleConnect = async () => {
    try {
      await onConnect(credentials);
      setStep('success');
      setTimeout(() => {
        onClose();
        setStep('auth');
      }, 2000);
    } catch (error) {
      console.error('Connection failed:', error);
    }
  };

  const renderGoogleDriveAuth = () => (
    <div className="space-y-4">
      <div className="text-center">
        <Cloud className="w-12 h-12 mx-auto mb-4 text-blue-400" />
        <h3 className="text-lg font-medium text-white mb-2">Connect to Google Drive</h3>
        <p className="text-sm text-gray-400 mb-4">
          Access your Google Drive files for knowledge ingestion
        </p>
      </div>

      {step === 'auth' && (
        <div className="space-y-4">
          <div className="bg-[#2a2a2a] p-4 rounded border border-[#444]">
            <div className="flex items-start space-x-3">
              <Shield className="w-5 h-5 text-green-400 mt-0.5" />
              <div>
                <h4 className="text-sm font-medium text-white mb-1">Secure Authentication</h4>
                <p className="text-xs text-gray-400">
                  We use OAuth2 to securely connect to your Google Drive. We only request read-only access to your files.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-[#2a2a2a] p-4 rounded border border-[#444]">
            <h4 className="text-sm font-medium text-white mb-2">Permissions Requested:</h4>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• View and download your Google Drive files</li>
              <li>• Access file metadata (name, size, modified date)</li>
              <li>• Browse folder structure</li>
            </ul>
          </div>

          <div className="bg-yellow-900/20 border border-yellow-600/30 p-3 rounded">
            <div className="flex items-start space-x-2">
              <AlertCircle className="w-4 h-4 text-yellow-400 mt-0.5" />
              <div className="text-xs text-yellow-200">
                <strong>Privacy Notice:</strong> Your files are processed locally and only metadata is stored. 
                We never store the actual content of your documents on our servers.
              </div>
            </div>
          </div>

          <button
            onClick={handleConnect}
            disabled={isLoading}
            className="w-full px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center space-x-2"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Connecting...</span>
              </>
            ) : (
              <>
                <ExternalLink className="w-4 h-4" />
                <span>Connect with Google</span>
              </>
            )}
          </button>
        </div>
      )}

      {step === 'success' && (
        <div className="text-center py-8">
          <CheckCircle className="w-16 h-16 mx-auto mb-4 text-green-400" />
          <h3 className="text-lg font-medium text-white mb-2">Successfully Connected!</h3>
          <p className="text-sm text-gray-400">
            Your Google Drive is now connected and ready for use.
          </p>
        </div>
      )}
    </div>
  );

  const renderCloudStorageAuth = () => (
    <div className="space-y-4">
      <div className="text-center">
        <Cloud className="w-12 h-12 mx-auto mb-4 text-blue-400" />
        <h3 className="text-lg font-medium text-white mb-2">Connect to {getSourceName(sourceType)}</h3>
        <p className="text-sm text-gray-400 mb-4">
          Enter your cloud storage credentials
        </p>
      </div>

      <div className="space-y-3">
        {sourceType === DataSourceType.AWS_S3 && (
          <>
            <div>
              <label className="block text-white text-sm mb-1">Access Key ID</label>
              <input
                type="text"
                value={credentials.accessKeyId || ''}
                onChange={(e) => setCredentials(prev => ({ ...prev, accessKeyId: e.target.value }))}
                className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded"
                placeholder="AKIA..."
              />
            </div>
            <div>
              <label className="block text-white text-sm mb-1">Secret Access Key</label>
              <input
                type="password"
                value={credentials.secretAccessKey || ''}
                onChange={(e) => setCredentials(prev => ({ ...prev, secretAccessKey: e.target.value }))}
                className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded"
                placeholder="••••••••"
              />
            </div>
            <div>
              <label className="block text-white text-sm mb-1">Region</label>
              <select
                value={credentials.region || 'us-east-1'}
                onChange={(e) => setCredentials(prev => ({ ...prev, region: e.target.value }))}
                className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded"
              >
                <option value="us-east-1">US East (N. Virginia)</option>
                <option value="us-west-2">US West (Oregon)</option>
                <option value="eu-west-1">Europe (Ireland)</option>
                <option value="ap-southeast-1">Asia Pacific (Singapore)</option>
              </select>
            </div>
            <div>
              <label className="block text-white text-sm mb-1">Bucket Name</label>
              <input
                type="text"
                value={credentials.bucketName || ''}
                onChange={(e) => setCredentials(prev => ({ ...prev, bucketName: e.target.value }))}
                className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded"
                placeholder="my-documents-bucket"
              />
            </div>
          </>
        )}

        {sourceType === DataSourceType.AZURE_BLOB && (
          <>
            <div>
              <label className="block text-white text-sm mb-1">Connection String</label>
              <textarea
                value={credentials.connectionString || ''}
                onChange={(e) => setCredentials(prev => ({ ...prev, connectionString: e.target.value }))}
                className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded h-20 resize-none"
                placeholder="DefaultEndpointsProtocol=https;AccountName=..."
              />
            </div>
            <div>
              <label className="block text-white text-sm mb-1">Container Name</label>
              <input
                type="text"
                value={credentials.containerName || ''}
                onChange={(e) => setCredentials(prev => ({ ...prev, containerName: e.target.value }))}
                className="w-full bg-[#2a2a2a] border border-[#444] text-white text-sm px-3 py-2 rounded"
                placeholder="documents"
              />
            </div>
          </>
        )}

        <div className="bg-[#2a2a2a] p-3 rounded border border-[#444]">
          <div className="flex items-start space-x-2">
            <Key className="w-4 h-4 text-yellow-400 mt-0.5" />
            <div className="text-xs text-gray-400">
              <strong>Security:</strong> Credentials are encrypted and stored securely. 
              We recommend using read-only access keys when possible.
            </div>
          </div>
        </div>

        <button
          onClick={handleConnect}
          disabled={isLoading || !credentials.accessKeyId || !credentials.secretAccessKey}
          className="w-full px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {isLoading ? 'Testing Connection...' : 'Connect'}
        </button>
      </div>
    </div>
  );

  const renderLocalSourceInfo = () => (
    <div className="space-y-4">
      <div className="text-center">
        <div className="w-12 h-12 mx-auto mb-4 text-green-400">
          {sourceType === DataSourceType.LOCAL_ZIP ? '📦' : 
           sourceType === DataSourceType.LOCAL_DIRECTORY ? '📁' : '📤'}
        </div>
        <h3 className="text-lg font-medium text-white mb-2">{getSourceName(sourceType)}</h3>
        <p className="text-sm text-gray-400 mb-4">
          {sourceType === DataSourceType.LOCAL_ZIP && 'Upload and process ZIP archives containing PDF files'}
          {sourceType === DataSourceType.LOCAL_DIRECTORY && 'Browse and select files from your local directories'}
          {sourceType === DataSourceType.INDIVIDUAL_UPLOAD && 'Upload individual PDF files for processing'}
        </p>
      </div>

      <div className="bg-[#2a2a2a] p-4 rounded border border-[#444]">
        <h4 className="text-sm font-medium text-white mb-2">Supported Features:</h4>
        <ul className="text-xs text-gray-400 space-y-1">
          <li>• PDF file processing and indexing</li>
          <li>• Batch processing capabilities</li>
          <li>• Metadata extraction and tagging</li>
          <li>• Semantic chunking and embedding generation</li>
        </ul>
      </div>

      <div className="bg-green-900/20 border border-green-600/30 p-3 rounded">
        <div className="flex items-start space-x-2">
          <CheckCircle className="w-4 h-4 text-green-400 mt-0.5" />
          <div className="text-xs text-green-200">
            <strong>Ready to Use:</strong> This data source is automatically available and requires no additional setup.
          </div>
        </div>
      </div>

      <button
        onClick={onClose}
        className="w-full px-4 py-2 bg-green-600 text-white text-sm rounded hover:bg-green-700"
      >
        Got it
      </button>
    </div>
  );

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-[#1a1a1a] border border-[#444] rounded-lg p-6 w-96 max-w-[90vw] max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-[#ff8c00] text-lg font-medium">Data Source Connection</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {sourceType === DataSourceType.GOOGLE_DRIVE && renderGoogleDriveAuth()}
        {(sourceType === DataSourceType.AWS_S3 || sourceType === DataSourceType.AZURE_BLOB) && renderCloudStorageAuth()}
        {(sourceType === DataSourceType.LOCAL_ZIP || 
          sourceType === DataSourceType.LOCAL_DIRECTORY || 
          sourceType === DataSourceType.INDIVIDUAL_UPLOAD) && renderLocalSourceInfo()}
      </div>
    </div>
  );
}