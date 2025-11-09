/**
 * File Upload Component
 * 
 * Allows users to upload CSV/Parquet files via drag & drop or file picker
 */

import { useState, useCallback, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { uploadFile, getDatasetInfo, type UploadFileResponse } from "@/lib/api";
import { useDatasetState } from "@/hooks/use-dataset-state";
import { CardModern, CardModernHeader, CardModernTitle, CardModernContent } from "@/components/ui/card-modern";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { 
  Upload, 
  FileSpreadsheet, 
  CheckCircle, 
  AlertCircle,
  Trash2
} from "lucide-react";

export function FileUpload() {
  const navigate = useNavigate();
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<UploadFileResponse | null>(null);
  const { setDataset, clearDataset, datasetInfo, path } = useDatasetState();

  // Load uploaded file info from sessionStorage on mount
  useEffect(() => {
    if (path && datasetInfo) {
      // Reconstruct uploaded file info from stored path
      const fileName = path.split('/').pop() || '';
      setUploadedFile({
        file_path: path,
        filename: fileName,
        original_filename: fileName,
        size_bytes: 0, // Size not stored, but not critical
      });
    }
  }, [path, datasetInfo]);

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: uploadFile,
    onSuccess: async (data) => {
      setUploadedFile(data);
      
      // Automatically fetch dataset info after upload
      try {
        const info = await getDatasetInfo(data.file_path);
        setDataset(data.file_path, info);
      } catch (error) {
        console.error('Failed to load dataset info:', error);
      }
    },
  });

  // Handle file selection
  const handleFile = useCallback((file: File) => {
    // Validate file type
    if (!file.name.endsWith('.csv') && !file.name.endsWith('.parquet')) {
      alert('Please upload a CSV or Parquet file');
      return;
    }

    // Validate file size (100MB max)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
      alert('File too large. Maximum size is 100MB');
      return;
    }

    uploadMutation.mutate(file);
  }, [uploadMutation]);

  // Drag handlers
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, [handleFile]);

  // File input handler
  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  }, [handleFile]);

  // Clear upload
  const handleClear = useCallback(() => {
    setUploadedFile(null);
    clearDataset();
    uploadMutation.reset();
  }, [clearDataset, uploadMutation]);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="space-y-4">
      {/* Upload Area - Large Version */}
      {!uploadedFile && !uploadMutation.isPending && (
        <div className="bg-white rounded-2xl shadow-lg p-12">
          <div
            className={`
              relative border-[3px] border-dashed rounded-2xl p-16
              transition-all duration-200 ease-in-out
              ${dragActive 
                ? 'border-primary bg-primary/5 scale-[1.01]' 
                : 'border-gray-300 hover:border-primary/50 hover:bg-gray-50'
              }
            `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              id="file-upload"
              className="sr-only"
              accept=".csv,.parquet"
              onChange={handleChange}
            />
            <label
              htmlFor="file-upload"
              className="flex flex-col items-center justify-center cursor-pointer"
            >
              <div className="p-4 rounded-full bg-primary/10 mb-6">
                <Upload className="w-12 h-12 text-primary" />
              </div>
              <h3 className="text-2xl font-semibold mb-3">
                Upload CSV Dataset
              </h3>
              <p className="text-base text-muted-foreground mb-6">
                Drag and drop your file here, or click to browse
              </p>
              <Button type="button" size="lg" variant="default">
                Select File
              </Button>
            </label>
          </div>
        </div>
      )}

      {/* Uploading State */}
      {uploadMutation.isPending && (
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <div className="space-y-3 p-6 border rounded-lg bg-muted/20">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-md bg-primary/10">
                <FileSpreadsheet className="w-5 h-5 text-primary animate-pulse" />
              </div>
              <div className="flex-1">
                <p className="font-medium">Uploading file...</p>
                <p className="text-sm text-muted-foreground">Please wait</p>
              </div>
            </div>
            <Progress value={50} className="h-2" />
          </div>
        </div>
      )}

      {/* Upload Error */}
      {uploadMutation.isError && (
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {uploadMutation.error instanceof Error 
                ? uploadMutation.error.message 
                : 'Failed to upload file'}
            </AlertDescription>
          </Alert>
        </div>
      )}

      {/* Upload Success */}
      {uploadedFile && (
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 border rounded-lg bg-green-50 dark:bg-green-950/20 border-green-200 dark:border-green-900">
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <div className="p-2 rounded-md bg-green-500/10">
                    <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="font-medium text-green-900 dark:text-green-100">
                      {uploadedFile.original_filename}
                    </p>
                    <p className="text-sm text-green-700 dark:text-green-300 font-mono">
                      {formatFileSize(uploadedFile.size_bytes)}
                    </p>
                  </div>
                </div>
                <Button
                  onClick={handleClear}
                  variant="ghost"
                  size="sm"
                  className="flex-shrink-0"
                  title="Remove file"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
            </div>

            {/* Dataset Info */}
            {datasetInfo && (
              <div className="p-4 border rounded-lg bg-card">
                <h4 className="font-medium mb-3 flex items-center gap-2">
                  <FileSpreadsheet className="w-4 h-4" />
                  Dataset Preview
                </h4>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-xs text-muted-foreground">Rows</p>
                    <p className="text-lg font-semibold">
                      {datasetInfo.num_rows.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Columns</p>
                    <p className="text-lg font-semibold">
                      {datasetInfo.num_columns}
                    </p>
                  </div>
                </div>

                {/* Sample Data */}
                {datasetInfo.sample_data && datasetInfo.sample_data.length > 0 && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-2">
                      First {Math.min(3, datasetInfo.sample_data.length)} rows
                    </p>
                    <div className="bg-muted/50 rounded p-3 overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-border">
                            {datasetInfo.columns.slice(0, 5).map((col) => (
                              <th key={col} className="text-left py-1 px-2 font-medium">
                                {col}
                              </th>
                            ))}
                            {datasetInfo.columns.length > 5 && (
                              <th className="text-left py-1 px-2 font-medium text-muted-foreground">
                                +{datasetInfo.columns.length - 5} more
                              </th>
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          {datasetInfo.sample_data.slice(0, 3).map((row, idx) => (
                            <tr key={idx} className="border-b border-border/50">
                              {datasetInfo.columns.slice(0, 5).map((col) => (
                                <td key={col} className="py-1 px-2 font-mono">
                                  {String(row[col] ?? '-')}
                                </td>
                              ))}
                              {datasetInfo.columns.length > 5 && (
                                <td className="py-1 px-2 text-muted-foreground">...</td>
                              )}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Quick Actions */}
            <div className="pt-2">
              <Button
                onClick={() => navigate('/schema')}
                variant="default"
                size="lg"
                className="w-full"
              >
                Upload
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
