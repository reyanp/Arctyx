/**
 * Dataset Selector Component
 * 
 * Allows users to specify a dataset path and preview its structure
 * Connects to the backend /api/dataset/info endpoint
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { getDatasetInfo, DatasetInfoResponse } from "@/lib/api";
import { CardModern } from "@/components/ui/card-modern";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { 
  FileSpreadsheet, 
  Search, 
  CheckCircle2, 
  AlertCircle,
  ChevronRight,
  Table,
  Columns,
  Database
} from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

// Common dataset paths for quick selection
// Note: Paths are relative to the backend directory where Flask is running
const COMMON_PATHS = [
  "testing_data/adult.csv",
];

export function DatasetSelector() {
  const navigate = useNavigate();
  const [dataPath, setDataPath] = useState("testing_data/adult.csv");
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfoResponse | null>(null);

  // Mutation for fetching dataset info
  const datasetInfoMutation = useMutation({
    mutationFn: (path: string) => getDatasetInfo(path),
    onSuccess: (data) => {
      setDatasetInfo(data);
      // Store dataset path in sessionStorage for use in other pages
      sessionStorage.setItem('currentDatasetPath', dataPath);
      sessionStorage.setItem('currentDatasetInfo', JSON.stringify(data));
    },
  });

  const handleLoadDataset = () => {
    if (dataPath.trim()) {
      datasetInfoMutation.mutate(dataPath.trim());
    }
  };

  const handleContinue = () => {
    if (datasetInfo) {
      navigate("/schema");
    }
  };

  const isLoading = datasetInfoMutation.isPending;
  const isError = datasetInfoMutation.isError;
  const isSuccess = datasetInfoMutation.isSuccess;

  return (
    <CardModern>
      <div className="space-y-6">
        {/* Path Input Section */}
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="dataset-path" className="text-sm font-medium">
              Dataset Path
            </Label>
            <div className="flex gap-2">
              <Input
                id="dataset-path"
                placeholder="testing_data/adult.csv"
                value={dataPath}
                onChange={(e) => setDataPath(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !isLoading) {
                    handleLoadDataset();
                  }
                }}
                disabled={isLoading}
                className="flex-1"
              />
              <Button
                onClick={handleLoadDataset}
                disabled={isLoading || !dataPath.trim()}
                size="default"
              >
                {isLoading ? (
                  <>
                    <Search className="h-4 w-4 mr-2 animate-spin" />
                    Loading...
                  </>
                ) : (
                  <>
                    <Search className="h-4 w-4 mr-2" />
                    Load
                  </>
                )}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Path relative to backend directory (e.g., "testing_data/adult.csv")
            </p>
          </div>

          {/* Quick Select Buttons */}
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground">Quick Select:</p>
            <div className="flex flex-wrap gap-2">
              {COMMON_PATHS.map((path) => (
                <Button
                  key={path}
                  variant="outline"
                  size="sm"
                  onClick={() => setDataPath(path)}
                  disabled={isLoading}
                  className="text-xs"
                >
                  {path.split('/').pop()}
                </Button>
              ))}
            </div>
          </div>
        </div>

        {/* Error State */}
        {isError && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {datasetInfoMutation.error?.message || "Failed to load dataset. Check the path and ensure the backend is running."}
            </AlertDescription>
          </Alert>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="space-y-3 p-4 border rounded-lg bg-muted/20">
            <div className="flex items-center gap-2">
              <Skeleton className="h-4 w-4 rounded-full" />
              <Skeleton className="h-4 w-32" />
            </div>
            <Skeleton className="h-20 w-full" />
          </div>
        )}

        {/* Success State - Dataset Preview */}
        {isSuccess && datasetInfo && (
          <div className="space-y-4">
            {/* Success Banner */}
            <Alert className="border-green-500/50 bg-green-500/10">
              <CheckCircle2 className="h-4 w-4 text-green-500" />
              <AlertDescription className="text-green-700 dark:text-green-400">
                Dataset loaded successfully!
              </AlertDescription>
            </Alert>

            {/* Dataset Stats */}
            <div className="grid grid-cols-3 gap-4">
              <div className="flex items-center gap-3 p-3 border rounded-lg bg-card">
                <div className="p-2 rounded-md bg-primary/10">
                  <Table className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Rows</p>
                  <p className="text-lg font-semibold">{datasetInfo.num_rows.toLocaleString()}</p>
                </div>
              </div>
              
              <div className="flex items-center gap-3 p-3 border rounded-lg bg-card">
                <div className="p-2 rounded-md bg-primary/10">
                  <Columns className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Columns</p>
                  <p className="text-lg font-semibold">{datasetInfo.num_columns}</p>
                </div>
              </div>

              <div className="flex items-center gap-3 p-3 border rounded-lg bg-card">
                <div className="p-2 rounded-md bg-primary/10">
                  <Database className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Size</p>
                  <p className="text-lg font-semibold">
                    {(datasetInfo.num_rows * datasetInfo.num_columns / 1000).toFixed(1)}K
                  </p>
                </div>
              </div>
            </div>

            {/* Column Preview */}
            <div className="border rounded-lg p-4 bg-card">
              <div className="flex items-center gap-2 mb-3">
                <FileSpreadsheet className="h-4 w-4 text-muted-foreground" />
                <h4 className="text-sm font-medium">Columns</h4>
              </div>
              <div className="flex flex-wrap gap-2">
                {datasetInfo.columns.slice(0, 12).map((col) => (
                  <Badge key={col} variant="secondary" className="text-xs">
                    {col}
                  </Badge>
                ))}
                {datasetInfo.columns.length > 12 && (
                  <Badge variant="outline" className="text-xs">
                    +{datasetInfo.columns.length - 12} more
                  </Badge>
                )}
              </div>
            </div>

            {/* Sample Data Preview */}
            {datasetInfo.sample_data && datasetInfo.sample_data.length > 0 && (
              <div className="border rounded-lg p-4 bg-card">
                <h4 className="text-sm font-medium mb-3">Sample Data (First 3 Rows)</h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b">
                        {datasetInfo.columns.slice(0, 6).map((col) => (
                          <th key={col} className="text-left p-2 font-medium text-muted-foreground">
                            {col}
                          </th>
                        ))}
                        {datasetInfo.columns.length > 6 && (
                          <th className="text-left p-2 font-medium text-muted-foreground">...</th>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {datasetInfo.sample_data.slice(0, 3).map((row, idx) => (
                        <tr key={idx} className="border-b last:border-0">
                          {datasetInfo.columns.slice(0, 6).map((col) => (
                            <td key={col} className="p-2 text-foreground/80">
                              {String(row[col] ?? '-')}
                            </td>
                          ))}
                          {datasetInfo.columns.length > 6 && (
                            <td className="p-2 text-muted-foreground">...</td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Continue Button */}
            <Button
              onClick={handleContinue}
              className="w-full"
              size="lg"
            >
              Continue to Configuration
              <ChevronRight className="h-4 w-4 ml-2" />
            </Button>
          </div>
        )}
      </div>
    </CardModern>
  );
}

