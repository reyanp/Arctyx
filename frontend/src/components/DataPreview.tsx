/**
 * Data Preview Component
 * 
 * Fetches and displays a preview of the synthetic data from the backend
 */

import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getDatasetInfo, type DatasetInfoResponse } from "@/lib/api";
import { CardModern, CardModernHeader, CardModernTitle, CardModernContent } from "@/components/ui/card-modern";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { 
  AlertCircle, 
  FileSpreadsheet,
  Eye,
  RefreshCw
} from "lucide-react";
import { Button } from "@/components/ui/button";

export function DataPreview() {
  const [syntheticDataPath, setSyntheticDataPath] = useState<string | null>(null);

  useEffect(() => {
    // Load the latest generation results
    const stored = sessionStorage.getItem('latestGeneration');
    if (stored) {
      try {
        const data = JSON.parse(stored);
        setSyntheticDataPath(data.syntheticDataPath);
      } catch (error) {
        console.error('Failed to parse generation data:', error);
      }
    }
  }, []);

  // Fetch dataset info (includes sample data)
  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['synthetic-preview', syntheticDataPath],
    queryFn: () => syntheticDataPath ? getDatasetInfo(syntheticDataPath) : Promise.reject('No data path'),
    enabled: !!syntheticDataPath,
    retry: 1,
  });

  if (!syntheticDataPath) {
    return (
      <CardModern>
        <CardModernHeader>
          <CardModernTitle>Data Preview</CardModernTitle>
        </CardModernHeader>
        <CardModernContent>
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              No synthetic data generated yet. Run the workflow first to see a preview here.
            </AlertDescription>
          </Alert>
        </CardModernContent>
      </CardModern>
    );
  }

  if (isLoading) {
    return (
      <CardModern>
        <CardModernHeader>
          <CardModernTitle>Data Preview</CardModernTitle>
        </CardModernHeader>
        <CardModernContent>
          <div className="space-y-3">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
          </div>
        </CardModernContent>
      </CardModern>
    );
  }

  if (isError) {
    return (
      <CardModern>
        <CardModernHeader>
          <CardModernTitle>Data Preview</CardModernTitle>
        </CardModernHeader>
        <CardModernContent>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {error instanceof Error ? error.message : 'Failed to load data preview'}
            </AlertDescription>
          </Alert>
          <Button 
            onClick={() => refetch()} 
            variant="outline" 
            size="sm" 
            className="mt-4"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardModernContent>
      </CardModern>
    );
  }

  if (!data || !data.sample_data || data.sample_data.length === 0) {
    return (
      <CardModern>
        <CardModernHeader>
          <CardModernTitle>Data Preview</CardModernTitle>
        </CardModernHeader>
        <CardModernContent>
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              No sample data available for preview.
            </AlertDescription>
          </Alert>
        </CardModernContent>
      </CardModern>
    );
  }

  // Check if labeling was skipped
  const latestGen = sessionStorage.getItem('latestGeneration');
  const skippedLabeling = latestGen ? JSON.parse(latestGen).skippedLabeling : false;

  // Filter out label columns if labeling was skipped
  const filteredColumns = data.columns.filter(col => {
    if (!skippedLabeling) return true; // Show all columns if labeling wasn't skipped
    // Hide label-related columns when labeling was skipped
    return !col.toLowerCase().includes('label');
  });

  // Get columns to display (limit to first 8 for readability)
  const columnsToShow = filteredColumns.slice(0, 8);
  const hasMoreColumns = filteredColumns.length > 8;

  return (
    <CardModern>
      <CardModernHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Eye className="h-5 w-5 text-muted-foreground" />
            <CardModernTitle>Synthetic Data Preview</CardModernTitle>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline">
              {data.num_rows.toLocaleString()} rows
            </Badge>
            <Badge variant="outline">
              {data.num_columns} columns
            </Badge>
          </div>
        </div>
      </CardModernHeader>
      <CardModernContent>
        <div className="space-y-4">
          {/* File Info */}
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <FileSpreadsheet className="h-3 w-3" />
            <span className="font-mono truncate">{syntheticDataPath}</span>
          </div>

          {/* Data Table */}
          <div className="border rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-12 bg-muted/50">#</TableHead>
                    {columnsToShow.map((col) => (
                      <TableHead key={col} className="bg-muted/50">
                        <div className="flex flex-col gap-1">
                          <span className="font-medium">{col}</span>
                          <span className="text-xs text-muted-foreground font-normal">
                            {data.dtypes[col]}
                          </span>
                        </div>
                      </TableHead>
                    ))}
                    {hasMoreColumns && (
                      <TableHead className="bg-muted/50 text-muted-foreground">
                        +{filteredColumns.length - 8} more
                      </TableHead>
                    )}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.sample_data.slice(0, 5).map((row, idx) => (
                    <TableRow key={idx}>
                      <TableCell className="font-medium text-muted-foreground">
                        {idx + 1}
                      </TableCell>
                      {columnsToShow.map((col) => (
                        <TableCell key={col} className="font-mono text-xs">
                          {String(row[col] ?? '-')}
                        </TableCell>
                      ))}
                      {hasMoreColumns && (
                        <TableCell className="text-muted-foreground text-xs">
                          ...
                        </TableCell>
                      )}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>

          {/* Footer Info */}
          <div className="flex items-center justify-between text-xs text-muted-foreground pt-2">
            <span>Showing first 5 of {data.num_rows.toLocaleString()} rows</span>
            {hasMoreColumns && (
              <span>Showing first 8 of {filteredColumns.length} columns</span>
            )}
          </div>
        </div>
      </CardModernContent>
    </CardModern>
  );
}

