import { CardModern, CardModernHeader, CardModernTitle, CardModernContent } from "@/components/ui/card-modern";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useDatasetState } from "@/hooks/use-dataset-state";
import { Database, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";

export function SchemaCard() {
  const { info, path, hasDataset } = useDatasetState();
  const navigate = useNavigate();

  // If no dataset loaded, show message
  if (!hasDataset || !info) {
    return (
      <CardModern>
        <CardModernHeader>
          <CardModernTitle>Dataset Schema</CardModernTitle>
        </CardModernHeader>
        <CardModernContent>
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              No dataset loaded. Please select a dataset first.
            </AlertDescription>
          </Alert>
          <Button 
            onClick={() => navigate('/')} 
            className="w-full mt-4"
            variant="outline"
          >
            Go to Dataset Selection
          </Button>
        </CardModernContent>
      </CardModern>
    );
  }

  return (
    <CardModern>
      <CardModernHeader>
        <div className="flex items-center justify-between">
          <CardModernTitle>Dataset Schema</CardModernTitle>
          <Badge variant="outline" className="bg-muted">
            {info.num_columns} columns
          </Badge>
        </div>
      </CardModernHeader>

      <CardModernContent>
        {/* Dataset Info */}
        <div className="mb-6 p-3 bg-muted/50 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <Database className="h-4 w-4 text-muted-foreground" />
            <p className="text-xs font-medium text-muted-foreground">Dataset Path</p>
          </div>
          <p className="text-xs text-foreground font-mono truncate">{path}</p>
          <div className="flex gap-4 mt-2 text-xs text-muted-foreground">
            <span>{info.num_rows.toLocaleString()} rows</span>
            <span>•</span>
            <span>{info.num_columns} columns</span>
          </div>
        </div>

        {/* Columns List */}
        <div className="space-y-3 mb-6">
          <h4 className="text-sm font-medium">Columns</h4>
          <div className="max-h-[300px] overflow-y-auto space-y-2">
            {info.columns.map((col) => {
              const dtype = info.dtypes[col] || 'unknown';
              const missingCount = info.missing_values?.[col] || 0;
              const hasMissing = missingCount > 0;

              return (
                <div
                  key={col}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <span className="font-medium text-sm truncate flex-1">{col}</span>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-xs">
                      {dtype}
                    </Badge>
                    {hasMissing && (
                      <Badge variant="outline" className="text-xs text-orange-600">
                        {missingCount} missing
                      </Badge>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Data Quality Summary */}
        {info.missing_values && Object.values(info.missing_values).some(v => v > 0) && (
          <div className="border-t border-ash pt-6">
            <h3 className="text-sm font-medium mb-4">Data Quality</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Total Cells</span>
                <span className="font-medium">
                  {(info.num_rows * info.num_columns).toLocaleString()}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Missing Values</span>
                <span className="font-medium text-orange-600">
                  {Object.values(info.missing_values).reduce((a, b) => a + b, 0).toLocaleString()}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Completeness</span>
                <span className="font-medium">
                  {((1 - Object.values(info.missing_values).reduce((a, b) => a + b, 0) / (info.num_rows * info.num_columns)) * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </CardModernContent>
    </CardModern>
  );
}
