import { useState, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import { CardModern, CardModernHeader, CardModernTitle, CardModernContent } from "@/components/ui/card-modern";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { getFileDownloadUrl, convertToCsv } from "@/lib/api";
import { 
  Download, 
  FileSpreadsheet, 
  Settings,
  AlertCircle,
  Package,
  Loader2
} from "lucide-react";

interface GenerationData {
  syntheticDataPath: string;
  originalDataPath: string;
  numSamples?: number;
  modelPath: string;
  configPath: string;
}

export function ExportSection() {
  const [generationData, setGenerationData] = useState<GenerationData | null>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem('latestGeneration');
    if (stored) {
      try {
        setGenerationData(JSON.parse(stored));
      } catch (error) {
        console.error('Failed to parse generation data:', error);
      }
    }
  }, []);

  // CSV conversion mutation
  const csvConversionMutation = useMutation({
    mutationFn: convertToCsv,
    onSuccess: (data) => {
      // Download the converted CSV file
      const downloadUrl = getFileDownloadUrl(data.csv_path);
      window.open(downloadUrl, '_blank');
    },
  });

  const handleDownload = (filePath: string) => {
    const downloadUrl = getFileDownloadUrl(filePath);
    window.open(downloadUrl, '_blank');
  };

  const handleConvertToCsv = () => {
    if (generationData?.syntheticDataPath) {
      csvConversionMutation.mutate({
        parquet_path: generationData.syntheticDataPath,
      });
    }
  };

  if (!generationData) {
    return (
      <CardModern>
        <CardModernHeader>
          <CardModernTitle>Export Results</CardModernTitle>
        </CardModernHeader>
        <CardModernContent>
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              No files available for export. Generate synthetic data first.
            </AlertDescription>
          </Alert>
        </CardModernContent>
      </CardModern>
    );
  }

  const getFileName = (path: string) => {
    return path.split('/').pop() || path;
  };

  return (
    <CardModern>
      <CardModernHeader>
        <CardModernTitle>Export Files</CardModernTitle>
      </CardModernHeader>
      <CardModernContent>
        <div className="space-y-4">
          {/* Synthetic Data */}
          <div className="flex items-center justify-between p-4 border rounded-lg bg-card hover:bg-muted/20 transition-colors">
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <div className="p-2 rounded-md bg-primary/10 flex-shrink-0">
                <FileSpreadsheet className="w-5 h-5 text-primary" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
              <p className="font-medium">Synthetic Dataset</p>
                  <Badge variant="default" className="text-xs">Primary</Badge>
                </div>
                <p className="text-sm text-muted-foreground font-mono truncate">
                  {getFileName(generationData.syntheticDataPath)}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Parquet format
                  {generationData.numSamples && ` • ${generationData.numSamples.toLocaleString()} samples`}
                </p>
              </div>
            </div>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => handleDownload(generationData.syntheticDataPath)}
              className="flex-shrink-0"
            >
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
        </div>

          {/* Model Weights */}
          <div className="flex items-center justify-between p-4 border rounded-lg bg-card hover:bg-muted/20 transition-colors">
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <div className="p-2 rounded-md bg-purple-500/10 flex-shrink-0">
                <Package className="w-5 h-5 text-purple-500" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="font-medium">Trained Model</p>
                <p className="text-sm text-muted-foreground font-mono truncate">
                  {getFileName(generationData.modelPath)}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  PyTorch model weights • Reusable for generation
                </p>
              </div>
            </div>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => handleDownload(generationData.modelPath)}
              className="flex-shrink-0"
            >
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
        </div>

          {/* Config File */}
          <div className="flex items-center justify-between p-4 border rounded-lg bg-card hover:bg-muted/20 transition-colors">
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <div className="p-2 rounded-md bg-blue-500/10 flex-shrink-0">
                <Settings className="w-5 h-5 text-blue-500" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="font-medium">Configuration</p>
                <p className="text-sm text-muted-foreground font-mono truncate">
                  {getFileName(generationData.configPath)}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  JSON config • Model architecture & parameters
                </p>
              </div>
            </div>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => handleDownload(generationData.configPath)}
              className="flex-shrink-0"
            >
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
        </div>

          {/* CSV Export */}
          <div className="flex items-center justify-between p-4 border rounded-lg bg-green-50 dark:bg-green-950/20 border-green-200 dark:border-green-900">
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <div className="p-2 rounded-md bg-green-500/10 flex-shrink-0">
                <FileSpreadsheet className="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="font-medium text-green-900 dark:text-green-100">Export as CSV</p>
                <p className="text-sm text-green-700 dark:text-green-300 truncate">
                  Convert synthetic data to CSV format
                </p>
                <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                  Universal format • Compatible with Excel & spreadsheets
                </p>
              </div>
            </div>
            <Button 
              variant="outline" 
              size="sm"
              onClick={handleConvertToCsv}
              disabled={csvConversionMutation.isPending}
              className="flex-shrink-0 border-green-300 hover:bg-green-100 dark:hover:bg-green-900/50"
            >
              {csvConversionMutation.isPending ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Converting...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4 mr-2" />
                  Download CSV
                </>
              )}
          </Button>
          </div>

          {/* CSV Conversion Error */}
          {csvConversionMutation.isError && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                {csvConversionMutation.error instanceof Error 
                  ? csvConversionMutation.error.message 
                  : 'Failed to convert file to CSV'}
              </AlertDescription>
            </Alert>
          )}
        </div>

        {/* Tips */}
        <div className="mt-6 p-4 bg-muted/20 rounded-lg">
          <h4 className="text-sm font-medium mb-2">💡 Export Tips</h4>
          <ul className="text-xs text-muted-foreground space-y-1">
            <li>• Parquet files are more efficient for large datasets</li>
            <li>• CSV files are universally compatible with Excel and spreadsheets</li>
            <li>• Model weights can be reused to generate more samples</li>
            <li>• Configuration file contains all training parameters</li>
          </ul>
        </div>
      </CardModernContent>
    </CardModern>
  );
}
