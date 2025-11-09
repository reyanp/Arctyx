import { useState, useEffect } from "react";
import { CardModern, CardModernHeader, CardModernTitle, CardModernContent } from "@/components/ui/card-modern";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { getFileDownloadUrl } from "@/lib/api";
import { 
  Download, 
  FileSpreadsheet, 
  FileCode, 
  Settings,
  AlertCircle,
  Package
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

  const handleDownload = (filePath: string) => {
    const downloadUrl = getFileDownloadUrl(filePath);
    window.open(downloadUrl, '_blank');
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

          {/* Original Data Reference */}
          <div className="flex items-center justify-between p-4 border rounded-lg bg-muted/30">
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <div className="p-2 rounded-md bg-muted flex-shrink-0">
                <FileCode className="w-5 h-5 text-muted-foreground" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="font-medium text-muted-foreground">Original Dataset</p>
                <p className="text-sm text-muted-foreground font-mono truncate">
                  {getFileName(generationData.originalDataPath)}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Source data used for training
                </p>
              </div>
            </div>
            <Button 
              variant="ghost" 
              size="sm"
              onClick={() => handleDownload(generationData.originalDataPath)}
              className="flex-shrink-0"
            >
              <Download className="w-4 h-4 mr-2" />
              Download
            </Button>
          </div>
        </div>

        {/* Tips */}
        <div className="mt-6 p-4 bg-muted/20 rounded-lg">
          <h4 className="text-sm font-medium mb-2">💡 Export Tips</h4>
          <ul className="text-xs text-muted-foreground space-y-1">
            <li>• Parquet files can be opened with pandas, Excel, or BI tools</li>
            <li>• Model weights can be reused to generate more samples</li>
            <li>• Configuration file contains all training parameters</li>
          </ul>
        </div>
      </CardModernContent>
    </CardModern>
  );
}
