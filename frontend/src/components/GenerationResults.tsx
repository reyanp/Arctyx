/**
 * Generation Results Component
 * 
 * Displays the results of the synthetic data generation workflow
 */

import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { CardModern, CardModernHeader, CardModernTitle, CardModernContent } from "@/components/ui/card-modern";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { getFileDownloadUrl } from "@/lib/api";
import { 
  Download, 
  CheckCircle, 
  FileSpreadsheet, 
  Database,
  Brain,
  Settings,
  ArrowLeft,
  ExternalLink
} from "lucide-react";

interface GenerationResultData {
  syntheticDataPath: string;
  originalDataPath: string;
  numSamples?: number; // Optional - may not be present
  modelPath: string;
  configPath: string;
}

export function GenerationResults() {
  const navigate = useNavigate();
  const [resultData, setResultData] = useState<GenerationResultData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Load results from sessionStorage
    const stored = sessionStorage.getItem('latestGeneration');
    console.log('Loading generation results:', stored); // Debug log
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        console.log('Parsed results:', parsed); // Debug log
        setResultData(parsed);
      } catch (error) {
        console.error('Failed to parse generation results:', error);
      }
    }
    setIsLoading(false);
  }, []);

  const handleDownload = (filePath: string) => {
    const downloadUrl = getFileDownloadUrl(filePath);
    window.open(downloadUrl, '_blank');
  };

  const handleRunAnother = () => {
    // Clear results and go back to home
    sessionStorage.removeItem('latestGeneration');
    navigate('/');
  };

  // Show loading state
  if (isLoading) {
    return (
      <CardModern>
        <CardModernContent>
          <div className="flex items-center justify-center py-12">
            <div className="text-center space-y-2">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
              <p className="text-sm text-muted-foreground">Loading results...</p>
            </div>
          </div>
        </CardModernContent>
      </CardModern>
    );
  }

  // Show message if no results
  if (!resultData) {
    return (
      <CardModern>
        <CardModernHeader>
          <CardModernTitle>Generation Results</CardModernTitle>
        </CardModernHeader>
        <CardModernContent>
          <Alert>
            <AlertDescription>
              No generation results found. Run the workflow first to see results here.
            </AlertDescription>
          </Alert>
          <Button 
            onClick={() => navigate('/')} 
            className="w-full mt-4"
            variant="outline"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Home
          </Button>
        </CardModernContent>
      </CardModern>
    );
  }

  return (
    <div className="space-y-6">
      {/* Success Header */}
      <CardModern>
        <CardModernContent>
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-full bg-green-500/10">
              <CheckCircle className="h-8 w-8 text-green-500" />
            </div>
            <div className="flex-1">
              <h2 className="text-2xl font-bold mb-2">Generation Complete!</h2>
              <p className="text-muted-foreground">
                Successfully generated synthetic data
                {resultData.numSamples && ` (${resultData.numSamples.toLocaleString()} samples)`}
              </p>
            </div>
          </div>
        </CardModernContent>
      </CardModern>

      {/* Generation Summary */}
      <CardModern>
        <CardModernHeader>
          <CardModernTitle>Generation Summary</CardModernTitle>
        </CardModernHeader>
        <CardModernContent>
          <div className="space-y-4">
            {/* Samples Generated */}
            {resultData.numSamples && (
              <div className="flex items-center justify-between p-4 bg-muted/20 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-md bg-primary/10">
                    <Database className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm font-medium">Samples Generated</p>
                    <p className="text-xs text-muted-foreground">Total synthetic records created</p>
                  </div>
                </div>
                <Badge variant="default" className="text-lg px-4 py-2">
                  {resultData.numSamples.toLocaleString()}
                </Badge>
              </div>
            )}

            <Separator />

            {/* File Outputs */}
            <div className="space-y-3">
              <h4 className="text-sm font-medium flex items-center gap-2">
                <FileSpreadsheet className="h-4 w-4 text-muted-foreground" />
                Output Files
              </h4>

              {/* Synthetic Data File */}
              <div className="p-3 border rounded-lg bg-card space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium">Synthetic Data</p>
                    <p className="text-xs text-muted-foreground font-mono truncate">
                      {resultData.syntheticDataPath}
                    </p>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleDownload(resultData.syntheticDataPath)}
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download
                  </Button>
                </div>
              </div>

              {/* Original Data File */}
              <div className="p-3 border rounded-lg bg-card">
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-muted-foreground">Original Dataset</p>
                    <p className="text-xs text-muted-foreground font-mono truncate">
                      {resultData.originalDataPath}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <Separator />

            {/* Model Artifacts */}
            <div className="space-y-3">
              <h4 className="text-sm font-medium flex items-center gap-2">
                <Brain className="h-4 w-4 text-muted-foreground" />
                Model Artifacts
              </h4>

              <div className="space-y-2">
                <div className="flex items-center gap-2 p-2 bg-muted/10 rounded text-xs">
                  <Settings className="h-3 w-3 text-muted-foreground" />
                  <span className="font-mono text-muted-foreground truncate">
                    {resultData.modelPath}
                  </span>
                </div>
                <div className="flex items-center gap-2 p-2 bg-muted/10 rounded text-xs">
                  <Settings className="h-3 w-3 text-muted-foreground" />
                  <span className="font-mono text-muted-foreground truncate">
                    {resultData.configPath}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </CardModernContent>
      </CardModern>

      {/* Quick Actions */}
      <CardModern>
        <CardModernHeader>
          <CardModernTitle>Next Steps</CardModernTitle>
        </CardModernHeader>
        <CardModernContent>
          <div className="space-y-3">
            <Button
              className="w-full"
              size="lg"
              variant="outline"
              onClick={() => handleDownload(resultData.syntheticDataPath)}
            >
              <Download className="h-4 w-4 mr-2" />
              Download Synthetic Data
            </Button>

            <Button
              className="w-full"
              size="lg"
              variant="outline"
              onClick={() => navigate('/export')}
            >
              <ExternalLink className="h-4 w-4 mr-2" />
              Export Options
            </Button>

            <Button
              className="w-full"
              size="lg"
              onClick={handleRunAnother}
            >
              Generate Another Dataset
            </Button>
          </div>
        </CardModernContent>
      </CardModern>

      {/* Tips */}
      <CardModern>
        <CardModernHeader>
          <CardModernTitle>💡 Tips</CardModernTitle>
        </CardModernHeader>
        <CardModernContent>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li className="flex gap-2">
              <span>•</span>
              <span>The synthetic data is in Parquet format for optimal performance</span>
            </li>
            <li className="flex gap-2">
              <span>•</span>
              <span>You can use the same model to generate more samples with different labels</span>
            </li>
            <li className="flex gap-2">
              <span>•</span>
              <span>Model artifacts are saved and can be reused for future generations</span>
            </li>
            <li className="flex gap-2">
              <span>•</span>
              <span>Use the Export page to convert data to different formats if needed</span>
            </li>
          </ul>
        </CardModernContent>
      </CardModern>
    </div>
  );
}

