import { useState, useEffect } from "react";
import { FileUpload } from "@/components/FileUpload";
import { PageWrapper } from "@/components/ui/page-wrapper";
import { SectionTitle } from "@/components/ui/section-title";
import { CardModern, CardModernHeader, CardModernTitle, CardModernContent } from "@/components/ui/card-modern";
import { FileSpreadsheet, Clock, Download } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { getFileDownloadUrl } from "@/lib/api";

interface GenerationHistoryItem {
  id: string;
  syntheticDataPath: string;
  originalDataPath: string;
  numSamples?: number;
  modelPath: string;
  configPath: string;
  timestamp: number;
}

const Index = () => {
  const [generationHistory, setGenerationHistory] = useState<GenerationHistoryItem[]>([]);

  useEffect(() => {
    // Load generation history from sessionStorage
    const historyKey = 'generationHistory';
    const storedHistory = sessionStorage.getItem(historyKey);
    
    if (storedHistory) {
      try {
        const history = JSON.parse(storedHistory);
        setGenerationHistory(history);
      } catch (error) {
        console.error('Failed to parse generation history:', error);
      }
    }
  }, []);

  const formatTimestamp = (timestamp: number) => {
    const now = Date.now();
    const diff = now - timestamp;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes} minute${minutes === 1 ? '' : 's'} ago`;
    if (hours < 24) return `${hours} hour${hours === 1 ? '' : 's'} ago`;
    return `${days} day${days === 1 ? '' : 's'} ago`;
  };

  const getFileName = (path: string) => {
    return path.split('/').pop() || path;
  };

  const handleDownload = (filePath: string, event: React.MouseEvent) => {
    event.stopPropagation();
    const downloadUrl = getFileDownloadUrl(filePath);
    window.open(downloadUrl, '_blank');
  };
  return (
    <PageWrapper>
      <div className="space-y-12">
        {/* Hero Section */}
        <section className="text-center py-8 space-y-4">
          {/* Title Logo */}
          <div className="flex justify-center">
            <img 
              src="/Arctyxlogo.png" 
              alt="Arctyx" 
              className="h-32"
            />
          </div>

          {/* Subtitle */}
          <p className="text-[17px] text-muted-foreground max-w-2xl mx-auto font-normal pt-1">
            A smarter way to label, generate, and clean data.
          </p>

          {/* Frost Divider */}
          <div className="pt-6">
            <div className="w-full max-w-3xl mx-auto h-px bg-gradient-to-r from-transparent via-ash to-transparent"></div>
          </div>
        </section>

        {/* Dataset Upload Section */}
        <section className="space-y-6">
          <SectionTitle subtitle="Upload your dataset to begin the data generation workflow">
            Upload Dataset
          </SectionTitle>
          <FileUpload />
        </section>

        {/* Generation History Section */}
        <section className="space-y-6">
          <SectionTitle subtitle="View your previously generated synthetic datasets">
            Generation History
          </SectionTitle>
          
          <CardModern>
            <CardModernContent>
              <div className="space-y-3">
                {generationHistory.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors group"
                  >
                    <div className="flex items-center justify-center w-10 h-10 bg-white rounded-lg border border-ash flex-shrink-0">
                      <FileSpreadsheet className="h-5 w-5 text-primary" />
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <p className="font-medium text-foreground truncate">
                          {getFileName(item.syntheticDataPath)}
                        </p>
                        {item.numSamples && (
                          <Badge variant="secondary" className="flex-shrink-0">
                            {item.numSamples.toLocaleString()} samples
                          </Badge>
                        )}
                      </div>
                      <div className="flex items-center gap-3 text-xs text-muted-foreground">
                        <span className="truncate">
                          From: {getFileName(item.originalDataPath)}
                        </span>
                        <span>•</span>
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {formatTimestamp(item.timestamp)}
                        </span>
                      </div>
                    </div>
                    
                    <Button 
                      variant="ghost" 
                      size="sm"
                      onClick={(e) => handleDownload(item.syntheticDataPath, e)}
                      className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download
                    </Button>
                  </div>
                ))}
                
                {generationHistory.length === 0 && (
                  <div className="text-center py-12 text-muted-foreground">
                    <FileSpreadsheet className="h-12 w-12 mx-auto mb-3 opacity-20" />
                    <p>No generation history yet</p>
                    <p className="text-xs mt-2">Generate synthetic data to see it here</p>
                  </div>
                )}
              </div>
            </CardModernContent>
          </CardModern>
        </section>
      </div>
    </PageWrapper>
  );
};

export default Index;
