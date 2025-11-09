import { FileUpload } from "@/components/FileUpload";
import { PageWrapper } from "@/components/ui/page-wrapper";
import { SectionTitle } from "@/components/ui/section-title";
import { CardModern, CardModernHeader, CardModernTitle, CardModernContent } from "@/components/ui/card-modern";
import { FileSpreadsheet, Clock, ChevronRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { BackendStatusIndicator } from "@/components/BackendStatusIndicator";

// Mock data for upload history
const uploadHistory = [
  {
    id: 1,
    name: "customer_data.csv",
    size: "2.3 MB",
    rows: 1000,
    uploadedAt: "2 hours ago",
    status: "completed"
  },
  {
    id: 2,
    name: "sales_records.csv",
    size: "5.7 MB",
    rows: 2500,
    uploadedAt: "1 day ago",
    status: "completed"
  },
  {
    id: 3,
    name: "financial_data.csv",
    size: "1.8 MB",
    rows: 850,
    uploadedAt: "3 days ago",
    status: "completed"
  },
];

const Index = () => {
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

        {/* Backend Status */}
        <section className="flex justify-center">
          <BackendStatusIndicator variant="badge" />
        </section>

        {/* Dataset Upload Section */}
        <section className="space-y-6">
          <SectionTitle subtitle="Upload your dataset to begin the data generation workflow">
            Upload Dataset
          </SectionTitle>
          <FileUpload />
        </section>

        {/* Upload History Section */}
        <section className="space-y-6">
          <SectionTitle subtitle="View and manage your previously uploaded datasets">
            History
          </SectionTitle>
          
          <CardModern>
            <CardModernContent>
              <div className="space-y-3">
                {uploadHistory.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer group"
                  >
                    <div className="flex items-center justify-center w-10 h-10 bg-white rounded-lg border border-ash flex-shrink-0">
                      <FileSpreadsheet className="h-5 w-5 text-primary" />
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <p className="font-medium text-foreground truncate">{item.name}</p>
                        <Badge variant="secondary" className="flex-shrink-0">
                          {item.rows} rows
                        </Badge>
                      </div>
                      <div className="flex items-center gap-3 text-xs text-muted-foreground">
                        <span>{item.size}</span>
                        <span>•</span>
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {item.uploadedAt}
                        </span>
                      </div>
                    </div>
                    
                    <Button 
                      variant="ghost" 
                      size="sm"
                      className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      View
                      <ChevronRight className="h-4 w-4 ml-1" />
                    </Button>
                  </div>
                ))}
                
                {uploadHistory.length === 0 && (
                  <div className="text-center py-12 text-muted-foreground">
                    <FileSpreadsheet className="h-12 w-12 mx-auto mb-3 opacity-20" />
                    <p>No upload history yet</p>
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
