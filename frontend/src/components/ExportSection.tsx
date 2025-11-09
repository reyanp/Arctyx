import { CardModern, CardModernHeader, CardModernTitle, CardModernContent } from "@/components/ui/card-modern";
import { Button } from "@/components/ui/button";
import { Download, FileText } from "lucide-react";

export function ExportSection() {
  return (
    <CardModern>
      <CardModernHeader>
        <CardModernTitle>Export Results</CardModernTitle>
      </CardModernHeader>
      <CardModernContent>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center gap-3">
            <FileText className="w-5 h-5 text-muted-foreground" />
            <div>
              <p className="font-medium">Synthetic Dataset</p>
              <p className="text-sm text-muted-foreground">CSV format • 1000 rows</p>
            </div>
          </div>
          <Button variant="outline" size="sm">
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
        </div>

        <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center gap-3">
            <FileText className="w-5 h-5 text-muted-foreground" />
            <div>
              <p className="font-medium">Summary Report</p>
              <p className="text-sm text-muted-foreground">PDF format • Metrics & charts</p>
            </div>
          </div>
          <Button variant="outline" size="sm">
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
        </div>

        <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center gap-3">
            <FileText className="w-5 h-5 text-muted-foreground" />
            <div>
              <p className="font-medium">Model Weights</p>
              <p className="text-sm text-muted-foreground">PKL format • Trained model</p>
            </div>
          </div>
          <Button variant="outline" size="sm">
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
        </div>

          <Button className="w-full mt-4" variant="default">
            Export All Files
          </Button>
        </div>
      </CardModernContent>
    </CardModern>
  );
}
