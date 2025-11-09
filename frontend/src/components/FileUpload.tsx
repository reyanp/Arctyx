import { FileSpreadsheet, Upload, X } from "lucide-react";
import { CardModern } from "@/components/ui/card-modern";
import { UploadDropzone } from "@/components/ui/upload-dropzone";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

export function FileUpload() {
  const [file, setFile] = useState<File | null>(null);
  const navigate = useNavigate();

  const handleFilesSelected = (files: FileList | null) => {
    if (files && files[0]) {
      setFile(files[0]);
    }
  };

  const handleUpload = () => {
    if (file) {
      // TODO: Add actual file upload logic here
      console.log("Uploading file:", file.name);
      // Navigate to schema page after upload
      navigate("/schema");
    }
  };

  return (
    <CardModern>
      {!file ? (
        <UploadDropzone
          accept=".csv"
          onFilesSelected={handleFilesSelected}
        >
          <p className="text-sm font-medium text-foreground mb-1">Upload CSV Dataset</p>
          <p className="text-xs text-foreground/60">Drag and drop your file here, or click to browse</p>
        </UploadDropzone>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg">
            <FileSpreadsheet className="h-10 w-10 text-primary flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="font-medium text-foreground truncate">{file.name}</p>
              <p className="text-sm text-muted-foreground">
                {(file.size / 1024).toFixed(2)} KB
              </p>
            </div>
            <Button 
              variant="ghost" 
              size="icon"
              onClick={() => setFile(null)}
              className="flex-shrink-0"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          
          <div className="flex gap-3">
            <Button 
              variant="outline" 
              onClick={() => setFile(null)}
              className="flex-1"
            >
              Cancel
            </Button>
            <Button 
              onClick={handleUpload}
              className="flex-1"
            >
              <Upload className="h-4 w-4 mr-2" />
              Upload & Continue
            </Button>
          </div>
        </div>
      )}
    </CardModern>
  );
}
