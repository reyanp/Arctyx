import { CardModern, CardModernHeader, CardModernTitle, CardModernContent } from "@/components/ui/card-modern";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Sparkles } from "lucide-react";
import { useNavigate } from "react-router-dom";

export function GenerationPanel() {
  const navigate = useNavigate();

  const handleGenerate = () => {
    // TODO: Add actual generation logic here
    console.log("Generating synthetic data...");
    // Navigate to results page after generation
    navigate("/results");
  };

  return (
    <CardModern>
      <CardModernHeader>
        <CardModernTitle>Generate Synthetic Data</CardModernTitle>
      </CardModernHeader>
      <CardModernContent>

      <div className="space-y-4 mb-6">
        <div className="space-y-2">
          <Label htmlFor="samples">Number of Samples</Label>
          <Input
            id="samples"
            type="number"
            defaultValue="1000"
            placeholder="Enter number of samples"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="model">AI Model</Label>
          <Select defaultValue="ctgan">
            <SelectTrigger id="model">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ctgan">CTGAN</SelectItem>
              <SelectItem value="copulagan">CopulaGAN</SelectItem>
              <SelectItem value="tvae">TVAE</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="epochs">Training Epochs</Label>
          <Input
            id="epochs"
            type="number"
            defaultValue="300"
            placeholder="Number of epochs"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="balance">Balance Classes</Label>
          <Select defaultValue="yes">
            <SelectTrigger id="balance">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="yes">Yes</SelectItem>
              <SelectItem value="no">No</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <Button className="w-full" size="lg" onClick={handleGenerate}>
        <Sparkles className="w-4 h-4 mr-2" />
        Generate Dataset
      </Button>
      </CardModernContent>
    </CardModern>
  );
}
