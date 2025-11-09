import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { SchemaCard } from "@/components/SchemaCard";
import { GenerationPanel } from "@/components/GenerationPanel";
import { PageWrapper } from "@/components/ui/page-wrapper";
import { SectionTitle } from "@/components/ui/section-title";
import { LoadingOverlay } from "@/components/LoadingOverlay";

const Schema = () => {
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleGenerate = () => {
    // Start loading
    setLoading(true);
    
    // After 75 seconds (1 min 15 sec), navigate to results
    setTimeout(() => {
      setLoading(false);
      navigate("/results");
    }, 75000);
  };

  return (
    <>
      {/* Loading Overlay */}
      {loading && <LoadingOverlay visible={loading} />}
      
      <PageWrapper>
        <div className="space-y-12">
          {/* Configuration Section */}
          <section className="space-y-6">
            <SectionTitle subtitle="Review your dataset structure and configure generation settings">
              Configuration
            </SectionTitle>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <SchemaCard />
              <GenerationPanel onGenerate={handleGenerate} />
            </div>
          </section>
        </div>
      </PageWrapper>
    </>
  );
};

export default Schema;

