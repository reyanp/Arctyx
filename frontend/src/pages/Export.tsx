import { ExportSection } from "@/components/ExportSection";
import { DataPreview } from "@/components/DataPreview";
import { PageWrapper } from "@/components/ui/page-wrapper";
import { SectionTitle } from "@/components/ui/section-title";

const Export = () => {
  return (
    <PageWrapper>
      <div className="space-y-12">
        {/* Export Section */}
        <section className="space-y-6">
          <SectionTitle subtitle="Download your generated synthetic data">
            Export Results
          </SectionTitle>
          <ExportSection />
        </section>

        {/* Synthetic Samples Preview */}
        <section className="space-y-6">
          <SectionTitle subtitle="Preview of generated synthetic samples">
            Synthetic Data Preview
          </SectionTitle>
          <DataPreview />
        </section>
      </div>
    </PageWrapper>
  );
};

export default Export;

