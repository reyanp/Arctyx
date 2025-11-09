import { GenerationResults } from "@/components/GenerationResults";
import { PageWrapper } from "@/components/ui/page-wrapper";
import { SectionTitle } from "@/components/ui/section-title";

const Results = () => {
  return (
    <PageWrapper>
      <div className="space-y-12">
        {/* Results Section */}
        <section className="space-y-6">
          <SectionTitle subtitle="View your synthetic data generation results and download outputs">
            Generation Results
          </SectionTitle>
          <GenerationResults />
        </section>
      </div>
    </PageWrapper>
  );
};

export default Results;

