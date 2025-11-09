import { MetricsGrid } from "@/components/MetricsGrid";
import { ComparisonChart } from "@/components/ComparisonChart";
import { PageWrapper } from "@/components/ui/page-wrapper";
import { SectionTitle } from "@/components/ui/section-title";

const Results = () => {
  return (
    <PageWrapper>
      <div className="space-y-12">
        {/* Metrics Section */}
        <section className="space-y-6">
          <SectionTitle subtitle="Real-time performance metrics for your synthetic data">
            Performance Metrics
          </SectionTitle>
          <MetricsGrid />
        </section>

        {/* Distribution Comparison Section */}
        <section className="space-y-6">
          <SectionTitle subtitle="Compare real vs synthetic data distributions">
            Distribution Comparison
          </SectionTitle>
          <ComparisonChart />
        </section>
      </div>
    </PageWrapper>
  );
};

export default Results;

