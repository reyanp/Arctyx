import { SchemaCard } from "@/components/SchemaCard";
import { GenerationPanel } from "@/components/GenerationPanel";
import { PageWrapper } from "@/components/ui/page-wrapper";
import { SectionTitle } from "@/components/ui/section-title";

const Schema = () => {
  return (
    <PageWrapper>
      <div className="space-y-12">
        {/* Configuration Section */}
        <section className="space-y-6">
          <SectionTitle subtitle="Review your dataset structure and configure generation settings">
            Configuration
          </SectionTitle>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SchemaCard />
            <GenerationPanel />
          </div>
        </section>
      </div>
    </PageWrapper>
  );
};

export default Schema;

