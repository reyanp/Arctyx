import { PageWrapper } from "@/components/ui/page-wrapper";
import { SectionTitle } from "@/components/ui/section-title";
import { CardModern, CardModernContent } from "@/components/ui/card-modern";

const Settings = () => {
  return (
    <PageWrapper>
      <div className="space-y-12">
        {/* Settings Section */}
        <section className="space-y-6">
          <SectionTitle subtitle="Configure application preferences">
            Settings
          </SectionTitle>
          <CardModern>
            <CardModernContent>
              <p className="text-muted-foreground text-center py-12">
                Settings page coming soon...
              </p>
            </CardModernContent>
          </CardModern>
        </section>
      </div>
    </PageWrapper>
  );
};

export default Settings;

