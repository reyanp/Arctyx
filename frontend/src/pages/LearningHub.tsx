import { PageWrapper } from "@/components/ui/page-wrapper";
import { SectionTitle } from "@/components/ui/section-title";
import { CardModern, CardModernContent } from "@/components/ui/card-modern";

const LearningHub = () => {
  return (
    <PageWrapper>
      <div className="space-y-12">
        {/* Sections 1 & 2 - Two Column Layout */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Section 1: Why This Matters */}
          <div className="space-y-6">
            <SectionTitle>
              Why This Matters
            </SectionTitle>
            <CardModern>
              <CardModernContent>
                <div className="space-y-6">
                  {/* Problem Statement */}
                  <p className="text-sm text-foreground leading-relaxed">
                    Modern ML is limited far more by data than by models. Teams spend most of their time 
                    labeling, cleaning, and fixing datasets rather than training. DataFoundry exists to 
                    automate these painful steps.
                  </p>

                  {/* Core Issues */}
                  <div className="space-y-3">
                    <h4 className="text-sm font-semibold text-foreground">The Core Issues:</h4>
                    <ul className="space-y-2 text-sm text-muted-foreground">
                      <li className="flex items-start gap-2">
                        <span className="text-primary mt-0.5">•</span>
                        <span>Labeled data is slow and expensive to create.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-primary mt-0.5">•</span>
                        <span>Most datasets are too small or imbalanced.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-primary mt-0.5">•</span>
                        <span>Real-world data is messy and inconsistent.</span>
                      </li>
                    </ul>
                  </div>

                  {/* Our Goal */}
                  <div className="space-y-3">
                    <h4 className="text-sm font-semibold text-foreground">Our Goal:</h4>
                    <ul className="space-y-2 text-sm text-muted-foreground">
                      <li className="flex items-start gap-2">
                        <span className="text-primary mt-0.5">•</span>
                        <span>Automate labeling with weak supervision.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-primary mt-0.5">•</span>
                        <span>Generate high-quality synthetic samples.</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-primary mt-0.5">•</span>
                        <span>Find and clean anomalies automatically.</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </CardModernContent>
            </CardModern>
          </div>

          {/* Section 2: The Three-Pipeline System */}
          <div className="space-y-6">
            <SectionTitle>
              The Three-Pipeline System
            </SectionTitle>
            <CardModern>
              <CardModernContent>
                <div className="space-y-5">
                  {/* Intro */}
                  <p className="text-sm text-foreground leading-relaxed">
                    DataFoundry automates the three hardest parts of data-centric AI through a coordinated pipeline system.
                  </p>

                  {/* Pipeline 1: Labeling */}
                  <div className="space-y-1">
                    <h4 className="text-sm font-semibold text-foreground">
                      <span className="text-primary">1.</span> Labeling (Weak Supervision):
                    </h4>
                    <p className="text-sm text-muted-foreground leading-relaxed pl-5">
                      AI-generated heuristic rules label large datasets quickly using Snorkel-style weak supervision.
                    </p>
                  </div>

                  {/* Pipeline 2: Generation */}
                  <div className="space-y-1">
                    <h4 className="text-sm font-semibold text-foreground">
                      <span className="text-primary">2.</span> Generation (Synthetic Data):
                    </h4>
                    <p className="text-sm text-muted-foreground leading-relaxed pl-5">
                      A generative model (CTGAN or cVAE) produces high-quality synthetic samples to expand and rebalance datasets.
                    </p>
                  </div>

                  {/* Pipeline 3: Cleaning */}
                  <div className="space-y-1">
                    <h4 className="text-sm font-semibold text-foreground">
                      <span className="text-primary">3.</span> Cleaning (Anomaly Detection):
                    </h4>
                    <p className="text-sm text-muted-foreground leading-relaxed pl-5">
                      Outlier detection and reconstruction-based checks identify and remove noisy or inconsistent data.
                    </p>
                  </div>
                </div>
              </CardModernContent>
            </CardModern>
          </div>
        </section>

        {/* Section 3: How the Agents Work - Full Width */}
        <section className="mt-12 space-y-6">
          <SectionTitle>
            How the Agents Work
          </SectionTitle>
          <CardModern>
            <CardModernContent>
              <div className="space-y-5">
                {/* Intro */}
                <p className="text-sm text-foreground leading-relaxed">
                  DataFoundry uses an orchestrated team of AI agents, each responsible for a specific part of the data pipeline. 
                  This creates a fast, reliable workflow without manual intervention.
                </p>

                {/* Flow Step 1 */}
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-foreground">
                    <span className="text-primary">1.</span> Orchestrator Agent:
                  </h4>
                  <p className="text-sm text-muted-foreground leading-relaxed pl-5">
                    Coordinates the workflow and decides which pipeline to run based on the user's dataset and settings.
                  </p>
                </div>

                {/* Flow Step 2 */}
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-foreground">
                    <span className="text-primary">2.</span> Specialist Agents:
                  </h4>
                  <p className="text-sm text-muted-foreground leading-relaxed pl-5">
                    Individual agents handle labeling, generation, or anomaly detection using Nemotron-powered reasoning.
                  </p>
                </div>

                {/* Flow Step 3 */}
                <div className="space-y-1">
                  <h4 className="text-sm font-semibold text-foreground">
                    <span className="text-primary">3.</span> Unified Output:
                  </h4>
                  <p className="text-sm text-muted-foreground leading-relaxed pl-5">
                    Results from each agent are merged into a cleaned, labeled, and optionally expanded dataset ready for export or training.
                  </p>
                </div>
              </div>
            </CardModernContent>
          </CardModern>
        </section>

        {/* Sections 4 & 5 - Two Column Layout */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-12">
          {/* Section 4: How We Use Nemotron */}
          <div className="space-y-6">
            <SectionTitle>
              How We Use Nemotron
            </SectionTitle>
            <CardModern>
              <CardModernContent>
                <div className="space-y-5">
                  {/* Intro */}
                  <p className="text-sm text-foreground leading-relaxed">
                    Nemotron acts as the reasoning engine behind several key steps in the DataFoundry pipeline.
                  </p>

                  {/* Bullet Points */}
                  <div className="space-y-3">
                    <div className="flex items-start gap-2">
                      <span className="text-primary mt-0.5">•</span>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        <span className="font-semibold text-foreground">Weak Label Rule Generation:</span> Nemotron proposes heuristic labeling rules that our system converts into Snorkel-style weak supervision.
                      </p>
                    </div>

                    <div className="flex items-start gap-2">
                      <span className="text-primary mt-0.5">•</span>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        <span className="font-semibold text-foreground">Schema & Pattern Understanding:</span> Nemotron analyzes column names, distributions, and relationships to help agents infer data types and potential issues.
                      </p>
                    </div>

                    <div className="flex items-start gap-2">
                      <span className="text-primary mt-0.5">•</span>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        <span className="font-semibold text-foreground">Anomaly Insights:</span> Nemotron provides natural-language reasoning about suspicious patterns, enabling downstream agents to detect outliers more effectively.
                      </p>
                    </div>
                  </div>

                  {/* Closing */}
                  <p className="text-sm text-foreground leading-relaxed">
                    These contributions allow each agent to make more informed decisions with minimal human input.
                  </p>
                </div>
              </CardModernContent>
            </CardModern>
          </div>

          {/* Section 5: FAQ & Open Source */}
          <div className="space-y-6">
            <SectionTitle>
              FAQ & Open Source
            </SectionTitle>
            <CardModern>
              <CardModernContent>
                <div className="space-y-6">
                  {/* Intro */}
                  <p className="text-sm text-foreground leading-relaxed">
                    A few common questions about synthetic data and how we approach openness.
                  </p>

                  {/* FAQ Items */}
                  <div className="space-y-4">
                    {/* FAQ 1 */}
                    <div className="space-y-1">
                      <h4 className="text-sm font-semibold text-foreground">
                        Is synthetic data safe to use?
                      </h4>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        Yes, when generated correctly, it preserves patterns without exposing sensitive records.
                      </p>
                    </div>

                    {/* FAQ 2 */}
                    <div className="space-y-1">
                      <h4 className="text-sm font-semibold text-foreground">
                        How accurate are weak labels?
                      </h4>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        They're not perfect, but they provide strong signal quickly and can be improved with more rules.
                      </p>
                    </div>

                    {/* FAQ 3 */}
                    <div className="space-y-1">
                      <h4 className="text-sm font-semibold text-foreground">
                        Why focus on data instead of models?
                      </h4>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        Most ML failures come from poor data quality, not model architecture.
                      </p>
                    </div>

                    {/* FAQ 4 */}
                    <div className="space-y-1">
                      <h4 className="text-sm font-semibold text-foreground">
                        Can the system be extended?
                      </h4>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        Absolutely. Each pipeline is modular and easy to customize or replace.
                      </p>
                    </div>
                  </div>

                  {/* Open Source Section */}
                  <div className="pt-2 space-y-2">
                    <h4 className="text-sm font-semibold text-foreground">
                      Open Source & Learning
                    </h4>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      One of our goals is to make data-centric AI more accessible. We openly document our pipelines, 
                      share implementation details, and encourage others to learn from or build on the project.
                    </p>
                  </div>
                </div>
              </CardModernContent>
            </CardModern>
          </div>
        </section>
      </div>
    </PageWrapper>
  );
};

export default LearningHub;

