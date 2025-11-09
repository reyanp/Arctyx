import { MetricCard } from "@/components/ui/metric-card";
import { TrendingUp, Target, Zap, CheckCircle } from "lucide-react";

const metrics = [
  {
    label: "AUC Score",
    value: "0.94",
    delta: 2.3,
    deltaType: "increase" as const,
    icon: Target,
  },
  {
    label: "Accuracy",
    value: "89.2%",
    delta: 1.8,
    deltaType: "increase" as const,
    icon: CheckCircle,
  },
  {
    label: "F1 Score",
    value: "0.87",
    delta: 3.1,
    deltaType: "increase" as const,
    icon: TrendingUp,
  },
  {
    label: "Generation Speed",
    value: "1.2s",
    delta: -0.3,
    deltaType: "decrease" as const,
    icon: Zap,
  },
];

export function MetricsGrid() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {metrics.map((metric) => (
        <MetricCard
          key={metric.label}
          icon={metric.icon}
          value={metric.value}
          label={metric.label}
          delta={metric.delta}
          deltaType={metric.deltaType}
        />
      ))}
    </div>
  );
}
