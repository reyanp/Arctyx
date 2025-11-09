import { CardModern, CardModernHeader, CardModernTitle, CardModernContent } from "@/components/ui/card-modern";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const data = [
  { feature: "Age", real: 35.2, synthetic: 34.8 },
  { feature: "Income", real: 62.5, synthetic: 61.9 },
  { feature: "Credit", real: 680, synthetic: 675 },
  { feature: "Balance", real: 12.5, synthetic: 13.1 },
];

export function ComparisonChart() {
  return (
    <CardModern>
      <CardModernHeader>
        <CardModernTitle>Distribution Comparison</CardModernTitle>
      </CardModernHeader>
      <CardModernContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--ash))" />
            <XAxis dataKey="feature" stroke="hsl(var(--muted-foreground))" />
            <YAxis stroke="hsl(var(--muted-foreground))" />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--ash))",
                borderRadius: "16px",
              }}
            />
            <Legend />
            <Bar dataKey="real" fill="hsl(var(--muted-foreground))" name="Real Data" radius={[4, 4, 0, 0]} />
            <Bar dataKey="synthetic" fill="hsl(var(--primary))" name="Synthetic Data" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardModernContent>
    </CardModern>
  );
}
