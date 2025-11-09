import { CardModern, CardModernHeader, CardModernTitle, CardModernContent } from "@/components/ui/card-modern";
import { Badge } from "@/components/ui/badge";

const mockSchema = [
  { name: "customer_id", type: "integer", nullable: false },
  { name: "age", type: "integer", nullable: false },
  { name: "income", type: "float", nullable: false },
  { name: "credit_score", type: "integer", nullable: false },
  { name: "loan_default", type: "boolean", nullable: false },
];

const classBalance = {
  positive: 152,
  negative: 848,
  total: 1000,
};

export function SchemaCard() {
  return (
    <CardModern>
      <CardModernHeader>
        <div className="flex items-center justify-between">
          <CardModernTitle>Detected Schema</CardModernTitle>
          <Badge variant="outline" className="bg-muted">
            {mockSchema.length} columns
          </Badge>
        </div>
      </CardModernHeader>

      <CardModernContent>
        <div className="space-y-3 mb-6">
          {mockSchema.map((col) => (
            <div
              key={col.name}
              className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
            >
              <span className="font-medium text-sm">{col.name}</span>
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="text-xs">
                  {col.type}
                </Badge>
                {!col.nullable && (
                  <Badge variant="outline" className="text-xs">
                    required
                  </Badge>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="border-t border-ash pt-6">
          <h3 className="text-sm font-medium mb-4">Class Imbalance</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Positive Class</span>
              <span className="font-medium">
                {classBalance.positive} ({((classBalance.positive / classBalance.total) * 100).toFixed(1)}%)
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Negative Class</span>
              <span className="font-medium">
                {classBalance.negative} ({((classBalance.negative / classBalance.total) * 100).toFixed(1)}%)
              </span>
            </div>
            <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary"
                style={{ width: `${(classBalance.positive / classBalance.total) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </CardModernContent>
    </CardModern>
  );
}
