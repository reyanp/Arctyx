import { CardModern, CardModernHeader, CardModernTitle, CardModernContent } from "@/components/ui/card-modern";
import { SectionTitle } from "@/components/ui/section-title";
import {
  DataTable,
  DataTableBody,
  DataTableCell,
  DataTableHead,
  DataTableHeader,
  DataTableRow,
} from "@/components/ui/data-table";
import { Badge } from "@/components/ui/badge";

const sampleData = [
  { id: 1, age: 32, income: 58000, credit: 720, default: "No" },
  { id: 2, age: 45, income: 72000, credit: 680, default: "No" },
  { id: 3, age: 28, income: 48000, credit: 650, default: "Yes" },
  { id: 4, age: 52, income: 95000, credit: 740, default: "No" },
  { id: 5, age: 38, income: 64000, credit: 690, default: "No" },
];

export function ResultsTable() {
  return (
    <CardModern>
      <CardModernHeader>
        <SectionTitle>Synthetic Samples Preview</SectionTitle>
      </CardModernHeader>
      <CardModernContent>
        <DataTable>
          <DataTableHeader>
            <DataTableRow>
              <DataTableHead className="text-right">ID</DataTableHead>
              <DataTableHead>Age</DataTableHead>
              <DataTableHead className="text-right">Income</DataTableHead>
              <DataTableHead className="text-right">Credit Score</DataTableHead>
              <DataTableHead>Default</DataTableHead>
            </DataTableRow>
          </DataTableHeader>
          <DataTableBody>
            {sampleData.map((row) => (
              <DataTableRow key={row.id}>
                <DataTableCell className="font-medium text-right">{row.id}</DataTableCell>
                <DataTableCell>{row.age}</DataTableCell>
                <DataTableCell className="text-right">${row.income.toLocaleString()}</DataTableCell>
                <DataTableCell className="text-right">{row.credit}</DataTableCell>
                <DataTableCell>
                  <Badge variant={row.default === "No" ? "secondary" : "destructive"}>
                    {row.default}
                  </Badge>
                </DataTableCell>
              </DataTableRow>
            ))}
          </DataTableBody>
        </DataTable>
      </CardModernContent>
    </CardModern>
  );
}
