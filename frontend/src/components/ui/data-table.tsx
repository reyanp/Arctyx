import * as React from "react"
import { cn } from "@/lib/utils"

const DataTable = React.forwardRef<HTMLTableElement, React.HTMLAttributes<HTMLTableElement>>(({ className, ...props }, ref) => (
	<div className="relative w-full overflow-auto rounded-card border border-ash bg-card">
		<table ref={ref} className={cn("w-full caption-bottom text-sm", className)} {...props} />
	</div>
))
DataTable.displayName = "DataTable"

const DataTableHeader = React.forwardRef<HTMLTableSectionElement, React.HTMLAttributes<HTMLTableSectionElement>>(
	({ className, ...props }, ref) => <thead ref={ref} className={cn("bg-gray-50 border-b border-ash", className)} {...props} />,
)
DataTableHeader.displayName = "DataTableHeader"

const DataTableBody = React.forwardRef<HTMLTableSectionElement, React.HTMLAttributes<HTMLTableSectionElement>>(
	({ className, ...props }, ref) => <tbody ref={ref} className={cn("[&_tr:last-child]:border-0", className)} {...props} />,
)
DataTableBody.displayName = "DataTableBody"

const DataTableFooter = React.forwardRef<HTMLTableSectionElement, React.HTMLAttributes<HTMLTableSectionElement>>(
	({ className, ...props }, ref) => <tfoot ref={ref} className={cn("border-t border-ash bg-gray-50 font-medium", className)} {...props} />,
)
DataTableFooter.displayName = "DataTableFooter"

const DataTableRow = React.forwardRef<HTMLTableRowElement, React.HTMLAttributes<HTMLTableRowElement>>(({ className, ...props }, ref) => (
	<tr ref={ref} className={cn("border-b border-ash transition-colors duration-150 ease-in-out hover:bg-gray-50 h-11", className)} {...props} />
))
DataTableRow.displayName = "DataTableRow"

const DataTableHead = React.forwardRef<HTMLTableCellElement, React.ThHTMLAttributes<HTMLTableCellElement>>(({ className, ...props }, ref) => (
	<th
		ref={ref}
		className={cn(
			"h-11 px-4 text-left align-middle font-semibold text-[13px] text-foreground/75 [&:has([role=checkbox])]:pr-0",
			className,
		)}
		{...props}
	/>
))
DataTableHead.displayName = "DataTableHead"

const DataTableCell = React.forwardRef<HTMLTableCellElement, React.TdHTMLAttributes<HTMLTableCellElement>>(({ className, ...props }, ref) => (
	<td ref={ref} className={cn("px-4 align-middle text-foreground [&:has([role=checkbox])]:pr-0", className)} {...props} />
))
DataTableCell.displayName = "DataTableCell"

const DataTableCaption = React.forwardRef<HTMLTableCaptionElement, React.HTMLAttributes<HTMLTableCaptionElement>>(
	({ className, ...props }, ref) => <caption ref={ref} className={cn("mt-4 text-sm text-foreground/70", className)} {...props} />,
)
DataTableCaption.displayName = "DataTableCaption"

export {
	DataTable,
	DataTableHeader,
	DataTableBody,
	DataTableFooter,
	DataTableHead,
	DataTableRow,
	DataTableCell,
	DataTableCaption,
}

