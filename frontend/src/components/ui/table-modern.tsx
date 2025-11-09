import * as React from "react"
import { cn } from "@/lib/utils"

const TableModern = React.forwardRef<
  HTMLTableElement,
  React.HTMLAttributes<HTMLTableElement>
>(({ className, ...props }, ref) => (
  <div className="relative w-full overflow-auto rounded-2xl border border-ash">
    <table
      ref={ref}
      className={cn("w-full caption-bottom text-sm", className)}
      {...props}
    />
  </div>
))
TableModern.displayName = "TableModern"

const TableModernHeader = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <thead 
    ref={ref} 
    className={cn("bg-gray-50 border-b border-ash", className)} 
    {...props} 
  />
))
TableModernHeader.displayName = "TableModernHeader"

const TableModernBody = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <tbody
    ref={ref}
    className={cn("[&_tr:last-child]:border-0 bg-white", className)}
    {...props}
  />
))
TableModernBody.displayName = "TableModernBody"

const TableModernFooter = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <tfoot
    ref={ref}
    className={cn(
      "border-t border-ash bg-gray-50 font-medium",
      className
    )}
    {...props}
  />
))
TableModernFooter.displayName = "TableModernFooter"

const TableModernRow = React.forwardRef<
  HTMLTableRowElement,
  React.HTMLAttributes<HTMLTableRowElement>
>(({ className, ...props }, ref) => (
  <tr
    ref={ref}
    className={cn(
      "border-b border-ash transition-colors hover:bg-gray-50 h-11",
      className
    )}
    {...props}
  />
))
TableModernRow.displayName = "TableModernRow"

const TableModernHead = React.forwardRef<
  HTMLTableCellElement,
  React.ThHTMLAttributes<HTMLTableCellElement>
>(({ className, ...props }, ref) => (
  <th
    ref={ref}
    className={cn(
      "h-11 px-4 text-left align-middle font-semibold text-[13px] [&:has([role=checkbox])]:pr-0",
      className
    )}
    {...props}
  />
))
TableModernHead.displayName = "TableModernHead"

const TableModernCell = React.forwardRef<
  HTMLTableCellElement,
  React.TdHTMLAttributes<HTMLTableCellElement>
>(({ className, ...props }, ref) => (
  <td
    ref={ref}
    className={cn("px-4 align-middle [&:has([role=checkbox])]:pr-0", className)}
    {...props}
  />
))
TableModernCell.displayName = "TableModernCell"

const TableModernCaption = React.forwardRef<
  HTMLTableCaptionElement,
  React.HTMLAttributes<HTMLTableCaptionElement>
>(({ className, ...props }, ref) => (
  <caption
    ref={ref}
    className={cn("mt-4 text-sm text-muted-foreground", className)}
    {...props}
  />
))
TableModernCaption.displayName = "TableModernCaption"

export {
  TableModern,
  TableModernHeader,
  TableModernBody,
  TableModernFooter,
  TableModernHead,
  TableModernRow,
  TableModernCell,
  TableModernCaption,
}

