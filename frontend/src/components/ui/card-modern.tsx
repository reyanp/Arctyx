import * as React from "react"
import { cn } from "@/lib/utils"

const CardModern = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
	<div
		ref={ref}
		className={cn(
			"rounded-card bg-card text-card-foreground shadow-soft border border-ash p-6 transition-all duration-150 ease-in-out",
			className,
		)}
		{...props}
	/>
))
CardModern.displayName = "CardModern"

const CardModernHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
	<div ref={ref} className={cn("flex flex-col gap-1.5 mb-4", className)} {...props} />
))
CardModernHeader.displayName = "CardModernHeader"

const CardModernTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
	<h3 ref={ref} className={cn("text-[22px] font-semibold leading-none tracking-tight text-foreground", className)} {...props} />
))
CardModernTitle.displayName = "CardModernTitle"

const CardModernDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
	<p ref={ref} className={cn("text-sm text-foreground/75", className)} {...props} />
))
CardModernDescription.displayName = "CardModernDescription"

const CardModernContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => <div ref={ref} className={cn("flex flex-col gap-4", className)} {...props} />)
CardModernContent.displayName = "CardModernContent"

const CardModernFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
	<div ref={ref} className={cn("flex items-center mt-4 pt-4 border-t border-ash", className)} {...props} />
))
CardModernFooter.displayName = "CardModernFooter"

export {
  CardModern,
  CardModernHeader,
  CardModernFooter,
  CardModernTitle,
  CardModernDescription,
  CardModernContent,
}

