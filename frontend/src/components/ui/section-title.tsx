import * as React from "react"
import { cn } from "@/lib/utils"

interface SectionTitleProps extends React.HTMLAttributes<HTMLHeadingElement> {
	subtitle?: string
	underline?: boolean
}

const SectionTitle = React.forwardRef<HTMLHeadingElement, SectionTitleProps>(({ className, subtitle, underline = true, children, ...props }, ref) => {
	return (
		<div className="mb-6">
			<h2
				ref={ref}
				className={cn(
					"text-[22px] font-semibold tracking-tight mb-3 text-foreground",
					underline && "inline-block pb-1 border-b-2 border-teal/30",
					className,
				)}
				{...props}
			>
				{children}
			</h2>
			{subtitle && <p className="text-sm text-foreground/75 mt-2">{subtitle}</p>}
		</div>
	)
})
SectionTitle.displayName = "SectionTitle"

export { SectionTitle }

