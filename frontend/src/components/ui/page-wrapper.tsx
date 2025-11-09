import * as React from "react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

interface PageWrapperProps extends React.HTMLAttributes<HTMLDivElement> {
	animated?: boolean
}

const PageWrapper = React.forwardRef<HTMLDivElement, PageWrapperProps>(({ className, animated = true, children, ...props }, ref) => {
	const content = (
		<div
			ref={ref}
			className={cn(
				// Centered container within 1200–1440px band with comfortable paddings
				"min-h-screen bg-background",
				"container mx-auto",
				"pt-8 pb-32 px-6",
				className,
			)}
			{...props}
		>
			{children}
		</div>
	)

	if (animated) {
		return (
			<motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, ease: "easeOut" }}>
				{content}
			</motion.div>
		)
	}

	return content
})
PageWrapper.displayName = "PageWrapper"

export { PageWrapper }

