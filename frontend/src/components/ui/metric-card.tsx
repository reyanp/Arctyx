import * as React from "react"
import { LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"

interface MetricCardProps extends React.HTMLAttributes<HTMLDivElement> {
  icon: LucideIcon
  value: string | number
  label: string
  delta?: number
  deltaType?: "increase" | "decrease" | "neutral"
}

const MetricCard = React.forwardRef<HTMLDivElement, MetricCardProps>(
  ({ className, icon: Icon, value, label, delta, deltaType = "neutral", ...props }, ref) => {
    const deltaColor = 
      deltaType === "increase" ? "text-success" :
      deltaType === "decrease" ? "text-destructive" :
      "text-foreground/70"

    const deltaSign = delta && delta > 0 ? "+" : ""

    return (
      <div
        ref={ref}
        className={cn(
          "rounded-card bg-card shadow-soft border border-ash p-6",
          "transition-all duration-150 ease-in-out h-40 flex flex-col justify-between",
          className
        )}
        {...props}
      >
        {/* Top row: icon and delta */}
        <div className="flex items-start justify-between">
          <div className="rounded-lg bg-gray-50 p-2">
            <Icon className="h-5 w-5 text-foreground" />
          </div>
          {delta !== undefined && (
            <span className={cn("text-sm font-semibold", deltaColor)}>
              {deltaSign}{delta}%
            </span>
          )}
        </div>

        {/* Middle: main value */}
        <div className="text-2xl font-semibold tracking-tight">
          {value}
        </div>

        {/* Bottom: label */}
        <div className="text-sm text-foreground/75">
          {label}
        </div>
      </div>
    )
  }
)
MetricCard.displayName = "MetricCard"

export { MetricCard }

