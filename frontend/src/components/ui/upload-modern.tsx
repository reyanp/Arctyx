import * as React from "react"
import { Upload } from "lucide-react"
import { cn } from "@/lib/utils"

interface UploadModernProps extends React.HTMLAttributes<HTMLDivElement> {
  onFileSelect?: (files: FileList | null) => void
  accept?: string
  multiple?: boolean
}

const UploadModern = React.forwardRef<HTMLDivElement, UploadModernProps>(
  ({ className, onFileSelect, accept, multiple = false, children, ...props }, ref) => {
    const inputRef = React.useRef<HTMLInputElement>(null)

    const handleClick = () => {
      inputRef.current?.click()
    }

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (onFileSelect) {
        onFileSelect(e.target.files)
      }
    }

    return (
      <div
        ref={ref}
        onClick={handleClick}
        className={cn(
          "rounded-2xl bg-white border-2 border-dashed border-ash p-8",
          "flex flex-col items-center justify-center cursor-pointer",
          "transition-all duration-150 hover:bg-gray-50",
          "min-h-[200px]",
          className
        )}
        {...props}
      >
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept={accept}
          multiple={multiple}
          onChange={handleChange}
        />
        {children || (
          <>
            <Upload className="h-10 w-10 text-muted-foreground mb-4" />
            <p className="text-sm font-medium text-foreground mb-1">
              Click to upload or drag and drop
            </p>
            <p className="text-xs text-muted-foreground">
              {accept || "Any file type"}
            </p>
          </>
        )}
      </div>
    )
  }
)
UploadModern.displayName = "UploadModern"

export { UploadModern }

