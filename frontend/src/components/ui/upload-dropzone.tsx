import * as React from "react"
import { UploadCloud } from "lucide-react"
import { cn } from "@/lib/utils"

type UploadDropzoneProps = React.HTMLAttributes<HTMLDivElement> & {
	accept?: string
	multiple?: boolean
	onFilesSelected?: (files: FileList | null) => void
}

export const UploadDropzone = React.forwardRef<HTMLDivElement, UploadDropzoneProps>(
	({ className, accept, multiple = false, onFilesSelected, children, ...props }, ref) => {
		const inputRef = React.useRef<HTMLInputElement>(null)

		const handleClick = () => inputRef.current?.click()
		const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => onFilesSelected?.(e.target.files)

		return (
			<div
				ref={ref}
				onClick={handleClick}
			className={cn(
				"rounded-card border-2 border-dashed border-ash bg-white",
				"flex flex-col items-center justify-center text-center",
				"transition-colors duration-150 ease-in-out cursor-pointer",
				"p-8 min-h-[200px] hover:border-ash/80",
				className,
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
				{children ?? (
					<>
						<UploadCloud className="h-10 w-10 text-foreground/60 mb-4" />
						<p className="text-sm font-medium text-foreground mb-1">Click to upload or drag and drop</p>
						<p className="text-xs text-foreground/60">{accept || "All file types"}</p>
					</>
				)}
			</div>
		)
	},
)
UploadDropzone.displayName = "UploadDropzone"


