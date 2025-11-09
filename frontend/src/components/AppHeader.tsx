import { useLocation } from "react-router-dom"
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { User, Settings, LogOut, Info } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"

// Page title mapping based on route
const getPageTitle = (pathname: string): string => {
  const routeMap: Record<string, string> = {
    "/": "Dashboard",
    "/schema": "Schema",
    "/generate": "Generate",
    "/results": "Results",
    "/export": "Export",
  }
  return routeMap[pathname] || "Dashboard"
}

export function AppHeader() {
  const location = useLocation()
  const pageTitle = getPageTitle(location.pathname)

  return (
    <header 
      className="sticky top-0 z-10 flex h-16 items-center gap-4 border-b px-6"
      style={{
        backgroundColor: "#FAFAF8",
        borderBottomColor: "#E8E4D9",
        borderBottomWidth: "1px",
        borderBottomStyle: "solid",
      }}
    >
      {/* Left side: Page title */}
      <div className="flex items-center gap-4 flex-1">
        <h1 className="text-lg font-semibold text-foreground">
          {pageTitle}
        </h1>
      </div>

      {/* Right side: User dropdown with Settings */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" className="relative h-10 w-10 rounded-full">
            <Avatar className="h-9 w-9">
              <AvatarFallback>
                <User className="h-4 w-4" />
              </AvatarFallback>
            </Avatar>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-56" align="end" forceMount>
          <DropdownMenuLabel className="font-normal">
            <div className="flex flex-col space-y-1">
              <p className="text-sm font-medium leading-none">Session Info</p>
              <p className="text-xs leading-none text-muted-foreground">
                Active session
              </p>
            </div>
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem>
            <Info className="mr-2 h-4 w-4" />
            <span>Session Details</span>
          </DropdownMenuItem>
          <DropdownMenuItem>
            <Settings className="mr-2 h-4 w-4" />
            <span>Settings</span>
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem>
            <LogOut className="mr-2 h-4 w-4" />
            <span>Log out</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </header>
  )
}

