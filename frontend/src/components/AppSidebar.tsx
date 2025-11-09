import { Database, Upload, BarChart3, Settings, Download } from "lucide-react";
import { NavLink } from "@/components/NavLink";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";

const navItems = [
  { title: "Upload Data", url: "/", icon: Upload },
  { title: "Schema", url: "/schema", icon: Database },
  { title: "Generate", url: "/generate", icon: BarChart3 },
  { title: "Export", url: "/export", icon: Download },
];

export function AppSidebar() {
  const { open } = useSidebar();

  return (
    <Sidebar collapsible="icon" className="border-r border-sidebar-border">
      <SidebarContent>
        <div className="px-6 py-4">
          <h1 className={`font-semibold text-sidebar-foreground transition-all ${open ? "text-lg" : "text-xs"}`}>
            {open ? "Synthetic Data Factory" : "SDF"}
          </h1>
        </div>
        
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navItems.map((item) => (
                <SidebarMenuItem key={item.url}>
                  <SidebarMenuButton asChild>
                    <NavLink
                      to={item.url}
                      end
                      className="flex items-center gap-3 hover:bg-sidebar-accent transition-colors"
                      activeClassName="bg-sidebar-accent text-sidebar-foreground font-medium"
                    >
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <div className="mt-auto px-4 py-4 border-t border-sidebar-border">
          <SidebarMenuButton asChild>
            <button className="flex items-center gap-3 w-full hover:bg-sidebar-accent transition-colors rounded-md px-2 py-2">
              <Settings className="h-4 w-4" />
              {open && <span className="text-sm">Settings</span>}
            </button>
          </SidebarMenuButton>
        </div>
      </SidebarContent>
    </Sidebar>
  );
}
