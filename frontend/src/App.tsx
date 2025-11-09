import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { NavBar } from "@/components/ui/tubelight-navbar";
import { Database, Upload, Download, BookOpen } from "lucide-react";
import Index from "./pages/Index";
import Schema from "./pages/Schema";
import Export from "./pages/Export";
import LearningHub from "./pages/LearningHub";
import NotFound from "./pages/NotFound";
import { useEffect } from "react";

const queryClient = new QueryClient();

// Component to scroll to top on route change
function ScrollToTop() {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return null;
}

const navItems = [
  { name: 'Upload', url: '/', icon: Upload },
  { name: 'Schema', url: '/schema', icon: Database },
  { name: 'Export', url: '/export', icon: Download },
  { name: 'Learn', url: '/learn', icon: BookOpen },
];

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <ScrollToTop />
        <div className="flex min-h-screen w-full flex-col bg-background">
          <main className="flex-1 overflow-auto">
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/schema" element={<Schema />} />
              <Route path="/export" element={<Export />} />
              <Route path="/learn" element={<LearningHub />} />
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </main>
        </div>
        {/* Tubelight Navbar - fixed at bottom center */}
        <NavBar items={navItems} />
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
