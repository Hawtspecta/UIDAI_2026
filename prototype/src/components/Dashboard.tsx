import { useEffect } from 'react';
import { useDashboardStore } from '@/store/dashboardStore';
import { Navigation } from '@/components/Navigation';
import { HeroSection } from '@/components/HeroSection';
import { TabNavigation } from '@/components/TabNavigation';
import { FilterSidebar } from '@/components/FilterSidebar';
import { OverviewTab } from '@/components/tabs/OverviewTab';
import { GeospatialTab } from '@/components/tabs/GeospatialTab';
import { ComponentsTab } from '@/components/tabs/ComponentsTab';
import { InsightsTab } from '@/components/tabs/InsightsTab';
import { PredictiveTab } from '@/components/tabs/PredictiveTab';
import { motion, AnimatePresence } from 'framer-motion';
import { Sheet, SheetContent, SheetTitle, SheetDescription } from '@/components/ui/sheet'; // Replaced Drawer with Sheet

const Dashboard = () => {
  const { activeTab, initializeData, isLoading, sidebarOpen, mobileFiltersOpen, setMobileFiltersOpen } = useDashboardStore();

  useEffect(() => {
    initializeData();
  }, [initializeData]);

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="text-center space-y-4">
          <div className="relative">
            <div className="inline-block h-16 w-16 animate-spin rounded-full border-4 border-primary/30 border-t-primary" />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="h-8 w-8 rounded-full bg-primary animate-pulse" />
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-lg font-semibold text-foreground">Loading UFI Dashboard</p>
            <p className="text-sm text-muted-foreground">Processing 817 districts...</p>
          </div>
          <div className="flex gap-2 justify-center">
            <div className="h-2 w-2 rounded-full bg-primary animate-bounce [animation-delay:-0.3s]" />
            <div className="h-2 w-2 rounded-full bg-primary animate-bounce [animation-delay:-0.15s]" />
            <div className="h-2 w-2 rounded-full bg-primary animate-bounce" />
          </div>
        </div>
      </div>
    );
  }
  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewTab />;
      case 'geospatial':
        return <GeospatialTab />;
      case 'components':
        return <ComponentsTab />;
      case 'insights':
        return <InsightsTab />;
      case 'predictive':
        return <PredictiveTab />;
      default:
        return <OverviewTab />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <HeroSection />
      <TabNavigation />

      <div className="container px-4 py-6 md:px-6">
        <div className="flex gap-6">
          {/* Sidebar */}
          <div className={`hidden lg:block ${sidebarOpen ? '' : 'lg:hidden'}`}>
            <FilterSidebar />
          </div>

          {/* Main Content */}
          <main className="flex-1 min-w-0">
            {/* Mobile Filter Sheet - full screen */}
            <Sheet open={mobileFiltersOpen} onOpenChange={setMobileFiltersOpen}>
              <SheetContent side="left" className="w-80 p-4 overflow-y-auto">
                <SheetDescription className="sr-only">Adjust filters to customize the dashboard view.</SheetDescription>
                <FilterSidebar />
              </SheetContent>
            </Sheet>
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
              >
                {renderTabContent()}
              </motion.div>
            </AnimatePresence>
          </main>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-border/40 bg-card/50 py-6 mt-12">
        <div className="container px-4 md:px-6 text-center">
          <p className="text-sm text-muted-foreground">
            UIDAI Update Friction Index Dashboard • Data Intelligence Platform
          </p>
          <p className="text-xs text-muted-foreground/60 mt-1">
            © 2024 Government of India • All Rights Reserved
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Dashboard;
