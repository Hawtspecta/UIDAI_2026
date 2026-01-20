import { motion } from 'framer-motion';
import { 
  Sun, 
  Moon, 
  HelpCircle, 
  Filter,
  X,
  BarChart3,
  Shield
} from 'lucide-react';
import { useDashboardStore } from '@/store/dashboardStore';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';

export const Navigation = () => {
  const { isDarkMode, toggleDarkMode,  mobileFiltersOpen, setMobileFiltersOpen } = useDashboardStore();

  return (
    <motion.nav 
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/80 backdrop-blur-xl"
    >
      <div className="container flex h-16 items-center justify-between px-4 md:px-6">
        {/* Left Section - Logo */}
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setMobileFiltersOpen(!mobileFiltersOpen)}
          >
            {mobileFiltersOpen ? <X className="h-5 w-5" /> : <Filter className="h-5 w-5" />}
          </Button>
          
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-accent shadow-lg shadow-primary/20">
                <Shield className="h-5 w-5 text-primary-foreground" />
              </div>
              <div className="absolute -bottom-1 -right-1 flex h-4 w-4 items-center justify-center rounded-full bg-accent text-[8px] font-bold text-accent-foreground">
                UFI
              </div>
            </div>
            
            <div className="hidden sm:block">
              <h1 className="text-lg font-bold tracking-tight">
                <span className="text-foreground">UIDAI</span>
                <span className="ml-1.5 text-primary">UFI Dashboard</span>
              </h1>
              <p className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
                Update Friction Index
              </p>
            </div>
          </div>
        </div>

        {/* Right Section - Actions */}
        <div className="flex items-center gap-2">
          {/* Stats Badge */}
          <div className="hidden items-center gap-2 rounded-full bg-secondary/50 px-3 py-1.5 md:flex">
            <BarChart3 className="h-3.5 w-3.5 text-primary" />
            <span className="text-xs font-medium text-muted-foreground">
              817 Districts Analyzed
            </span>
          </div>

          {/* Theme Toggle */}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleDarkMode}
            className="relative h-9 w-9 rounded-lg"
          >
            <Sun className="h-4 w-4 rotate-0 scale-100 transition-transform dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-transform dark:rotate-0 dark:scale-100" />
            <span className="sr-only">Toggle theme</span>
          </Button>

          {/* Help Dialog */}
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="ghost" size="icon" className="h-9 w-9 rounded-lg">
                <HelpCircle className="h-4 w-4" />
                <span className="sr-only">Help</span>
              </Button>
            </DialogTrigger>
            <DialogContent className="glass-card-dark max-w-lg">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2 text-xl">
                  <Shield className="h-5 w-5 text-primary" />
                  Understanding UFI Dashboard
                </DialogTitle>
                <DialogDescription className="text-muted-foreground">
                  Your guide to the Update Friction Index
                </DialogDescription>
              </DialogHeader>
              
              <div className="mt-4 space-y-6">
                <div>
                  <h3 className="mb-2 font-semibold text-foreground">What is UFI?</h3>
                  <p className="text-sm text-muted-foreground">
                    The Update Friction Index measures system-level stress in the Aadhaar ecosystem 
                    by analyzing update patterns. Think of it as a "health score" for India's 
                    identity infrastructure.
                  </p>
                </div>
                
                <div>
                  <h3 className="mb-2 font-semibold text-foreground">How to use this dashboard?</h3>
                  <ol className="list-inside list-decimal space-y-1 text-sm text-muted-foreground">
                    <li>Start with <strong>Overview</strong> tab for system health</li>
                    <li>Use <strong>Geospatial</strong> tab to identify regional patterns</li>
                    <li>Dive into <strong>Components</strong> to understand drivers</li>
                    <li>Review <strong>Insights</strong> for actionable recommendations</li>
                    <li>Simulate interventions in <strong>Predictive</strong> tab</li>
                  </ol>
                </div>
                
                <div>
                  <h3 className="mb-3 font-semibold text-foreground">What do the colors mean?</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="flex items-center gap-2 rounded-lg bg-emerald-500/10 p-2">
                      <div className="h-3 w-3 rounded-full bg-emerald-500" />
                      <span className="text-xs font-medium">0-25: Low Friction</span>
                    </div>
                    <div className="flex items-center gap-2 rounded-lg bg-amber-500/10 p-2">
                      <div className="h-3 w-3 rounded-full bg-amber-500" />
                      <span className="text-xs font-medium">25-50: Moderate</span>
                    </div>
                    <div className="flex items-center gap-2 rounded-lg bg-orange-500/10 p-2">
                      <div className="h-3 w-3 rounded-full bg-orange-500" />
                      <span className="text-xs font-medium">50-75: High Friction</span>
                    </div>
                    <div className="flex items-center gap-2 rounded-lg bg-red-500/10 p-2">
                      <div className="h-3 w-3 rounded-full bg-red-500" />
                      <span className="text-xs font-medium">75-100: Critical</span>
                    </div>
                  </div>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>
    </motion.nav>
  );
};
