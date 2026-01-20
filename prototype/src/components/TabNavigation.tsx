import { motion, AnimatePresence } from 'framer-motion';
import { 
  LayoutDashboard, 
  Map, 
  Layers, 
  Lightbulb, 
  TrendingUp,
  LucideIcon
} from 'lucide-react';
import { useDashboardStore, TabType } from '@/store/dashboardStore';
import { cn } from '@/lib/utils';

interface TabItem {
  id: TabType;
  label: string;
  icon: LucideIcon;
  emoji: string;
}

const tabs: TabItem[] = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard, emoji: '📊' },
  { id: 'geospatial', label: 'Geospatial', icon: Map, emoji: '🗺️' },
  { id: 'components', label: 'Components', icon: Layers, emoji: '🧬' },
  { id: 'insights', label: 'Insights', icon: Lightbulb, emoji: '💡' },
  { id: 'predictive', label: 'Predictive', icon: TrendingUp, emoji: '🔮' },
];

export const TabNavigation = () => {
  const { activeTab, setActiveTab } = useDashboardStore();

  return (
    <div className="border-b border-border/40 bg-background/50 backdrop-blur-sm">
      <div className="container px-4 md:px-6">
        <nav className="flex items-center gap-1 overflow-x-auto py-2 scrollbar-hide md:gap-2">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            const Icon = tab.icon;
            
            return (
              <motion.button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={cn(
                  'relative flex items-center gap-2 whitespace-nowrap rounded-lg px-4 py-2.5 text-sm font-medium transition-all duration-200',
                  isActive 
                    ? 'text-primary' 
                    : 'text-muted-foreground hover:bg-secondary/50 hover:text-foreground'
                )}
              >
                <span className="hidden text-base md:inline">{tab.emoji}</span>
                <Icon className="h-4 w-4 md:hidden" />
                <span className="hidden sm:inline">{tab.label}</span>
                
                {/* Active Indicator */}
                <AnimatePresence>
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="absolute inset-0 -z-10 rounded-lg bg-primary/10"
                    />
                  )}
                </AnimatePresence>
                
                {/* Bottom Line Indicator */}
                {isActive && (
                  <motion.div
                    layoutId="activeTabLine"
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ duration: 0.2 }}
                    className="absolute -bottom-2 left-0 right-0 h-0.5 rounded-full bg-primary"
                  />
                )}
              </motion.button>
            );
          })}
        </nav>
      </div>
    </div>
  );
};
