import { motion } from 'framer-motion';
import { LucideIcon, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { cn } from '@/lib/utils';

interface KPICardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  trendValue?: string;
  color?: 'default' | 'success' | 'warning' | 'danger' | 'info';
  delay?: number;
}

const colorStyles = {
  default: 'from-primary/20 to-primary/5 border-primary/20',
  success: 'from-emerald-500/20 to-emerald-500/5 border-emerald-500/20',
  warning: 'from-amber-500/20 to-amber-500/5 border-amber-500/20',
  danger: 'from-red-500/20 to-red-500/5 border-red-500/20',
  info: 'from-accent/20 to-accent/5 border-accent/20',
};

const iconColors = {
  default: 'text-primary bg-primary/10',
  success: 'text-emerald-500 bg-emerald-500/10',
  warning: 'text-amber-500 bg-amber-500/10',
  danger: 'text-red-500 bg-red-500/10',
  info: 'text-accent bg-accent/10',
};

export const KPICard = ({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  trendValue,
  color = 'default',
  delay = 0,
}: KPICardProps) => {
  const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus;
  const trendColor = trend === 'up' ? 'text-emerald-500' : trend === 'down' ? 'text-red-500' : 'text-muted-foreground';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: 'easeOut' }}
      className={cn(
        'kpi-card group relative overflow-hidden border bg-gradient-to-br',
        colorStyles[color]
      )}
    >
      {/* Background Glow Effect */}
      <div className="absolute -right-8 -top-8 h-32 w-32 rounded-full bg-gradient-to-br from-primary/10 to-transparent opacity-0 blur-2xl transition-opacity duration-500 group-hover:opacity-100" />
      
      <div className="relative">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              {title}
            </p>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="number-display text-3xl font-bold tracking-tight text-foreground">
                {value}
              </span>
              {trendValue && (
                <span className={cn('flex items-center gap-0.5 text-xs font-medium', trendColor)}>
                  <TrendIcon className="h-3 w-3" />
                  {trendValue}
                </span>
              )}
            </div>
            {subtitle && (
              <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>
            )}
          </div>
          
          <div className={cn('rounded-xl p-3', iconColors[color])}>
            <Icon className="h-5 w-5" />
          </div>
        </div>
      </div>
    </motion.div>
  );
};
