import { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Area,
  AreaChart,
  Legend,
} from 'recharts';
import { useDashboardStore } from '@/store/dashboardStore';
import { getUFIColor } from '@/data/ufiData';
import { TrendingUp, AlertTriangle, MapPin, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

const COLORS = {
  low: '#10b981',
  moderate: '#f59e0b',
  high: '#f97316',
  critical: '#ef4444',
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  
  return (
    <div className="glass-card-dark rounded-lg p-3 shadow-xl border border-border/50">
      <p className="text-sm font-medium text-foreground">{label}</p>
      {payload.map((entry: any, index: number) => (
        <p key={index} className="text-xs text-muted-foreground mt-1">
          <span style={{ color: entry.color }}>●</span> {entry.name}: {entry.value}
        </p>
      ))}
    </div>
  );
};

export const OverviewTab = () => {
  const { filteredData, stateStats, setActiveTab } = useDashboardStore();
  const data = filteredData();

  // UFI Distribution Histogram
  const histogramData = useMemo(() => {
    const buckets = Array.from({ length: 10 }, (_, i) => ({
      range: `${i * 10}-${(i + 1) * 10}`,
      count: 0,
      start: i * 10,
    }));
    
    data.forEach(d => {
      const bucket = Math.min(Math.floor(d.ufi / 10), 9);
      buckets[bucket].count++;
    });
    
    return buckets;
  }, [data]);

  // Category Distribution for Pie Chart
  const categoryData = useMemo(() => {
    const counts = {
      'Low Friction': 0,
      'Moderate Friction': 0,
      'High Friction': 0,
      'Very High Friction': 0,
    };
    
    data.forEach(d => {
      counts[d.ufiCategory]++;
    });
    
    return Object.entries(counts).map(([name, value]) => ({
      name,
      value,
      color: name === 'Low Friction' ? COLORS.low 
        : name === 'Moderate Friction' ? COLORS.moderate
        : name === 'High Friction' ? COLORS.high
        : COLORS.critical,
    }));
  }, [data]);

  // Top 10 High Friction Districts
  const topDistricts = useMemo(() => {
    return [...data]
      .sort((a, b) => b.ufi - a.ufi)
      .slice(0, 10)
      .map(d => ({
        name: d.district,
        state: d.state,
        ufi: d.ufi,
        fill: getUFIColor(d.ufi),
      }));
  }, [data]);

  // State Performance Summary
  const topStates = useMemo(() => {
    return stateStats.slice(0, 8).map(s => ({
      name: s.state.length > 12 ? s.state.slice(0, 12) + '...' : s.state,
      fullName: s.state,
      ufi: s.meanUFI,
      districts: s.districtCount,
      highFriction: s.highFrictionCount,
    }));
  }, [stateStats]);

  return (
    <div className="space-y-6">
      {/* Main Histogram */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="glass-card-dark p-6"
      >
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-foreground">UFI Distribution</h3>
            <p className="text-sm text-muted-foreground">
              Distribution of districts by Update Friction Index score
            </p>
          </div>
          <div className="hidden sm:flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="h-3 w-3 rounded-sm bg-emerald-500" />
              <span className="text-muted-foreground">Low</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="h-3 w-3 rounded-sm bg-amber-500" />
              <span className="text-muted-foreground">Moderate</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="h-3 w-3 rounded-sm bg-orange-500" />
              <span className="text-muted-foreground">High</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="h-3 w-3 rounded-sm bg-red-500" />
              <span className="text-muted-foreground">Critical</span>
            </div>
          </div>
        </div>
        
        <div className="h-64 md:h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={histogramData} barCategoryGap="10%">
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
              <XAxis 
                dataKey="range" 
                tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                axisLine={{ stroke: 'hsl(var(--border))' }}
              />
              <YAxis 
                tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                axisLine={{ stroke: 'hsl(var(--border))' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar 
                dataKey="count" 
                name="Districts"
                radius={[4, 4, 0, 0]}
              >
                {histogramData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={
                      entry.start < 25 ? COLORS.low
                        : entry.start < 50 ? COLORS.moderate
                        : entry.start < 75 ? COLORS.high
                        : COLORS.critical
                    }
                    opacity={0.9}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* Secondary Visualizations Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Category Pie Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="glass-card-dark p-6"
        >
          <h3 className="mb-4 text-lg font-semibold text-foreground">UFI Categories</h3>
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={categoryData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={3}
                  dataKey="value"
                  label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                  labelLine={false}
                >
                  {categoryData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-2">
            {categoryData.map(cat => (
              <div 
                key={cat.name}
                className="flex items-center gap-2 rounded-lg bg-secondary/30 p-2"
              >
                <div 
                  className="h-3 w-3 rounded-full" 
                  style={{ backgroundColor: cat.color }}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-foreground truncate">
                    {cat.name.replace(' Friction', '')}
                  </p>
                  <p className="text-xs text-muted-foreground">{cat.value} districts</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Top Friction Districts */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="glass-card-dark p-6"
        >
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-orange-500" />
              <h3 className="text-lg font-semibold text-foreground">Top Friction Districts</h3>
            </div>
            <Button 
              variant="ghost" 
              size="sm"
              onClick={() => setActiveTab('geospatial')}
              className="gap-1 text-xs text-primary"
            >
              View Map <ArrowRight className="h-3 w-3" />
            </Button>
          </div>
          
          <div className="space-y-2">
            {topDistricts.map((district, index) => (
              <div
                key={district.name}
                className="group flex items-center gap-3 rounded-lg bg-secondary/20 p-2.5 transition-colors hover:bg-secondary/40"
              >
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-secondary text-xs font-bold text-muted-foreground">
                  {index + 1}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">
                    {district.name}
                  </p>
                  <p className="text-xs text-muted-foreground">{district.state}</p>
                </div>
                <div className="text-right">
                  <p 
                    className="text-sm font-bold"
                    style={{ color: district.fill }}
                  >
                    {district.ufi.toFixed(1)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* State Performance Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="glass-card-dark p-6"
      >
        <div className="mb-4 flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-semibold text-foreground">State Performance Summary</h3>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border/50">
                <th className="pb-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">State</th>
                <th className="pb-3 text-center text-xs font-medium uppercase tracking-wider text-muted-foreground">Mean UFI</th>
                <th className="pb-3 text-center text-xs font-medium uppercase tracking-wider text-muted-foreground">Districts</th>
                <th className="pb-3 text-center text-xs font-medium uppercase tracking-wider text-muted-foreground">High Friction</th>
                <th className="pb-3 text-right text-xs font-medium uppercase tracking-wider text-muted-foreground">Distribution</th>
              </tr>
            </thead>
            <tbody>
              {topStates.map((state) => (
                <tr 
                  key={state.fullName}
                  className="border-b border-border/20 hover:bg-secondary/20 transition-colors"
                >
                  <td className="py-3">
                    <div className="flex items-center gap-2">
                      <MapPin className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm font-medium text-foreground">{state.name}</span>
                    </div>
                  </td>
                  <td className="py-3 text-center">
                    <span 
                      className="text-sm font-bold"
                      style={{ color: getUFIColor(state.ufi) }}
                    >
                      {state.ufi.toFixed(1)}
                    </span>
                  </td>
                  <td className="py-3 text-center text-sm text-muted-foreground">
                    {state.districts}
                  </td>
                  <td className="py-3 text-center">
                    <span className={`text-sm font-medium ${state.highFriction > 5 ? 'text-red-500' : 'text-muted-foreground'}`}>
                      {state.highFriction}
                    </span>
                  </td>
                  <td className="py-3">
                    <div className="flex justify-end">
                      <div className="h-2 w-24 rounded-full bg-secondary overflow-hidden">
                        <div 
                          className="h-full rounded-full transition-all"
                          style={{ 
                            width: `${state.ufi}%`,
                            background: `linear-gradient(90deg, ${getUFIColor(state.ufi * 0.5)}, ${getUFIColor(state.ufi)})`
                          }}
                        />
                      </div>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
  );
};
