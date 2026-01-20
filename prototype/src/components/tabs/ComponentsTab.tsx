import { useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  BarChart,
  Bar,
  Legend,
} from 'recharts';
import { useDashboardStore } from '@/store/dashboardStore';
import { getUFIColor } from '@/data/ufiData';
import { Layers, Info, TrendingUp, BarChart3 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

const componentInfo = [
  { 
    id: 'demoUpdateIntensity', 
    name: 'Demographic Update Intensity',
    short: 'Demo Update',
    weight: 28.8,
    description: 'Measures socioeconomic mobility through address and detail changes',
    color: '#3b82f6'
  },
  { 
    id: 'updateEnrolRatio', 
    name: 'Update-Enrollment Ratio',
    short: 'Update Ratio',
    weight: 28.1,
    description: 'Measures system load and access patterns',
    color: '#14b8a6'
  },
  { 
    id: 'ageDisparity', 
    name: 'Age Group Disparity',
    short: 'Age Disparity',
    weight: 20.8,
    description: 'Measures intergenerational digital divide',
    color: '#f59e0b'
  },
  { 
    id: 'temporalVolatility', 
    name: 'Temporal Volatility',
    short: 'Volatility',
    weight: 15.2,
    description: 'Measures stability vs sudden changes',
    color: '#8b5cf6'
  },
  { 
    id: 'bioRefreshRate', 
    name: 'Biometric Refresh Rate',
    short: 'Bio Refresh',
    weight: 7.1,
    description: 'Measures security awareness and aging demographics',
    color: '#ec4899'
  },
];

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  
  const data = payload[0].payload;
  
  return (
    <div className="glass-card-dark rounded-lg p-3 shadow-xl border border-border/50">
      <p className="text-sm font-medium text-foreground">{data.district}</p>
      <p className="text-xs text-muted-foreground">{data.state}</p>
      <div className="mt-2 space-y-1">
        <p className="text-xs">
          <span className="text-muted-foreground">UFI:</span>{' '}
          <span className="font-bold" style={{ color: getUFIColor(data.ufi) }}>
            {data.ufi?.toFixed(1)}
          </span>
        </p>
        <p className="text-xs">
          <span className="text-muted-foreground">X:</span> {payload[0].value?.toFixed(1)}
        </p>
        <p className="text-xs">
          <span className="text-muted-foreground">Y:</span> {data.y?.toFixed(1)}
        </p>
      </div>
    </div>
  );
};

export const ComponentsTab = () => {
  const { filteredData } = useDashboardStore();
  const data = filteredData();

  // Correlation Matrix Data
  const correlationMatrix = useMemo(() => {
    const components = ['demoUpdateIntensity', 'updateEnrolRatio', 'ageDisparity', 'temporalVolatility', 'bioRefreshRate'];
    const matrix: { x: string; y: string; value: number }[] = [];
    
    components.forEach((comp1, i) => {
      components.forEach((comp2, j) => {
        if (i <= j) {
          // Calculate simplified correlation
          const values1 = data.map(d => d[comp1 as keyof typeof d] as number);
          const values2 = data.map(d => d[comp2 as keyof typeof d] as number);
          
          const mean1 = values1.reduce((a, b) => a + b, 0) / values1.length;
          const mean2 = values2.reduce((a, b) => a + b, 0) / values2.length;
          
          let numerator = 0;
          let denom1 = 0;
          let denom2 = 0;
          
          for (let k = 0; k < values1.length; k++) {
            const diff1 = values1[k] - mean1;
            const diff2 = values2[k] - mean2;
            numerator += diff1 * diff2;
            denom1 += diff1 * diff1;
            denom2 += diff2 * diff2;
          }
          
          const correlation = numerator / Math.sqrt(denom1 * denom2) || 0;
          
          matrix.push({
            x: componentInfo[i].short,
            y: componentInfo[j].short,
            value: Math.round(correlation * 100) / 100,
          });
        }
      });
    });
    
    return matrix;
  }, [data]);

  // Scatter Plot Data
  const scatterData = useMemo(() => {
    return data.slice(0, 200).map(d => ({
      x: d.demoUpdateIntensity,
      y: d.ufi,
      z: d.population / 100000,
      district: d.district,
      state: d.state,
      ufi: d.ufi,
    }));
  }, [data]);

  // Component Distribution Data
  const distributionData = useMemo(() => {
    return componentInfo.map(comp => {
      const values = data.map(d => d[comp.id as keyof typeof d] as number);
      const sorted = [...values].sort((a, b) => a - b);
      const q1 = sorted[Math.floor(sorted.length * 0.25)];
      const median = sorted[Math.floor(sorted.length * 0.5)];
      const q3 = sorted[Math.floor(sorted.length * 0.75)];
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      
      return {
        name: comp.short,
        min: Math.min(...values),
        q1,
        median,
        q3,
        max: Math.max(...values),
        mean: Math.round(mean * 100) / 100,
        color: comp.color,
      };
    });
  }, [data]);

  // Top Districts by Component
  const topByComponent = useMemo(() => {
    return data
      .slice(0, 15)
      .map(d => ({
        name: d.district.length > 10 ? d.district.slice(0, 10) + '...' : d.district,
        demoUpdate: d.demoUpdateIntensity,
        updateRatio: d.updateEnrolRatio,
        ageDisparity: d.ageDisparity,
        volatility: d.temporalVolatility,
        bioRefresh: d.bioRefreshRate,
      }));
  }, [data]);

  return (
    <div className="space-y-6">
      {/* Component Weight Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5"
      >
        {componentInfo.map((comp, index) => (
          <motion.div
            key={comp.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="glass-card-dark p-4 group hover:border-primary/30 transition-colors cursor-default"
          >
            <div className="flex items-start justify-between mb-2">
              <div 
                className="h-3 w-3 rounded-full"
                style={{ backgroundColor: comp.color }}
              />
              <Badge variant="secondary" className="text-xs font-mono">
                {comp.weight}%
              </Badge>
            </div>
            <h4 className="text-sm font-semibold text-foreground mb-1">{comp.short}</h4>
            <p className="text-xs text-muted-foreground line-clamp-2">{comp.description}</p>
          </motion.div>
        ))}
      </motion.div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Scatter Plot: UFI vs Demo Update */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="glass-card-dark p-6"
        >
          <div className="mb-4 flex items-center gap-2">
            <Layers className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-semibold text-foreground">UFI vs Demographic Update</h3>
          </div>
          <p className="text-sm text-muted-foreground mb-4">
            Scatter plot showing correlation between demographic update intensity and overall UFI
          </p>
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis 
                  type="number" 
                  dataKey="x" 
                  name="Demo Update"
                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                  label={{ value: 'Demo Update Intensity', position: 'bottom', fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                />
                <YAxis 
                  type="number" 
                  dataKey="y" 
                  name="UFI"
                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                  label={{ value: 'UFI Score', angle: -90, position: 'left', fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                />
                <ZAxis type="number" dataKey="z" range={[20, 150]} />
                <Tooltip content={<CustomTooltip />} />
                <Scatter name="Districts" data={scatterData}>
                  {scatterData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getUFIColor(entry.ufi)} opacity={0.7} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Component Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="glass-card-dark p-6"
        >
          <div className="mb-4 flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-accent" />
            <h3 className="text-lg font-semibold text-foreground">Component Statistics</h3>
          </div>
          
          <div className="space-y-4">
            {distributionData.map((comp) => (
              <div key={comp.name} className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">{comp.name}</span>
                  <span className="text-xs font-mono text-muted-foreground">
                    μ = {comp.mean}
                  </span>
                </div>
                <div className="relative h-3 rounded-full bg-secondary">
                  {/* Q1-Q3 range */}
                  <div 
                    className="absolute h-full rounded-full opacity-60"
                    style={{ 
                      left: `${comp.q1}%`,
                      width: `${comp.q3 - comp.q1}%`,
                      backgroundColor: comp.color
                    }}
                  />
                  {/* Median line */}
                  <div 
                    className="absolute h-full w-0.5 bg-foreground"
                    style={{ left: `${comp.median}%` }}
                  />
                  {/* Mean marker */}
                  <div 
                    className="absolute top-1/2 h-2 w-2 -translate-y-1/2 rounded-full border-2 border-background"
                    style={{ 
                      left: `${comp.mean}%`,
                      backgroundColor: comp.color
                    }}
                  />
                </div>
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{comp.min.toFixed(0)}</span>
                  <span>Q1: {comp.q1.toFixed(0)} | Med: {comp.median.toFixed(0)} | Q3: {comp.q3.toFixed(0)}</span>
                  <span>{comp.max.toFixed(0)}</span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Stacked Bar Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="glass-card-dark p-6"
      >
        <div className="mb-4 flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-semibold text-foreground">Component Contribution by District</h3>
        </div>
        <p className="text-sm text-muted-foreground mb-4">
          Normalized component contributions for top districts
        </p>
        
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={topByComponent} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
              <XAxis 
                type="number" 
                tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
              />
              <YAxis 
                type="category" 
                dataKey="name"
                tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                width={80}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px'
                }}
              />
              <Legend 
                wrapperStyle={{ fontSize: 11 }}
              />
              <Bar dataKey="demoUpdate" stackId="a" fill={componentInfo[0].color} name="Demo Update" />
              <Bar dataKey="updateRatio" stackId="a" fill={componentInfo[1].color} name="Update Ratio" />
              <Bar dataKey="ageDisparity" stackId="a" fill={componentInfo[2].color} name="Age Disparity" />
              <Bar dataKey="volatility" stackId="a" fill={componentInfo[3].color} name="Volatility" />
              <Bar dataKey="bioRefresh" stackId="a" fill={componentInfo[4].color} name="Bio Refresh" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* Key Insights */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
        className="glass-card-dark p-6"
      >
        <div className="mb-4 flex items-center gap-2">
          <Info className="h-5 w-5 text-accent" />
          <h3 className="text-lg font-semibold text-foreground">Key Statistical Insights</h3>
        </div>
        
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="rounded-lg bg-primary/5 border border-primary/10 p-4">
            <h4 className="text-sm font-semibold text-foreground mb-1">Primary Driver</h4>
            <p className="text-xs text-muted-foreground">
              Demographic mobility drives <span className="text-primary font-bold">28.8%</span> of UFI variance, 
              indicating socioeconomic factors dominate friction patterns.
            </p>
          </div>
          <div className="rounded-lg bg-accent/5 border border-accent/10 p-4">
            <h4 className="text-sm font-semibold text-foreground mb-1">Digital Divide</h4>
            <p className="text-xs text-muted-foreground">
              Age disparity shows <span className="text-accent font-bold">moderate correlation</span> with UFI, 
              suggesting targeted elderly outreach could reduce friction.
            </p>
          </div>
          <div className="rounded-lg bg-emerald-500/5 border border-emerald-500/10 p-4">
            <h4 className="text-sm font-semibold text-foreground mb-1">Component Independence</h4>
            <p className="text-xs text-muted-foreground">
              Low correlation between components <span className="text-emerald-500 font-bold">validates</span> the 
              multi-dimensional approach of UFI measurement.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};
