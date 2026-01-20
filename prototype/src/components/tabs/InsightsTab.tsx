import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '@/store/dashboardStore';
import { getUFIColor } from '@/data/ufiData';
import {
  AlertTriangle,
  MapPin,
  Users,
  TrendingUp,
  Download,
  ArrowRight,
  Lightbulb,
  Target,
  Zap,
  Award
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface InsightCardProps {
  icon: React.ReactNode;
  iconBg: string;
  title: string;
  subtitle: string;
  count?: number;
  children: React.ReactNode;
  recommendation: string;
  priority?: 'critical' | 'high' | 'medium' | 'low';
  delay?: number;
}

const InsightCard = ({
  icon,
  iconBg,
  title,
  subtitle,
  count,
  children,
  recommendation,
  priority = 'medium',
  delay = 0
}: InsightCardProps) => {
  const priorityColors = {
    critical: 'bg-red-500/10 text-red-500 border-red-500/20',
    high: 'bg-orange-500/10 text-orange-500 border-orange-500/20',
    medium: 'bg-amber-500/10 text-amber-500 border-amber-500/20',
    low: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className="insight-card group"
    >
      <div className="flex items-start gap-4 mb-4">
        <div className={`rounded-xl p-3 ${iconBg}`}>
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-lg font-semibold text-foreground">{title}</h3>
            {count !== undefined && (
              <Badge variant="secondary" className="font-mono">
                {count} districts
              </Badge>
            )}
          </div>
          <p className="text-sm text-muted-foreground">{subtitle}</p>
        </div>
      </div>

      <div className="mb-4">
        {children}
      </div>

      <div className={`rounded-lg border p-4 ${priorityColors[priority]}`}>
        <div className="flex items-start gap-3">
          <Lightbulb className="h-4 w-4 mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium mb-1">Recommendation</p>
            <p className="text-sm opacity-90">{recommendation}</p>
          </div>
        </div>
      </div>

      <div className="mt-4 flex items-center gap-2">
        <Button variant="outline" size="sm" className="gap-1.5">
          <Download className="h-3.5 w-3.5" />
          Export
        </Button>
        <Button variant="ghost" size="sm" className="gap-1.5 text-primary">
          View Details
          <ArrowRight className="h-3.5 w-3.5" />
        </Button>
      </div>
    </motion.div>
  );
};

export const InsightsTab = () => {
  const { filteredData, stateStats, setActiveTab } = useDashboardStore();
  const data = filteredData();

  // Update Deserts (Low UFI but high population)
  const updateDeserts = useMemo(() => {
    return data
      .filter(d => d.ufi < 30 && d.totalEnrollments < 100000)  // Low activity, low enrollments
      .sort((a, b) => a.demoUpdateIntensity - b.demoUpdateIntensity)  // Lowest demo update first
      .slice(0, 10);
  }, [data]);

  // High Friction Zones
  const highFrictionZones = useMemo(() => {
    return data
      .filter(d => d.ufi >= 50)  // Changed from >= 70
      .sort((a, b) => b.ufi - a.ufi)
      .slice(0, 10);
  }, [data]);

  // Age Gap Crisis
  const ageGapCrisis = useMemo(() => {
    return data
      .filter(d => d.ageDisparity >= 2)  // Changed from >= 15 - much lower threshold
      .sort((a, b) => b.ageDisparity - a.ageDisparity)
      .slice(0, 20);  // Increased slice from 10 to 20 for more options
  }, [data]);

  // State Performance
  const statePerformance = useMemo(() => {
    return stateStats.slice(0, 10).map(s => ({
      name: s.state.length > 14 ? s.state.slice(0, 14) + '...' : s.state,
      fullName: s.state,
      ufi: s.meanUFI,
      districts: s.districtCount,
    }));
  }, [stateStats]);

  // Metrics
  const metrics = useMemo(() => {
    const totalPop = data.reduce((sum, d) => sum + d.population, 0);
    const criticalPop = data.filter(d => d.ufi >= 70).reduce((sum, d) => sum + d.population, 0);

    return {
      criticalPopulation: (criticalPop / 1000000).toFixed(1),
      potentialImpact: (criticalPop / totalPop * 100).toFixed(1),
      avgReduction: 15.3,
      estimatedROI: 2.4,
    };
  }, [data]);

  const getUpdateDesertsRecommendation = () => {
    if (updateDeserts.length === 0) return "No data available";
    const totalPop = updateDeserts.reduce((sum, d) => sum + d.population, 0);
    const avgUFI = updateDeserts.reduce((sum, d) => sum + d.ufi, 0) / updateDeserts.length;
    const unitsNeeded = Math.ceil(updateDeserts.length / 3);

    return `Deploy ${unitsNeeded} mobile enrollment units to ${updateDeserts.length} underserved districts. Target population: ${(totalPop / 1000000).toFixed(1)}M. Current avg UFI: ${avgUFI.toFixed(1)}. Estimated impact: ${(totalPop * 0.15 / 1000000).toFixed(1)}M additional citizens served.`;
  };

  const getHighFrictionRecommendation = () => {
    if (highFrictionZones.length === 0) return "No data available";
    const avgUFI = highFrictionZones.reduce((sum, d) => sum + d.ufi, 0) / highFrictionZones.length;
    const criticalCount = highFrictionZones.filter(d => d.ufi >= 80).length;
    const capacityIncrease = Math.ceil(avgUFI / 50 * 40);

    return `Increase UIDAI center capacity by ${capacityIncrease}% in ${highFrictionZones.length} critical zones (${criticalCount} at critical level). Avg UFI: ${avgUFI.toFixed(1)}. Estimated cost: ₹${(highFrictionZones.length * 5).toFixed(0)} Cr. Expected UFI reduction: ${Math.min(avgUFI - 30, 25).toFixed(1)} points.`;
  };

  const getAgeGapRecommendation = () => {
    if (ageGapCrisis.length === 0) return "No data available";
    const avgDisparity = ageGapCrisis.reduce((sum, d) => sum + d.ageDisparity, 0) / ageGapCrisis.length;
    const targetPopulation = ageGapCrisis.reduce((sum, d) => sum + d.population, 0) / 1000000;

    return `Launch elderly-focused outreach campaign in top ${Math.min(ageGapCrisis.length, 15)} districts with age disparity avg: ${avgDisparity.toFixed(1)}. Target: ${targetPopulation.toFixed(1)}M elderly citizens. Include assisted enrollment camps and community centers. Staffing: 200+ trained counselors.`;
  };

  const getStatePerformanceRecommendation = () => {
    const bestState = stateStats[0];
    const worstState = stateStats[stateStats.length - 1];
    const improvement = (worstState.meanUFI - bestState.meanUFI).toFixed(1);

    return `Study best practices from top performer ${bestState.state} (UFI: ${bestState.meanUFI.toFixed(1)}). Implement cross-state knowledge transfer with ${worstState.state} (UFI: ${worstState.meanUFI.toFixed(1)}). Potential improvement: ${improvement} UFI points. Focus on: infrastructure, staffing, technology adoption.`;
  };

  return (
    <div className="space-y-6">
      {/* Metrics Row */}
      <div className="grid gap-4 sm:grid-cols-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
          className="glass-card-dark p-4 text-center"
        >
          <Target className="h-5 w-5 mx-auto mb-2 text-primary" />
          <p className="text-2xl font-bold text-foreground">{metrics.criticalPopulation}M</p>
          <p className="text-xs text-muted-foreground">Population in Critical Zones</p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.1 }}
          className="glass-card-dark p-4 text-center"
        >
          <Zap className="h-5 w-5 mx-auto mb-2 text-amber-500" />
          <p className="text-2xl font-bold text-foreground">{metrics.potentialImpact}%</p>
          <p className="text-xs text-muted-foreground">Potential Impact Area</p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          className="glass-card-dark p-4 text-center"
        >
          <TrendingUp className="h-5 w-5 mx-auto mb-2 text-emerald-500" />
          <p className="text-2xl font-bold text-foreground">-{metrics.avgReduction}</p>
          <p className="text-xs text-muted-foreground">Avg UFI Reduction (Projected)</p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.3 }}
          className="glass-card-dark p-4 text-center"
        >
          <Award className="h-5 w-5 mx-auto mb-2 text-accent" />
          <p className="text-2xl font-bold text-foreground">₹{metrics.estimatedROI}</p>
          <p className="text-xs text-muted-foreground">Saved per ₹1 Invested</p>
        </motion.div>
      </div>

      {/* Insight Cards Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Update Deserts */}
        <InsightCard
          icon={<MapPin className="h-5 w-5 text-amber-500" />}
          iconBg="bg-amber-500/10"
          title="Update Deserts"
          subtitle="High population, minimal update activity"
          count={updateDeserts.length}
          recommendation={getUpdateDesertsRecommendation()}
          priority="medium"
          delay={0.1}
        >
          <div className="space-y-2 max-h-48 overflow-y-auto custom-scrollbar pr-2">
            {updateDeserts.slice(0, 5).map((d, i) => (
              <div
                key={`${d.state}-${d.district}-${i}`}
                className="flex items-center justify-between p-2 rounded-lg bg-secondary/30"
              >
                <div className="min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">{d.district}</p>
                  <p className="text-xs text-muted-foreground">{d.state}</p>
                </div>
                <div className="text-right ml-2">
                  <p className="text-sm font-bold" style={{ color: getUFIColor(d.ufi) }}>
                    {d.ufi.toFixed(1)}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {(d.population / 1000000).toFixed(1)}M pop
                  </p>
                </div>
              </div>
            ))}
          </div>
        </InsightCard>

        {/* High Friction Zones */}
        <InsightCard
          icon={<AlertTriangle className="h-5 w-5 text-red-500" />}
          iconBg="bg-red-500/10"
          title="High Friction Zones"
          subtitle="Severe system stress requiring immediate action"
          count={highFrictionZones.length}
          recommendation={getHighFrictionRecommendation()}
          priority="critical"
          delay={0.2}
        >
          <div className="space-y-2 max-h-48 overflow-y-auto custom-scrollbar pr-2">
            {highFrictionZones.slice(0, 5).map((d, i) => (
              <div
                key={`${d.state}-${d.district}-${i}`}
                className="flex items-center justify-between p-2 rounded-lg bg-red-500/5 border border-red-500/10"
              >
                <div className="min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">{d.district}</p>
                  <p className="text-xs text-muted-foreground">{d.state}</p>
                </div>
                <div className="text-right ml-2">
                  <p className="text-sm font-bold text-red-500">
                    {d.ufi.toFixed(1)}
                  </p>
                  <Badge variant="destructive" className="text-xs">Critical</Badge>
                </div>
              </div>
            ))}
          </div>
        </InsightCard>

        {/* Digital Age Gap */}
        <InsightCard
          icon={<Users className="h-5 w-5 text-purple-500" />}
          iconBg="bg-purple-500/10"
          title="Digital Age Gap Crisis"
          subtitle="Elderly population significantly underserved"
          count={ageGapCrisis.length}
          recommendation={getAgeGapRecommendation()}
          priority="high"
          delay={0.3}
        >
          <div className="space-y-2 max-h-48 overflow-y-auto custom-scrollbar pr-2">
            {ageGapCrisis.slice(0, 5).map((d, i) => (
              <div
                key={`${d.state}-${d.district}-${i}`}
                className="flex items-center justify-between p-2 rounded-lg bg-secondary/30"
              >
                <div className="min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">{d.district}</p>
                  <p className="text-xs text-muted-foreground">{d.state}</p>
                </div>
                <div className="text-right ml-2">
                  <p className="text-sm font-bold text-purple-500">
                    {d.ageDisparity.toFixed(1)}
                  </p>
                  <p className="text-xs text-muted-foreground">Age Disparity</p>
                </div>
              </div>
            ))}
          </div>
        </InsightCard>

        {/* State Performance */}
        <InsightCard
          icon={<TrendingUp className="h-5 w-5 text-emerald-500" />}
          iconBg="bg-emerald-500/10"
          title="State Performance Rankings"
          subtitle="Comparative analysis across states"
          recommendation={getStatePerformanceRecommendation()}
          priority="low"
          delay={0.4}
        >
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={statePerformance} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                <XAxis
                  type="number"
                  domain={[0, 100]}
                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 10 }}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 10 }}
                  width={90}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                    fontSize: 12
                  }}
                  formatter={(value: number) => [value.toFixed(1), 'Mean UFI']}
                />
                <Bar
                  dataKey="ufi"
                  radius={[0, 4, 4, 0]}
                  fill="hsl(var(--primary))"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </InsightCard>
      </div>

      {/* Action CTA */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
        className="glass-card-dark p-6 text-center"
      >
        <h3 className="text-lg font-semibold text-foreground mb-2">
          Ready to Take Action?
        </h3>
        <p className="text-sm text-muted-foreground mb-4 max-w-lg mx-auto">
          Use our Predictive Analytics tool to simulate policy interventions and calculate ROI before implementation.
        </p>
        <Button
          onClick={() => setActiveTab('predictive')}
          className="gap-2"
        >
          <TrendingUp className="h-4 w-4" />
          Open Scenario Builder
          <ArrowRight className="h-4 w-4" />
        </Button>
      </motion.div>
    </div>
  );
};
