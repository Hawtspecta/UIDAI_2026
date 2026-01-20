import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '@/store/dashboardStore';
import { 
  Zap, 
  TrendingUp, 
  TrendingDown,
  Users, 
  Building2, 
  Truck,
  Megaphone,
  Calculator,
  Play,
  RotateCcw,
  Download,
  Share2,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';

const interventionTypes = [
  { id: 'centers', name: 'Add UIDAI Centers', icon: Building2, costMultiplier: 10 },
  { id: 'mobile', name: 'Mobile Enrollment Units', icon: Truck, costMultiplier: 5 },
  { id: 'awareness', name: 'Awareness Campaign', icon: Megaphone, costMultiplier: 2 },
  { id: 'digital', name: 'Digital Infrastructure', icon: Zap, costMultiplier: 8 },
];

export const PredictiveTab = () => {
  const { filteredData, stateStats } = useDashboardStore();
  const data = filteredData();

  // Simulation State
  const [intervention, setIntervention] = useState('centers');
  const [investment, setInvestment] = useState([100]);
  const [timeline, setTimeline] = useState([12]);
  const [targetStates, setTargetStates] = useState<string[]>([]);
  const [hasSimulated, setHasSimulated] = useState(false);

  // Get unique states
  const states = useMemo(() => {
    return [...new Set(data.map(d => d.state))].sort();
  }, [data]);

  // Simulation Results
  const simulationResults = useMemo(() => {
    if (!hasSimulated) return null;

    const selectedIntervention = interventionTypes.find(i => i.id === intervention);
    const cost = investment[0];
    const months = timeline[0];
    
    // Simplified simulation logic
    const baseReduction = (cost / 10) * (selectedIntervention?.costMultiplier || 1) / 10;
    const timeAdjustedReduction = baseReduction * (months / 12);
    const maxReduction = Math.min(timeAdjustedReduction, 25);
    
    // Districts improved
    const targetDistricts = targetStates.length > 0
      ? data.filter(d => targetStates.includes(d.state))
      : data.filter(d => d.ufi >= 50);
    
    const districtsImproved = Math.min(targetDistricts.length, Math.floor(cost / 5));
    
    // Population impact
    const populationServed = targetDistricts
      .slice(0, districtsImproved)
      .reduce((sum, d) => sum + d.population, 0);

    // ROI calculation
    const estimatedSavings = populationServed * 50; // ₹50 per person in avoided costs
    const roi = estimatedSavings / (cost * 10000000);

    // Monthly progression
    const progression = Array.from({ length: months }, (_, i) => {
      const progress = (i + 1) / months;
      const currentReduction = maxReduction * Math.pow(progress, 0.7);
      const avgUFI = data.reduce((sum, d) => sum + d.ufi, 0) / data.length;
      
      return {
        month: `M${i + 1}`,
        before: avgUFI,
        after: Math.max(avgUFI - currentReduction, 0),
        reduction: currentReduction,
      };
    });

    return {
      ufiReduction: maxReduction,
      districtsImproved,
      populationServed,
      roi,
      progression,
      cost: cost,
      intervention: selectedIntervention,
    };
  }, [hasSimulated, intervention, investment, timeline, targetStates, data]);

  const runSimulation = () => {
    setHasSimulated(true);
  };

  const resetSimulation = () => {
    setHasSimulated(false);
    setIntervention('centers');
    setInvestment([100]);
    setTimeline([12]);
    setTargetStates([]);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="glass-card-dark p-6"
      >
        <div className="flex items-center gap-3 mb-2">
          <div className="rounded-xl bg-primary/10 p-3">
            <Calculator className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-foreground">Policy Impact Simulator</h2>
            <p className="text-sm text-muted-foreground">
              Model interventions and predict UFI changes before implementation
            </p>
          </div>
        </div>
      </motion.div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Scenario Builder */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="glass-card-dark p-6 lg:col-span-1"
        >
          <h3 className="text-lg font-semibold text-foreground mb-4">Scenario Builder</h3>
          
          <div className="space-y-6">
            {/* Intervention Type */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Intervention Type</label>
              <Select value={intervention} onValueChange={setIntervention}>
                <SelectTrigger className="bg-secondary/50">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {interventionTypes.map(type => (
                    <SelectItem key={type.id} value={type.id}>
                      <div className="flex items-center gap-2">
                        <type.icon className="h-4 w-4 text-primary" />
                        {type.name}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Investment */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-foreground">Investment</label>
                <span className="text-sm font-mono text-primary">₹{investment[0]} Cr</span>
              </div>
              <Slider
                value={investment}
                onValueChange={setInvestment}
                min={10}
                max={500}
                step={10}
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>₹10 Cr</span>
                <span>₹500 Cr</span>
              </div>
            </div>

            {/* Timeline */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-foreground">Timeline</label>
                <span className="text-sm font-mono text-accent">{timeline[0]} months</span>
              </div>
              <Slider
                value={timeline}
                onValueChange={setTimeline}
                min={3}
                max={24}
                step={3}
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>3 months</span>
                <span>24 months</span>
              </div>
            </div>

            {/* Target States */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">
                Target States <span className="text-muted-foreground">(optional)</span>
              </label>
              <Select 
                value={targetStates[0] || 'all'} 
                onValueChange={(v) => setTargetStates(v === 'all' ? [] : [v])}
              >
                <SelectTrigger className="bg-secondary/50">
                  <SelectValue placeholder="All High-Friction States" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All High-Friction States</SelectItem>
                  {states.slice(0, 15).map(state => (
                    <SelectItem key={state} value={state}>{state}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-2 pt-4">
              <Button 
                onClick={runSimulation}
                className="flex-1 gap-2"
              >
                <Play className="h-4 w-4" />
                Run Simulation
              </Button>
              {hasSimulated && (
                <Button 
                  variant="outline" 
                  size="icon"
                  onClick={resetSimulation}
                >
                  <RotateCcw className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>
        </motion.div>

        {/* Results */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="glass-card-dark p-6 lg:col-span-2"
        >
          {!hasSimulated ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center">
              <div className="rounded-full bg-secondary p-4 mb-4">
                <Zap className="h-8 w-8 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">
                Configure Your Scenario
              </h3>
              <p className="text-sm text-muted-foreground max-w-sm">
                Select intervention type, set investment amount and timeline, then run the simulation to see projected impact.
              </p>
            </div>
          ) : simulationResults && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-foreground">Projected Impact</h3>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" className="gap-1.5">
                    <Download className="h-3.5 w-3.5" />
                    Export
                  </Button>
                  <Button variant="outline" size="sm" className="gap-1.5">
                    <Share2 className="h-3.5 w-3.5" />
                    Share
                  </Button>
                </div>
              </div>

              {/* Impact Metrics */}
              <div className="grid gap-4 sm:grid-cols-4">
                <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/20 p-4 text-center">
                  <TrendingDown className="h-5 w-5 mx-auto mb-2 text-emerald-500" />
                  <p className="text-2xl font-bold text-emerald-500">
                    -{simulationResults.ufiReduction.toFixed(1)}
                  </p>
                  <p className="text-xs text-muted-foreground">UFI Points Reduced</p>
                </div>
                <div className="rounded-lg bg-primary/10 border border-primary/20 p-4 text-center">
                  <CheckCircle2 className="h-5 w-5 mx-auto mb-2 text-primary" />
                  <p className="text-2xl font-bold text-primary">
                    {simulationResults.districtsImproved}
                  </p>
                  <p className="text-xs text-muted-foreground">Districts Improved</p>
                </div>
                <div className="rounded-lg bg-accent/10 border border-accent/20 p-4 text-center">
                  <Users className="h-5 w-5 mx-auto mb-2 text-accent" />
                  <p className="text-2xl font-bold text-accent">
                    {(simulationResults.populationServed / 1000000).toFixed(1)}M
                  </p>
                  <p className="text-xs text-muted-foreground">Population Served</p>
                </div>
                <div className="rounded-lg bg-amber-500/10 border border-amber-500/20 p-4 text-center">
                  <TrendingUp className="h-5 w-5 mx-auto mb-2 text-amber-500" />
                  <p className="text-2xl font-bold text-amber-500">
                    ₹{simulationResults.roi.toFixed(1)}
                  </p>
                  <p className="text-xs text-muted-foreground">ROI per ₹1 Invested</p>
                </div>
              </div>

              {/* Progression Chart */}
              <div>
                <h4 className="text-sm font-semibold text-foreground mb-3">UFI Progression Over Time</h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={simulationResults.progression}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                      <XAxis 
                        dataKey="month" 
                        tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                      />
                      <YAxis 
                        tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                        domain={[0, 100]}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'hsl(var(--card))',
                          border: '1px solid hsl(var(--border))',
                          borderRadius: '8px',
                          fontSize: 12
                        }}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="before" 
                        name="Current UFI"
                        stroke="hsl(var(--muted-foreground))"
                        fill="hsl(var(--muted))"
                        strokeDasharray="5 5"
                      />
                      <Area 
                        type="monotone" 
                        dataKey="after" 
                        name="Projected UFI"
                        stroke="hsl(var(--primary))"
                        fill="hsl(var(--primary) / 0.2)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Summary */}
              <div className="rounded-lg bg-primary/5 border border-primary/10 p-4">
                <div className="flex items-start gap-3">
                  <AlertCircle className="h-5 w-5 text-primary mt-0.5" />
                  <div>
                    <h4 className="text-sm font-semibold text-foreground mb-1">Simulation Summary</h4>
                    <p className="text-sm text-muted-foreground">
                      With <span className="text-primary font-medium">₹{simulationResults.cost} Cr</span> invested in{' '}
                      <span className="text-primary font-medium">{simulationResults.intervention?.name}</span> over{' '}
                      <span className="text-accent font-medium">{timeline[0]} months</span>, we project a{' '}
                      <span className="text-emerald-500 font-bold">{simulationResults.ufiReduction.toFixed(1)} point reduction</span> in 
                      average UFI across {simulationResults.districtsImproved} districts, 
                      serving {(simulationResults.populationServed / 1000000).toFixed(1)}M citizens.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </div>

      {/* Disclaimer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="text-center text-xs text-muted-foreground"
      >
        <p>
          * Projections are based on historical data patterns and simplified models. 
          Actual results may vary based on implementation quality and external factors.
        </p>
      </motion.div>
    </div>
  );
};
