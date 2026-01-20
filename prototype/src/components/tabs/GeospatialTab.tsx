import { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDashboardStore } from '@/store/dashboardStore';
import { getUFIColor, UFIRecord } from '@/data/ufiData';
import { MapPin, X, TrendingUp, Users, AlertTriangle, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

// Simplified India map component using SVG circles for each district
const IndiaMapVisualization = ({ 
  data, 
  onDistrictClick 
}: { 
  data: UFIRecord[];
  onDistrictClick: (district: UFIRecord) => void;
}) => {
  const [hoveredDistrict, setHoveredDistrict] = useState<UFIRecord | null>(null);

  // Normalize coordinates to fit SVG viewbox
  const normalizedData = useMemo(() => {
    const minLat = 8;
    const maxLat = 37;
    const minLng = 68;
    const maxLng = 97;
    
    return data.map(d => ({
      ...d,
      x: ((d.longitude - minLng) / (maxLng - minLng)) * 100,
      y: 100 - ((d.latitude - minLat) / (maxLat - minLat)) * 100,
    }));
  }, [data]);

  return (
    <div className="relative w-full h-full">
      <svg
        viewBox="0 0 100 100"
        className="w-full h-full"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Background Grid */}
        <defs>
          <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
            <path d="M 10 0 L 0 0 0 10" fill="none" stroke="hsl(var(--border))" strokeWidth="0.1" opacity="0.3" />
          </pattern>
          <filter id="glow">
            <feGaussianBlur stdDeviation="0.5" result="coloredBlur"/>
            <feMerge> 
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        <rect width="100" height="100" fill="url(#grid)" />
        
        {/* India Outline (simplified) */}
        <path
          d="M30,15 Q45,10 55,12 Q65,8 75,15 L80,25 Q85,35 80,45 L75,55 Q72,65 75,75 L65,85 Q55,95 45,90 L35,80 Q25,75 20,65 L15,50 Q12,40 15,30 Q20,20 30,15 Z"
          fill="hsl(var(--secondary) / 0.3)"
          stroke="hsl(var(--border))"
          strokeWidth="0.3"
        />

        {/* District Points */}
        {normalizedData.map((district, index) => (
          <g key={`${district.state}-${district.district}-${index}`}>
            <motion.circle
              cx={district.x}
              cy={district.y}
              r={hoveredDistrict?.district === district.district ? 1.8 : 1.2}
              fill={getUFIColor(district.ufi)}
              opacity={0.85}
              filter={hoveredDistrict?.district === district.district ? "url(#glow)" : undefined}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 0.85 }}
              transition={{ delay: index * 0.001, duration: 0.3 }}
              className="cursor-pointer transition-all duration-200"
              onMouseEnter={() => setHoveredDistrict(district)}
              onMouseLeave={() => setHoveredDistrict(null)}
              onClick={() => onDistrictClick(district)}
            />
          </g>
        ))}
      </svg>

      {/* Hover Tooltip */}
      <AnimatePresence>
        {hoveredDistrict && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="absolute top-4 right-4 glass-card-dark p-3 rounded-lg shadow-xl max-w-xs"
          >
            <p className="font-semibold text-foreground">{hoveredDistrict.district}</p>
            <p className="text-xs text-muted-foreground">{hoveredDistrict.state}</p>
            <div className="mt-2 flex items-center gap-2">
              <span 
                className="text-lg font-bold"
                style={{ color: getUFIColor(hoveredDistrict.ufi) }}
              >
                {hoveredDistrict.ufi.toFixed(1)}
              </span>
              <Badge 
                variant="outline"
                className="text-xs"
                style={{ borderColor: getUFIColor(hoveredDistrict.ufi) }}
              >
                {hoveredDistrict.ufiCategory}
              </Badge>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// District Detail Modal
const DistrictDetailModal = ({ 
  district, 
  onClose 
}: { 
  district: UFIRecord; 
  onClose: () => void;
}) => {
  const components = [
    { name: 'Demographic Update', value: district.demoUpdateIntensity, weight: '28.8%' },
    { name: 'Update-Enrollment Ratio', value: district.updateEnrolRatio, weight: '28.1%' },
    { name: 'Age Group Disparity', value: district.ageDisparity, weight: '20.8%' },
    { name: 'Temporal Volatility', value: district.temporalVolatility, weight: '15.2%' },
    { name: 'Biometric Refresh', value: district.bioRefreshRate, weight: '7.1%' },
  ];

  const getRecommendation = (ufi: number) => {
    if (ufi >= 75) return { text: 'Urgent: Deploy mobile enrollment units immediately', priority: 'Critical' };
    if (ufi >= 50) return { text: 'Increase UIDAI center capacity by 40%', priority: 'High' };
    if (ufi >= 25) return { text: 'Monitor and plan for seasonal fluctuations', priority: 'Medium' };
    return { text: 'Continue current operations', priority: 'Low' };
  };

  const recommendation = getRecommendation(district.ufi);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="glass-card-dark w-full max-w-lg p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-foreground">{district.district}</h2>
            <p className="text-sm text-muted-foreground flex items-center gap-1">
              <MapPin className="h-3 w-3" />
              {district.state}
            </p>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* UFI Score */}
        <div className="flex items-center gap-4 mb-6 p-4 rounded-lg bg-secondary/30">
          <div 
            className="flex h-16 w-16 items-center justify-center rounded-xl text-2xl font-bold text-white"
            style={{ backgroundColor: getUFIColor(district.ufi) }}
          >
            {district.ufi.toFixed(0)}
          </div>
          <div>
            <p className="text-sm text-muted-foreground">UFI Score</p>
            <Badge 
              style={{ 
                backgroundColor: getUFIColor(district.ufi),
                color: 'white'
              }}
            >
              {district.ufiCategory}
            </Badge>
          </div>
        </div>

        {/* Component Breakdown */}
        <div className="space-y-3 mb-6">
          <h3 className="text-sm font-semibold text-foreground">Component Breakdown</h3>
          {components.map((comp) => (
            <div key={comp.name} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">{comp.name}</span>
                <span className="font-mono font-medium text-foreground">
                  {comp.value.toFixed(1)} 
                  <span className="text-muted-foreground ml-1">({comp.weight})</span>
                </span>
              </div>
              <div className="h-2 rounded-full bg-secondary overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${comp.value}%` }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                  className="h-full rounded-full"
                  style={{ backgroundColor: getUFIColor(comp.value) }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Recommendation */}
        <div className="rounded-lg border border-primary/20 bg-primary/5 p-4">
          <div className="flex items-start gap-3">
            <div className="rounded-full bg-primary/10 p-2">
              <TrendingUp className="h-4 w-4 text-primary" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <p className="text-sm font-semibold text-foreground">Recommendation</p>
                <Badge variant="outline" className="text-xs">
                  {recommendation.priority} Priority
                </Badge>
              </div>
              <p className="mt-1 text-sm text-muted-foreground">{recommendation.text}</p>
            </div>
          </div>
        </div>

        {/* Stats Footer */}
        <div className="mt-4 grid grid-cols-2 gap-4 pt-4 border-t border-border/50">
          <div className="flex items-center gap-2">
            <Users className="h-4 w-4 text-muted-foreground" />
            <div>
              <p className="text-xs text-muted-foreground">Population</p>
              <p className="text-sm font-medium text-foreground">
                {(district.population / 1000000).toFixed(2)}M
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Check className="h-4 w-4 text-muted-foreground" />
            <div>
              <p className="text-xs text-muted-foreground">Enrollments</p>
              <p className="text-sm font-medium text-foreground">
                {(district.totalEnrollments / 1000).toFixed(0)}K
              </p>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export const GeospatialTab = () => {
  const { filteredData } = useDashboardStore();
  const data = filteredData();
  const [selectedDistrict, setSelectedDistrict] = useState<UFIRecord | null>(null);

  // Stats
  const stats = useMemo(() => {
    const total = data.length;
    const critical = data.filter(d => d.ufi >= 75).length;
    const high = data.filter(d => d.ufi >= 50 && d.ufi < 75).length;
    const moderate = data.filter(d => d.ufi >= 25 && d.ufi < 50).length;
    const low = data.filter(d => d.ufi < 25).length;
    
    return { total, critical, high, moderate, low };
  }, [data]);

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      {/* Map */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="glass-card-dark p-6 lg:col-span-2"
      >
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-foreground">
              District-Level UFI Map
            </h3>
            <p className="text-sm text-muted-foreground">
              Click on any district to view details
            </p>
          </div>
        </div>

        <div className="india-map-container">
          <IndiaMapVisualization 
            data={data} 
            onDistrictClick={setSelectedDistrict}
          />
        </div>

        {/* Legend */}
        <div className="mt-4 flex items-center justify-center gap-6">
          <div className="flex items-center gap-1.5">
            <div className="h-3 w-3 rounded-full bg-emerald-500" />
            <span className="text-xs text-muted-foreground">Low (0-25)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-3 w-3 rounded-full bg-amber-500" />
            <span className="text-xs text-muted-foreground">Moderate (25-50)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-3 w-3 rounded-full bg-orange-500" />
            <span className="text-xs text-muted-foreground">High (50-75)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="h-3 w-3 rounded-full bg-red-500" />
            <span className="text-xs text-muted-foreground">Critical (75+)</span>
          </div>
        </div>
      </motion.div>

      {/* Stats Sidebar */}
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="space-y-4"
      >
        {/* Quick Stats */}
        <div className="glass-card-dark p-4">
          <h4 className="text-sm font-semibold text-foreground mb-3">Distribution Overview</h4>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-red-500" />
                <span className="text-sm text-muted-foreground">Critical</span>
              </div>
              <span className="text-sm font-semibold text-foreground">{stats.critical}</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-orange-500" />
                <span className="text-sm text-muted-foreground">High Friction</span>
              </div>
              <span className="text-sm font-semibold text-foreground">{stats.high}</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-amber-500" />
                <span className="text-sm text-muted-foreground">Moderate</span>
              </div>
              <span className="text-sm font-semibold text-foreground">{stats.moderate}</span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-emerald-500" />
                <span className="text-sm text-muted-foreground">Low Friction</span>
              </div>
              <span className="text-sm font-semibold text-foreground">{stats.low}</span>
            </div>
          </div>
        </div>

        {/* Recent High Friction */}
        <div className="glass-card-dark p-4">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle className="h-4 w-4 text-orange-500" />
            <h4 className="text-sm font-semibold text-foreground">Attention Required</h4>
          </div>
          <div className="space-y-2">
            {data
              .filter(d => d.ufi >= 60)
              .slice(0, 5)
              .map((district, i) => (
                <div 
                  key={`${district.state}-${district.district}-${i}`}
                  className="flex items-center justify-between p-2 rounded-lg bg-secondary/30 cursor-pointer hover:bg-secondary/50 transition-colors"
                  onClick={() => setSelectedDistrict(district)}
                >
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-foreground truncate">
                      {district.district}
                    </p>
                    <p className="text-xs text-muted-foreground">{district.state}</p>
                  </div>
                  <span 
                    className="text-sm font-bold ml-2"
                    style={{ color: getUFIColor(district.ufi) }}
                  >
                    {district.ufi.toFixed(1)}
                  </span>
                </div>
              ))}
          </div>
        </div>
      </motion.div>

      {/* District Detail Modal */}
      <AnimatePresence>
        {selectedDistrict && (
          <DistrictDetailModal
            district={selectedDistrict}
            onClose={() => setSelectedDistrict(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
};
