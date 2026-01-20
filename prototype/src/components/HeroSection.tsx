import { motion } from 'framer-motion';
import { MapPin, Users, Activity, AlertTriangle } from 'lucide-react';
import { useDashboardStore } from '@/store/dashboardStore';
import { KPICard } from './KPICard';

export const HeroSection = () => {
  const { ufiData, filteredData } = useDashboardStore();
  const data = filteredData();
  
  // Calculate KPI values
  const totalDistricts = ufiData.length;
  const meanUFI = data.length > 0 
    ? (data.reduce((sum, d) => sum + d.ufi, 0) / data.length).toFixed(2)
    : '0.00';
  const highFrictionCount = data.filter(d => d.ufi >= 50).length;
  const highFrictionPercent = totalDistricts > 0 
    ? ((highFrictionCount / totalDistricts) * 100).toFixed(1)
    : '0.0';
  
  // Calculate system health (inverse of mean UFI percentage)
  const systemHealth = Math.max(0, 100 - parseFloat(meanUFI)).toFixed(1);

  return (
    <section className="relative overflow-hidden">
      {/* Hero Background */}
      <div className="absolute inset-0 hero-gradient-dark" />
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnM+PHBhdHRlcm4gaWQ9ImdyaWQiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHBhdGggZD0iTSA2MCAwIEwgMCAwIDAgNjAiIGZpbGw9Im5vbmUiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjAzKSIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2dyaWQpIi8+PC9zdmc+')] opacity-50" />
      
      <div className="container relative px-4 py-8 md:px-6 md:py-12">
        {/* Hero Title */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8 text-center md:mb-12"
        >
          <h1 className="mb-3 text-2xl font-bold tracking-tight md:text-4xl">
            <span className="gradient-text-ufi">Update Friction Index</span>
          </h1>
          <p className="mx-auto max-w-2xl text-sm text-muted-foreground md:text-base">
            Transform Aadhaar Data into Operational Intelligence
          </p>
        </motion.div>

        {/* KPI Grid */}
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <KPICard
            title="Districts Analyzed"
            value={totalDistricts.toLocaleString()}
            subtitle="Across 30 States & UTs"
            icon={MapPin}
            color="info"
            delay={0.1}
          />
          <KPICard
            title="Mean UFI Score"
            value={meanUFI}
            subtitle="National Average"
            icon={Activity}
            trend={parseFloat(meanUFI) > 40 ? 'up' : 'down'}
            trendValue="±2.3"
            color={parseFloat(meanUFI) > 50 ? 'warning' : 'default'}
            delay={0.2}
          />
          <KPICard
            title="High Friction"
            value={highFrictionCount}
            subtitle={`${highFrictionPercent}% of districts`}
            icon={AlertTriangle}
            color="danger"
            delay={0.3}
          />
          <KPICard
            title="System Health"
            value={`${systemHealth}%`}
            subtitle="Overall Performance"
            icon={Users}
            trend={parseFloat(systemHealth) > 60 ? 'up' : 'down'}
            color="success"
            delay={0.4}
          />
        </div>
      </div>
    </section>
  );
};
