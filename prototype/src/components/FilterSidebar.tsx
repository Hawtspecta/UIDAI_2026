import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Search, Filter, X, RotateCcw } from 'lucide-react';
import { useDashboardStore } from '@/store/dashboardStore';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

const categories = [
  { id: 'Low Friction', color: 'bg-emerald-500' },
  { id: 'Moderate Friction', color: 'bg-amber-500' },
  { id: 'High Friction', color: 'bg-orange-500' },
  { id: 'Very High Friction', color: 'bg-red-500' },
];

export const FilterSidebar = () => {
  const {
    ufiData,
    filters,
    setSelectedState,
    setSearchQuery,
    setUfiRange,
    setSelectedCategories,
    resetFilters,
    sidebarOpen,
  } = useDashboardStore();

  // Get unique states
  const states = useMemo(() => {
    const uniqueStates = [...new Set(ufiData.map(d => d.state))];
    return uniqueStates.sort();
  }, [ufiData]);

  // Toggle category
  const toggleCategory = (categoryId: string) => {
    const current = filters.selectedCategories;
    if (current.includes(categoryId)) {
      setSelectedCategories(current.filter(c => c !== categoryId));
    } else {
      setSelectedCategories([...current, categoryId]);
    }
  };

  // Check if any filters are active
  const hasActiveFilters = 
    filters.selectedState !== null ||
    filters.searchQuery !== '' ||
    filters.ufiRange[0] !== 0 ||
    filters.ufiRange[1] !== 100 ||
    filters.selectedCategories.length > 0;

  return (
    <motion.aside
      initial={{ x: -20, opacity: 0 }}
      animate={{ 
        x: sidebarOpen ? 0 : -300,
        opacity: sidebarOpen ? 1 : 0 
      }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={cn(
        'glass-card-dark w-72 shrink-0 overflow-hidden border-r border-border/40 p-4',
        !sidebarOpen && 'hidden md:block'
      )}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-primary" />
          <h3 className="font-semibold text-foreground">Filters</h3>
        </div>
        {hasActiveFilters && (
          <Button
            variant="ghost"
            size="sm"
            onClick={resetFilters}
            className="h-7 gap-1 px-2 text-xs text-muted-foreground hover:text-foreground"
          >
            <RotateCcw className="h-3 w-3" />
            Reset
          </Button>
        )}
      </div>

      <div className="space-y-6">
        {/* Search */}
        <div className="space-y-2">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Search Districts
          </label>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search district or state..."
              value={filters.searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 bg-secondary/50 border-border/50"
            />
            {filters.searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* State Filter */}
        <div className="space-y-2">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            State / UT
          </label>
          <Select
            value={filters.selectedState || 'all'}
            onValueChange={(value) => setSelectedState(value === 'all' ? null : value)}
          >
            <SelectTrigger className="bg-secondary/50 border-border/50">
              <SelectValue placeholder="All States" />
            </SelectTrigger>
            <SelectContent className="max-h-64">
              <SelectItem value="all">All States</SelectItem>
              {states.map(state => (
                <SelectItem key={state} value={state}>{state}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* UFI Range */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              UFI Range
            </label>
            <span className="text-xs font-mono text-muted-foreground">
              {filters.ufiRange[0]} - {filters.ufiRange[1]}
            </span>
          </div>
          
          {/* Gradient Bar */}
          <div className="relative h-2 rounded-full bg-gradient-to-r from-emerald-500 via-amber-500 via-orange-500 to-red-500">
            <div 
              className="absolute inset-y-0 left-0 bg-background/80 rounded-l-full"
              style={{ width: `${filters.ufiRange[0]}%` }}
            />
            <div 
              className="absolute inset-y-0 right-0 bg-background/80 rounded-r-full"
              style={{ width: `${100 - filters.ufiRange[1]}%` }}
            />
          </div>
          
          <Slider
            value={filters.ufiRange}
            onValueChange={(value) => setUfiRange(value as [number, number])}
            min={0}
            max={100}
            step={1}
            className="mt-2"
          />
        </div>

        {/* Category Filter */}
        <div className="space-y-3">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            UFI Categories
          </label>
          <div className="flex flex-wrap gap-2">
            {categories.map(cat => (
              <Badge
                key={cat.id}
                variant={filters.selectedCategories.includes(cat.id) ? 'default' : 'outline'}
                className={cn(
                  'cursor-pointer transition-all hover:scale-105',
                  filters.selectedCategories.includes(cat.id) && 'ring-2 ring-offset-2 ring-offset-background ring-primary/50'
                )}
                onClick={() => toggleCategory(cat.id)}
              >
                <span className={cn('mr-1.5 h-2 w-2 rounded-full', cat.color)} />
                {cat.id.replace(' Friction', '')}
              </Badge>
            ))}
          </div>
        </div>

        {/* Active Filters Summary */}
        {hasActiveFilters && (
          <div className="rounded-lg bg-primary/5 p-3 border border-primary/10">
            <p className="text-xs text-muted-foreground mb-2">Active Filters:</p>
            <div className="flex flex-wrap gap-1">
              {filters.selectedState && (
                <Badge variant="secondary" className="text-xs">
                  {filters.selectedState}
                </Badge>
              )}
              {filters.searchQuery && (
                <Badge variant="secondary" className="text-xs">
                  "{filters.searchQuery}"
                </Badge>
              )}
              {(filters.ufiRange[0] !== 0 || filters.ufiRange[1] !== 100) && (
                <Badge variant="secondary" className="text-xs">
                  UFI: {filters.ufiRange[0]}-{filters.ufiRange[1]}
                </Badge>
              )}
              {filters.selectedCategories.map(cat => (
                <Badge key={cat} variant="secondary" className="text-xs">
                  {cat.replace(' Friction', '')}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </div>
    </motion.aside>
  );
};
