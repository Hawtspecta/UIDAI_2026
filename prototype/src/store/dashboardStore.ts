import { create } from 'zustand';
import { UFIRecord, getUFIData, generateStateStats, StateStats } from '@/data/ufiData';

export type TabType = 'overview' | 'geospatial' | 'components' | 'insights' | 'predictive';

interface FilterState {
  selectedState: string | null;
  selectedDistrict: string | null;
  ufiRange: [number, number];
  selectedCategories: string[];
  searchQuery: string;
}

interface DashboardState {
  // Data
  ufiData: UFIRecord[];
  stateStats: StateStats[];
  isLoading: boolean;

  // Navigation
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;

  // Filters
  filters: FilterState;
  setSelectedState: (state: string | null) => void;
  setSelectedDistrict: (district: string | null) => void;
  setUfiRange: (range: [number, number]) => void;
  setSelectedCategories: (categories: string[]) => void;
  setSearchQuery: (query: string) => void;
  resetFilters: () => void;

  // UI State
  isDarkMode: boolean;
  toggleDarkMode: () => void;
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;

  // Computed
  filteredData: () => UFIRecord[];

  // Actions
  initializeData: () => void;
}

const initialFilters: FilterState = {
  selectedState: null,
  selectedDistrict: null,
  ufiRange: [0, 100],
  selectedCategories: [],
  searchQuery: '',
};

export const useDashboardStore = create<DashboardState>((set, get) => ({
  // Data
  ufiData: [],
  stateStats: [],
  isLoading: true,

  // Navigation
  activeTab: 'overview',
  setActiveTab: (tab) => set({ activeTab: tab }),

  // Filters
  filters: initialFilters,
  setSelectedState: (state) => set((s) => ({
    filters: { ...s.filters, selectedState: state, selectedDistrict: null }
  })),
  setSelectedDistrict: (district) => set((s) => ({
    filters: { ...s.filters, selectedDistrict: district }
  })),
  setUfiRange: (range) => set((s) => ({
    filters: { ...s.filters, ufiRange: range }
  })),
  setSelectedCategories: (categories) => set((s) => ({
    filters: { ...s.filters, selectedCategories: categories }
  })),
  setSearchQuery: (query) => set((s) => ({
    filters: { ...s.filters, searchQuery: query }
  })),
  resetFilters: () => set({ filters: initialFilters }),

  // UI State
  isDarkMode: true,
  toggleDarkMode: () => {
    const newMode = !get().isDarkMode;
    set({ isDarkMode: newMode });
    if (newMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  },
  sidebarOpen: true,
  setSidebarOpen: (open) => set({ sidebarOpen: open }),

  // Computed
  filteredData: () => {
    const { ufiData, filters } = get();

    return ufiData.filter(record => {
      // State filter
      if (filters.selectedState && record.state !== filters.selectedState) {
        return false;
      }

      // District filter
      if (filters.selectedDistrict && record.district !== filters.selectedDistrict) {
        return false;
      }

      // UFI range filter
      if (record.ufi < filters.ufiRange[0] || record.ufi > filters.ufiRange[1]) {
        return false;
      }

      // Category filter
      if (filters.selectedCategories.length > 0 &&
        !filters.selectedCategories.includes(record.ufiCategory)) {
        return false;
      }

      // Search query
      if (filters.searchQuery) {
        const query = filters.searchQuery.toLowerCase();
        const matchesDistrict = record.district.toLowerCase().includes(query);
        const matchesState = record.state.toLowerCase().includes(query);
        if (!matchesDistrict && !matchesState) {
          return false;
        }
      }

      return true;
    });
  },

  // Actions
  initializeData: async () => {
    set({ isLoading: true });
    try {
      const data = await getUFIData();
      console.log('Data received:', data);

      if (!Array.isArray(data) || data.length === 0) {
        console.error('Invalid data format:', data);
        set({ isLoading: false });
        return;
      }

      if (!Array.isArray(data) || data.length === 0) {
        set({
          ufiData: [],
          stateStats: [],
          isLoading: false
        });
        return;
      }
      const stats = generateStateStats(data);

      set({
        ufiData: data,
        stateStats: stats,
        isLoading: false
      });
    } catch (error) {
      console.error('Failed to initialize:', error);
      set({ isLoading: false });
    }
    document.documentElement.classList.add('dark');
  },
}));
