// UFI Data Types and Mock Data for the Dashboard

export interface UFIRecord {
  state: string;
  district: string;
  ufi: number;
  ufiCategory: 'Low Friction' | 'Moderate Friction' | 'High Friction' | 'Very High Friction';
  demoUpdateIntensity: number;
  bioRefreshRate: number;
  ageDisparity: number;
  updateEnrolRatio: number;
  temporalVolatility: number;
  totalEnrollments: number;
  population: number;
  latitude: number;
  longitude: number;
}

export interface StateStats {
  state: string;
  meanUFI: number;
  districtCount: number;
  highFrictionCount: number;
  totalPopulation: number;
}

// Helper to generate realistic UFI data
const generateUFI = (): { ufi: number; category: UFIRecord['ufiCategory'] } => {
  const ufi = Math.random() * 100;
  let category: UFIRecord['ufiCategory'];

  if (ufi < 25) category = 'Low Friction';
  else if (ufi < 50) category = 'Moderate Friction';
  else if (ufi < 75) category = 'High Friction';
  else category = 'Very High Friction';

  return { ufi: Math.round(ufi * 100) / 100, category };
};

// Indian States and their approximate coordinates
const statesData: { name: string; districts: string[]; lat: number; lng: number }[] = [
  {
    name: 'Karnataka',
    districts: ['Bengaluru Urban', 'Bengaluru Rural', 'Mysuru', 'Mangaluru', 'Hubballi-Dharwad', 'Belagavi', 'Kalaburagi', 'Ballari', 'Tumakuru', 'Shivamogga', 'Raichur', 'Bidar', 'Vijayapura', 'Davangere', 'Chitradurga', 'Hassan', 'Udupi', 'Chikkamagaluru', 'Kodagu', 'Mandya'],
    lat: 15.3173,
    lng: 75.7139
  },
  {
    name: 'Maharashtra',
    districts: ['Mumbai', 'Pune', 'Nagpur', 'Thane', 'Nashik', 'Aurangabad', 'Solapur', 'Kolhapur', 'Sangli', 'Satara', 'Ahmednagar', 'Raigad', 'Chandrapur', 'Jalgaon', 'Amravati', 'Latur', 'Nanded', 'Akola', 'Yavatmal', 'Buldana'],
    lat: 19.7515,
    lng: 75.7139
  },
  {
    name: 'Tamil Nadu',
    districts: ['Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli', 'Salem', 'Tirunelveli', 'Erode', 'Vellore', 'Thoothukudi', 'Thanjavur', 'Dindigul', 'Cuddalore', 'Kanchipuram', 'Tiruvallur', 'Tirupur', 'Villupuram', 'Virudhunagar', 'Karur', 'Namakkal', 'Sivaganga'],
    lat: 11.1271,
    lng: 78.6569
  },
  {
    name: 'Uttar Pradesh',
    districts: ['Lucknow', 'Kanpur', 'Varanasi', 'Agra', 'Prayagraj', 'Meerut', 'Ghaziabad', 'Noida', 'Bareilly', 'Aligarh', 'Moradabad', 'Gorakhpur', 'Saharanpur', 'Jhansi', 'Mathura', 'Firozabad', 'Muzaffarnagar', 'Rampur', 'Shahjahanpur', 'Farrukhabad'],
    lat: 26.8467,
    lng: 80.9462
  },
  {
    name: 'Gujarat',
    districts: ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Bhavnagar', 'Jamnagar', 'Junagadh', 'Gandhinagar', 'Anand', 'Nadiad', 'Morbi', 'Mehsana', 'Bharuch', 'Vapi', 'Navsari', 'Veraval', 'Porbandar', 'Godhra', 'Palanpur', 'Valsad'],
    lat: 22.2587,
    lng: 71.1924
  },
  {
    name: 'Rajasthan',
    districts: ['Jaipur', 'Jodhpur', 'Kota', 'Bikaner', 'Ajmer', 'Udaipur', 'Bhilwara', 'Alwar', 'Bharatpur', 'Sikar', 'Pali', 'Sri Ganganagar', 'Jhunjhunu', 'Churu', 'Tonk', 'Nagaur', 'Hanumangarh', 'Bundi', 'Chittorgarh', 'Barmer'],
    lat: 27.0238,
    lng: 74.2179
  },
  {
    name: 'West Bengal',
    districts: ['Kolkata', 'Howrah', 'Durgapur', 'Asansol', 'Siliguri', 'Bardhaman', 'Malda', 'Baharampur', 'Habra', 'Kharagpur', 'Shantipur', 'Dankuni', 'Dhulian', 'Ranaghat', 'Haldia', 'Raiganj', 'Krishnanagar', 'Nabadwip', 'Medinipur', 'Jalpaiguri'],
    lat: 22.9868,
    lng: 87.8550
  },
  {
    name: 'Madhya Pradesh',
    districts: ['Indore', 'Bhopal', 'Jabalpur', 'Gwalior', 'Ujjain', 'Sagar', 'Satna', 'Rewa', 'Dewas', 'Ratlam', 'Murwara', 'Chhindwara', 'Guna', 'Shivpuri', 'Vidisha', 'Damoh', 'Mandsaur', 'Khargone', 'Neemuch', 'Pithampur'],
    lat: 22.9734,
    lng: 78.6569
  },
  {
    name: 'Andhra Pradesh',
    districts: ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Nellore', 'Kurnool', 'Rajahmundry', 'Tirupati', 'Kadapa', 'Kakinada', 'Anantapur', 'Vizianagaram', 'Eluru', 'Ongole', 'Nandyal', 'Machilipatnam', 'Adoni', 'Tenali', 'Proddatur', 'Chittoor', 'Hindupur'],
    lat: 15.9129,
    lng: 79.7400
  },
  {
    name: 'Telangana',
    districts: ['Hyderabad', 'Warangal', 'Nizamabad', 'Karimnagar', 'Ramagundam', 'Khammam', 'Mahbubnagar', 'Nalgonda', 'Adilabad', 'Suryapet', 'Miryalaguda', 'Jagtial', 'Mancherial', 'Nirmal', 'Kamareddy', 'Kothagudem', 'Bodhan', 'Sangareddy', 'Siddipet', 'Wanaparthy'],
    lat: 18.1124,
    lng: 79.0193
  },
  {
    name: 'Kerala',
    districts: ['Thiruvananthapuram', 'Kochi', 'Kozhikode', 'Thrissur', 'Kollam', 'Kannur', 'Alappuzha', 'Kottayam', 'Palakkad', 'Malappuram', 'Kasaragod', 'Idukki', 'Pathanamthitta', 'Wayanad'],
    lat: 10.8505,
    lng: 76.2711
  },
  {
    name: 'Bihar',
    districts: ['Patna', 'Gaya', 'Bhagalpur', 'Muzaffarpur', 'Purnia', 'Darbhanga', 'Arrah', 'Begusarai', 'Katihar', 'Munger', 'Chhapra', 'Sasaram', 'Hajipur', 'Samastipur', 'Bettiah', 'Motihari', 'Siwan', 'Nawada', 'Jehanabad', 'Aurangabad'],
    lat: 25.0961,
    lng: 85.3131
  },
  {
    name: 'Odisha',
    districts: ['Bhubaneswar', 'Cuttack', 'Rourkela', 'Brahmapur', 'Sambalpur', 'Puri', 'Balasore', 'Bhadrak', 'Baripada', 'Jharsuguda', 'Jeypore', 'Bargarh', 'Paralakhemundi', 'Bhawanipatna', 'Angul', 'Dhenkanal', 'Kendujhar', 'Rayagada', 'Jajpur', 'Koraput'],
    lat: 20.9517,
    lng: 85.0985
  },
  {
    name: 'Punjab',
    districts: ['Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala', 'Bathinda', 'Mohali', 'Pathankot', 'Hoshiarpur', 'Moga', 'Firozpur', 'Kapurthala', 'Sangrur', 'Barnala', 'Faridkot', 'Fazilka', 'Gurdaspur', 'Mansa', 'Muktsar', 'Nawanshahr', 'Rupnagar'],
    lat: 31.1471,
    lng: 75.3412
  },
  {
    name: 'Haryana',
    districts: ['Gurugram', 'Faridabad', 'Panipat', 'Ambala', 'Hisar', 'Rohtak', 'Karnal', 'Sonipat', 'Yamunanagar', 'Panchkula', 'Bhiwani', 'Sirsa', 'Rewari', 'Jind', 'Kaithal', 'Kurukshetra', 'Fatehabad', 'Mahendragarh', 'Jhajjar', 'Palwal'],
    lat: 29.0588,
    lng: 76.0856
  },
  {
    name: 'Jharkhand',
    districts: ['Ranchi', 'Jamshedpur', 'Dhanbad', 'Bokaro', 'Hazaribagh', 'Deoghar', 'Giridih', 'Ramgarh', 'Phusro', 'Chaibasa', 'Medininagar', 'Chatra', 'Dumka', 'Lohardaga', 'Pakur', 'Sahebganj', 'Godda', 'Koderma', 'Gumla', 'Simdega'],
    lat: 23.6102,
    lng: 85.2799
  },
  {
    name: 'Chhattisgarh',
    districts: ['Raipur', 'Bhilai', 'Bilaspur', 'Korba', 'Durg', 'Rajnandgaon', 'Raigarh', 'Jagdalpur', 'Ambikapur', 'Dhamtari', 'Mahasamund', 'Kanker', 'Kawardha', 'Janjgir', 'Champa', 'Mungeli', 'Bemetara', 'Balod', 'Gariaband', 'Surajpur'],
    lat: 21.2787,
    lng: 81.8661
  },
  {
    name: 'Assam',
    districts: ['Guwahati', 'Silchar', 'Dibrugarh', 'Jorhat', 'Nagaon', 'Tinsukia', 'Tezpur', 'Bongaigaon', 'Dhubri', 'North Lakhimpur', 'Karimganj', 'Sivasagar', 'Goalpara', 'Barpeta', 'Mangaldoi', 'Nalbari', 'Haflong', 'Diphu', 'Golaghat', 'Kokrajhar'],
    lat: 26.2006,
    lng: 92.9376
  },
  {
    name: 'Uttarakhand',
    districts: ['Dehradun', 'Haridwar', 'Rishikesh', 'Haldwani', 'Roorkee', 'Kashipur', 'Rudrapur', 'Kotdwar', 'Srinagar', 'Pithoragarh', 'Almora', 'Nainital', 'Mussoorie', 'Bageshwar', 'Chamoli', 'Champawat', 'Pauri', 'Tehri', 'Uttarkashi', 'Udham Singh Nagar'],
    lat: 30.0668,
    lng: 79.0193
  },
  {
    name: 'Himachal Pradesh',
    districts: ['Shimla', 'Dharamshala', 'Solan', 'Mandi', 'Palampur', 'Baddi', 'Nahan', 'Hamirpur', 'Una', 'Kullu', 'Bilaspur', 'Chamba', 'Kangra', 'Kinnaur', 'Lahaul', 'Sirmaur', 'Spiti'],
    lat: 31.1048,
    lng: 77.1734
  },
  {
    name: 'Jammu and Kashmir',
    districts: ['Srinagar', 'Jammu', 'Anantnag', 'Baramulla', 'Udhampur', 'Sopore', 'Kathua', 'Kupwara', 'Pulwama', 'Poonch', 'Rajouri', 'Doda', 'Kishtwar', 'Ramban', 'Reasi', 'Samba', 'Bandipora', 'Ganderbal', 'Kulgam', 'Shopian'],
    lat: 33.7782,
    lng: 76.5762
  },
  {
    name: 'Delhi',
    districts: ['Central Delhi', 'East Delhi', 'New Delhi', 'North Delhi', 'North East Delhi', 'North West Delhi', 'Shahdara', 'South Delhi', 'South East Delhi', 'South West Delhi', 'West Delhi'],
    lat: 28.7041,
    lng: 77.1025
  },
  {
    name: 'Goa',
    districts: ['North Goa', 'South Goa'],
    lat: 15.2993,
    lng: 74.1240
  },
  {
    name: 'Tripura',
    districts: ['Agartala', 'Dharmanagar', 'Udaipur', 'Kailasahar', 'Belonia', 'Khowai', 'Ambassa', 'Sabroom'],
    lat: 23.9408,
    lng: 91.9882
  },
  {
    name: 'Meghalaya',
    districts: ['Shillong', 'Tura', 'Jowai', 'Nongstoin', 'Williamnagar', 'Baghmara', 'Nongpoh', 'Mairang', 'Resubelpara', 'Ampati', 'Khliehriat'],
    lat: 25.4670,
    lng: 91.3662
  },
  {
    name: 'Manipur',
    districts: ['Imphal East', 'Imphal West', 'Thoubal', 'Bishnupur', 'Churachandpur', 'Chandel', 'Ukhrul', 'Senapati', 'Tamenglong'],
    lat: 24.6637,
    lng: 93.9063
  },
  {
    name: 'Nagaland',
    districts: ['Kohima', 'Dimapur', 'Mokokchung', 'Tuensang', 'Mon', 'Wokha', 'Zunheboto', 'Phek', 'Peren', 'Kiphire', 'Longleng'],
    lat: 26.1584,
    lng: 94.5624
  },
  {
    name: 'Mizoram',
    districts: ['Aizawl', 'Lunglei', 'Champhai', 'Serchhip', 'Kolasib', 'Lawngtlai', 'Mamit', 'Saiha'],
    lat: 23.1645,
    lng: 92.9376
  },
  {
    name: 'Arunachal Pradesh',
    districts: ['Itanagar', 'Naharlagun', 'Pasighat', 'Namsai', 'Tezu', 'Ziro', 'Bomdila', 'Along', 'Anini', 'Changlang', 'Khonsa', 'Roing', 'Seppa', 'Tawang', 'Yingkiong', 'Daporijo'],
    lat: 28.2180,
    lng: 94.7278
  },
  {
    name: 'Sikkim',
    districts: ['Gangtok', 'Namchi', 'Gyalshing', 'Mangan', 'Rangpo', 'Singtam'],
    lat: 27.5330,
    lng: 88.5122
  },
];

// Generate UFI data for all districts
export const generateUFIData = (): UFIRecord[] => {
  const data: UFIRecord[] = [];

  statesData.forEach(state => {
    state.districts.forEach((district, index) => {
      const { ufi, category } = generateUFI();

      // Generate component scores with some correlation to UFI
      const baseScore = ufi;
      const variance = () => (Math.random() - 0.5) * 30;

      data.push({
        state: state.name,
        district,
        ufi,
        ufiCategory: category,
        demoUpdateIntensity: Math.max(0, Math.min(100, baseScore + variance())),
        bioRefreshRate: Math.max(0, Math.min(100, baseScore * 0.6 + variance())),
        ageDisparity: Math.max(0, Math.min(100, baseScore * 0.9 + variance())),
        updateEnrolRatio: Math.max(0, Math.min(100, baseScore * 0.8 + variance())),
        temporalVolatility: Math.max(0, Math.min(100, baseScore * 0.5 + variance())),
        totalEnrollments: Math.floor(50000 + Math.random() * 500000),
        population: Math.floor(100000 + Math.random() * 5000000),
        latitude: state.lat + (Math.random() - 0.5) * 2,
        longitude: state.lng + (Math.random() - 0.5) * 2,
      });
    });
  });

  return data;
};

// Generate State Statistics
export const generateStateStats = (data: UFIRecord[]): StateStats[] => {
  const stateMap = new Map<string, UFIRecord[]>();

  data.forEach(record => {
    const existing = stateMap.get(record.state) || [];
    existing.push(record);
    stateMap.set(record.state, existing);
  });

  const stats: StateStats[] = [];

  stateMap.forEach((records, state) => {
    const meanUFI = records.reduce((sum, r) => sum + r.ufi, 0) / records.length;
    const highFrictionCount = records.filter(r => r.ufi >= 50).length;
    const totalPopulation = records.reduce((sum, r) => sum + r.population, 0);

    stats.push({
      state,
      meanUFI: Math.round(meanUFI * 100) / 100,
      districtCount: records.length,
      highFrictionCount,
      totalPopulation,
    });
  });

  return stats.sort((a, b) => b.meanUFI - a.meanUFI);
};

// Helper functions
export const getUFIColor = (ufi: number): string => {
  if (ufi < 25) return 'hsl(160, 84%, 39%)'; // emerald
  if (ufi < 50) return 'hsl(45, 93%, 47%)'; // amber
  if (ufi < 75) return 'hsl(25, 95%, 53%)'; // orange
  return 'hsl(0, 84%, 60%)'; // red
};

export const getUFICategoryColor = (category: UFIRecord['ufiCategory']): string => {
  switch (category) {
    case 'Low Friction': return 'hsl(160, 84%, 39%)';
    case 'Moderate Friction': return 'hsl(45, 93%, 47%)';
    case 'High Friction': return 'hsl(25, 95%, 53%)';
    case 'Very High Friction': return 'hsl(0, 84%, 60%)';
  }
};

export const getUFICategoryClass = (category: UFIRecord['ufiCategory']): string => {
  switch (category) {
    case 'Low Friction': return 'ufi-low';
    case 'Moderate Friction': return 'ufi-moderate';
    case 'High Friction': return 'ufi-high';
    case 'Very High Friction': return 'ufi-critical';
  }
};

// Pre-generated data for consistent state
let cachedData: UFIRecord[] | null = null;

const parseCSV = async (filePath: string): Promise<Record<string, string>[]> => {
  const response = await fetch(filePath);
  const text = await response.text();
  const lines = text.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim());

  return lines.slice(1).map(line => {
    const values = line.split(',');
    const obj: Record<string, string> = {};
    headers.forEach((header, i) => {
      obj[header] = values[i]?.trim() || '';
    });
    return obj;
  });
};

export const getUFIData = async (): Promise<UFIRecord[]> => {
  if (cachedData) {
    return cachedData;
  }

  try {
    console.log('Loading CSV from /data/ufi_scores.csv');
    const csvData = await parseCSV('/data/ufi_scores.csv');
    console.log('CSV loaded:', csvData.length, 'records');

    // Create a map of state coordinates
    const stateCoords = new Map<string, { lat: number; lng: number }>();
    statesData.forEach(state => {
      stateCoords.set(state.name, { lat: state.lat, lng: state.lng });
    });

    cachedData = csvData.map(row => {
      const stateCoord = stateCoords.get(row.state) || { lat: 20, lng: 78 };
      const variance = () => (Math.random() - 0.5) * 1.5;

      return {
        state: row.state,
        district: row.district,
        ufi: parseFloat(row.UFI) || 0,
        ufiCategory: row.UFI_Category as UFIRecord['ufiCategory'],
        demoUpdateIntensity: parseFloat(row.demo_update_intensity_normalized) || 0,
        bioRefreshRate: parseFloat(row.bio_refresh_rate_normalized) || 0,
        ageDisparity: parseFloat(row.age_disparity_normalized) || 0,
        updateEnrolRatio: parseFloat(row.update_enrol_ratio_normalized) || 0,
        temporalVolatility: parseFloat(row.temporal_volatility_normalized) || 0,
        totalEnrollments: parseInt(row.total_enrollments) || 50000,
        population: parseInt(row.population) || 100000,
        latitude: stateCoord.lat + variance(),
        longitude: stateCoord.lng + variance(),
      };
    });

    return cachedData;
  } catch (error) {
    console.error('Failed to load CSV, using generated data:', error);
    cachedData = generateUFIData();
    return cachedData;
  }
};

export const refreshData = (): UFIRecord[] => {
  cachedData = generateUFIData();
  return cachedData;
};
