import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class AadhaarDataLoader:
    """
    Robust data loader for UIDAI datasets located under `data/raw`.

    Features:
    - Recursive file discovery (searches subfolders)
    - Flexible filename patterns (matches substrings)
    - Autodetects CSV vs Excel files
    - Friendly error messages listing candidates when no exact match
    """

    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = Path(data_dir)
        self.enrollment_data = None
        self.biometric_data = None
        self.demographic_data = None

    def _find_file(self, substring: str, sample_size: str = None):
        """Find the first file under data_dir whose name contains `substring` and optional `sample_size`.

        Uses recursive search (rglob) so files inside subfolders are found.
        Returns Path or raises FileNotFoundError.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir.resolve()}")

        # build pattern: match all files, then filter by substring(s)
        candidates = list(self.data_dir.rglob('*'))
        # keep only files
        candidates = [p for p in candidates if p.is_file()]

        def matches(p: Path):
            name = p.name.lower()
            if substring.lower() not in name:
                return False
            if sample_size and sample_size.lower() not in name:
                return False
            return True

        matches_list = [p for p in candidates if matches(p)]

        if not matches_list:
            # helpful debugging: return a short sample of filenames that contain the substring (if any)
            contains_sub = [p for p in candidates if substring.lower() in p.name.lower()]
            if contains_sub:
                example = '\n'.join(str(p) for p in contains_sub[:10])
                raise FileNotFoundError(
                    f"No file found containing both '{substring}' and '{sample_size}' under {self.data_dir}\n"
                    f"But found files containing '{substring}':\n{example}"
                )

            # nothing at all matching substring
            all_example = '\n'.join(str(p) for p in candidates[:10])
            raise FileNotFoundError(
                f"No files found under {self.data_dir} matching '{substring}' and sample '{sample_size}'.\n"
                f"Here are up to 10 files under the directory for inspection:\n{all_example}"
            )

        # return first match (sorted for determinism)
        matches_list.sort()
        return matches_list[0]

    def _read_file(self, path: Path) -> pd.DataFrame:
        """Autodetect CSV vs Excel and read into a DataFrame."""
        suf = path.suffix.lower()
        if suf in ['.csv', '.txt']:
            return pd.read_csv(path)
        elif suf in ['.xls', '.xlsx']:
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {suf} for file {path}")

    def _standardize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Standardize column names
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Best-effort date parsing if 'date' column exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Clean state/district if present
        if 'state' in df.columns:
            df['state'] = df['state'].astype(str).str.strip().str.title()
        if 'district' in df.columns:
            df['district'] = df['district'].astype(str).str.strip().str.title()

        return df

    def load_enrollment_data(self, sample_size: str = '500000', file_path: str = None) -> pd.DataFrame:
        """Load enrollment dataset.

        If `file_path` is provided it is used directly; otherwise the loader searches for a file
        containing 'enrol' (or 'enrolment') and the sample_size substring.
        """
        if file_path:
            file = Path(file_path)
            if not file.exists():
                raise FileNotFoundError(f"Provided enrollment file does not exist: {file}")
        else:
            # try both 'enrolment' and 'enrol' keywords to be robust
            try:
                file = self._find_file('enrolment', sample_size=sample_size)
            except FileNotFoundError:
                file = self._find_file('enrol', sample_size=sample_size)

        print(f"📊 Loading enrollment data: {file.name}")
        df = self._read_file(file)
        df = self._standardize_df(df)

        self.enrollment_data = df
        print(f"✅ Loaded {len(df):,} enrollment records")
        if 'date' in df.columns:
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        return df

    def load_biometric_data(self, sample_size: str = '500000', file_path: str = None) -> pd.DataFrame:
        """Load biometric update dataset."""
        if file_path:
            file = Path(file_path)
            if not file.exists():
                raise FileNotFoundError(f"Provided biometric file does not exist: {file}")
        else:
            file = self._find_file('biometric', sample_size=sample_size)

        print(f"📊 Loading biometric data: {file.name}")
        df = self._read_file(file)
        df = self._standardize_df(df)

        self.biometric_data = df
        print(f"✅ Loaded {len(df):,} biometric update records")
        if 'date' in df.columns:
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        return df

    def load_demographic_data(self, sample_size: str = '500000', file_path: str = None) -> pd.DataFrame:
        """Load demographic update dataset."""
        if file_path:
            file = Path(file_path)
            if not file.exists():
                raise FileNotFoundError(f"Provided demographic file does not exist: {file}")
        else:
            file = self._find_file('demographic', sample_size=sample_size)

        print(f"📊 Loading demographic data: {file.name}")
        df = self._read_file(file)
        df = self._standardize_df(df)

        self.demographic_data = df
        print(f"✅ Loaded {len(df):,} demographic update records")
        if 'date' in df.columns:
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        return df

    def load_all(self, sample_size: str = '500000'):
        """Load all three datasets using the same sample_size by default."""
        print("🚀 Loading all UIDAI datasets...\n")
        self.load_enrollment_data(sample_size=sample_size)
        self.load_biometric_data(sample_size=sample_size)
        self.load_demographic_data(sample_size=sample_size)
        print("\n✅ ALL DATASETS LOADED SUCCESSFULLY!")
        return self.enrollment_data, self.biometric_data, self.demographic_data

    def get_summary_stats(self):
        """Generate summary statistics"""
        stats = {}

        if self.enrollment_data is not None:
            stats['enrollment'] = {
                'records': len(self.enrollment_data),
                'date_range': f"{self.enrollment_data['date'].min()} to {self.enrollment_data['date'].max()}" if 'date' in self.enrollment_data.columns else None,
                'states': self.enrollment_data['state'].nunique() if 'state' in self.enrollment_data.columns else None,
                'districts': self.enrollment_data['district'].nunique() if 'district' in self.enrollment_data.columns else None,
                'columns': list(self.enrollment_data.columns)
            }

        if self.biometric_data is not None:
            stats['biometric'] = {
                'records': len(self.biometric_data),
                'date_range': f"{self.biometric_data['date'].min()} to {self.biometric_data['date'].max()}" if 'date' in self.biometric_data.columns else None,
                'states': self.biometric_data['state'].nunique() if 'state' in self.biometric_data.columns else None,
                'districts': self.biometric_data['district'].nunique() if 'district' in self.biometric_data.columns else None,
                'columns': list(self.biometric_data.columns)
            }

        if self.demographic_data is not None:
            stats['demographic'] = {
                'records': len(self.demographic_data),
                'date_range': f"{self.demographic_data['date'].min()} to {self.demographic_data['date'].max()}" if 'date' in self.demographic_data.columns else None,
                'states': self.demographic_data['state'].nunique() if 'state' in self.demographic_data.columns else None,
                'districts': self.demographic_data['district'].nunique() if 'district' in self.demographic_data.columns else None,
                'columns': list(self.demographic_data.columns)
            }

        return stats


def _main():
    parser = argparse.ArgumentParser(description='Load UIDAI datasets from data/raw')
    parser.add_argument('--data-dir', default='data/raw', help='Root raw data folder')
    parser.add_argument('--sample', default='500000', help='Sample size token to match in filenames (default: 500000)')
    args = parser.parse_args()

    loader = AadhaarDataLoader(data_dir=args.data_dir)
    loader.load_all(sample_size=args.sample)

    print('\n' + '=' * 60)
    print('DATASET SUMMARY')
    print('=' * 60)

    stats = loader.get_summary_stats()
    for dataset_name, dataset_stats in stats.items():
        print(f"\n📋 {dataset_name.upper()}")
        for key, value in dataset_stats.items():
            if key != 'columns':
                print(f"   {key}: {value}")


if __name__ == '__main__':
    _main()