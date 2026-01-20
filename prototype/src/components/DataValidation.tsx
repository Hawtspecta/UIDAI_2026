import { AlertTriangle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

export const DataValidation = ({ data }: { data: any[] }) => {
  if (!data || data.length === 0) {
    return (
      <div className="flex min-h-[400px] items-center justify-center">
        <Alert variant="destructive" className="max-w-lg">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>No Data Available</AlertTitle>
          <AlertDescription>
            Unable to load UFI data. Please ensure CSV files are in the /public/data/ folder.
          </AlertDescription>
        </Alert>
      </div>
    );
  }
  return null;
};